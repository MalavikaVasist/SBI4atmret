#!/usr/bin/env python
"""
Config-driven dataset generation script.

Reads simulator configs from YAML, samples from the prior,
runs each simulator, filters invalid spectra, and stores
as H5 splits (train/valid/test).

Usage:
    python generate_dataset.py --config experiments/config_MiriGeminiHST_cloudfree.yaml
    python generate_dataset.py --config experiments/config_MiriGeminiHST_cloudfree.yaml --batch-index 5
"""

import argparse
import sys
import os
from pathlib import Path
from itertools import starmap

# petitRADTRANS requires this before import
os.environ['pRT_input_data_path'] = os.path.join(
    os.environ.get('HOME', ''), 'pRT/input_data_v2.4.9/input_data'
)

import numpy as np
import torch
import yaml
from tqdm import tqdm

from pydantic import ValidationError
from lampe.data import H5Dataset
from zuko.distributions import BoxUniform

from sbi4atmret.config.configs import BaseConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Generate training dataset from config.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override output directory. Defaults to config dataset_path.")
    parser.add_argument("--n-array", type=int, default=300,
                        help="Number of array files to generate.")
    parser.add_argument("--batch-size", type=int, default=4096,
                        help="Samples per batch file.")
    parser.add_argument("--batch-index", type=int, default=None,
                        help="Generate a single batch (for parallel jobs). If None, generates all.")
    parser.add_argument("--aggregate-only", action="store_true",
                        help="Only aggregate existing batches into train/valid/test splits.")
    parser.add_argument("--parallel", action="store_true",
                        help="Submit as SLURM array job via dawgz (parallel generation).")
    parser.add_argument("--conda-env", type=str, default="test",
                        help="Conda environment name for SLURM jobs.")
    parser.add_argument("--cpus", type=int, default=1, help="CPUs per SLURM job.")
    parser.add_argument("--gpus", type=int, default=1, help="GPUs per SLURM job.")
    parser.add_argument("--ram", type=str, default="16GB", help="RAM per SLURM job.")
    parser.add_argument("--time", type=str, default="2-00:00:00", help="Wall time per SLURM job.")
    return parser.parse_args()


def filter_nan_inf(theta, x):
    """Remove samples with NaN or Inf in the spectrum."""
    mask = ~torch.any(torch.isnan(x), dim=-1)
    theta, x = theta[mask], x[mask]
    mask2 = ~torch.any(torch.isinf(x), dim=-1)
    return theta[mask2], x[mask2]


def generate_batch_all_simulators(
    simulators: dict,
    prior,
    theta_mapper,
    batch_size: int,
) -> dict:
    """
    Sample theta ONCE from the merged prior, split per-simulator,
    and simulate spectra for all instruments.

    The atmosphere (shared parameters) remains the same across instruments.
    Only instrument-specific parameters (e.g., b-factors) differ.

    Args:
        simulators: {sim_name: Simulator} dict
        prior: prior over the merged (posterior) parameter space
        theta_mapper: BaseThetaMapper with split_theta method
        batch_size: number of samples

    Returns:
        dict {sim_name: (theta_inst, x_inst)} — per-simulator theta and spectra
    """
    # Sample from the MERGED prior (all parameters at once)
    theta_merged = prior.sample((batch_size,))

    # Split into per-simulator parameter vectors
    theta_dict = theta_mapper.split_theta(theta_merged)

    results = {}

    for sim_name, simulator in simulators.items():
        theta_inst = theta_dict[sim_name]  # (B, D_inst)

        spectra = []
        valid_thetas = []

        for i in range(theta_inst.shape[0]):
            theta_i = theta_inst[i].numpy()
            try:
                output = simulator(theta_i)
                spectra.append(torch.from_numpy(output.spectrum).float())
                valid_thetas.append(theta_inst[i])
            except Exception:
                continue

        if not spectra:
            results[sim_name] = (
                torch.zeros(0, theta_inst.shape[-1]),
                torch.zeros(0, 1),
            )
            continue

        theta_out = torch.stack(valid_thetas)
        x_out = torch.stack(spectra)

        theta_out, x_out = filter_nan_inf(theta_out, x_out)
        results[sim_name] = (theta_out, x_out)

    return results


def generate_batch_file(
    config: BaseConfig,
    simulators: dict,
    prior,
    theta_mapper,
    output_dir: Path,
    batch_index: int,
    batch_size: int,
):
    """Generate a single batch file for ALL simulators (shared atmosphere).

    Samples theta once, splits per-simulator, simulates all instruments,
    saves one H5 file per simulator.
    """

    # Check if all files already exist
    all_exist = all(
        (output_dir / sim_name / f"samples_{batch_index:06d}.h5").exists()
        for sim_name in simulators.keys()
    )
    if all_exist:
        print(f"  Skipping batch {batch_index} (all files exist)")
        return

    # Generate for all simulators at once (shared atmosphere)
    results = generate_batch_all_simulators(
        simulators, prior, theta_mapper, batch_size
    )

    # Save per-simulator
    for sim_name, (theta, x) in results.items():
        output_path = output_dir / sim_name
        output_path.mkdir(parents=True, exist_ok=True)
        file_path = output_path / f"samples_{batch_index:06d}.h5"

        if file_path.exists():
            continue

        if len(theta) == 0:
            print(f"  WARNING: No valid samples for {sim_name} batch {batch_index}")
            continue

        def yield_batch(t=theta, xx=x):
            yield t, xx

        H5Dataset.store(yield_batch(), file_path, size=len(theta))

    n_valid = min(len(v[0]) for v in results.values()) if results else 0
    print(f"  Batch {batch_index}: {n_valid} valid samples saved")


def aggregate_splits(output_dir: Path, sim_name: str):
    """Aggregate batch files into train/valid/test splits."""

    sim_dir = output_dir / sim_name
    files = sorted(sim_dir.glob("samples_*.h5"))

    if not files:
        print(f"  No batch files found in {sim_dir}")
        return

    n = len(files)
    i = int(0.9 * n)
    j = int(0.99 * n)

    splits = {
        "train": files[:i],
        "valid": files[i:j],
        "test": files[j:],
    }

    for split_name, split_files in splits.items():
        if not split_files:
            print(f"  No files for split '{split_name}'")
            continue

        split_path = sim_dir / f"{split_name}.h5"

        if split_path.exists():
            print(f"  Skipping {split_path} (already exists)")
            continue

        dataset = H5Dataset(*split_files, batch_size=4096 * 3)

        H5Dataset.store(dataset, split_path, size=len(dataset))
        print(f"  Aggregated {len(split_files)} files → {split_path} ({len(dataset)} samples)")


def main():
    args = parse_args()

    # Load config
    with open(args.config, "r") as f:
        config_dict = yaml.safe_load(f)

    try:
        config = BaseConfig(**config_dict)
    except ValidationError as exc:
        print(f"Config validation failed: {exc}")
        sys.exit(1)

    # Build prior
    prior = config.build_prior()
    print(f"Prior: {config.get_no_of_params()} parameters")

    # Build simulators
    simulators = config.build_simulators()
    print(f"Simulators: {list(simulators.keys())}")

    # Build theta mapper (knows how merged params map to per-simulator params)
    from sbi4atmret.datasets.theta_mapper.thetamapperbase import BaseThetaMapper

    # Create a minimal domain-like object for theta mapper
    class _MinimalDomain:
        def __init__(self, simulator_dict):
            self.simulator_dict = simulator_dict

    theta_mapper = BaseThetaMapper(
        domain=_MinimalDomain(simulators),
        posterior_param_names=config.get_parameter_names(),
    )
    print(f"Theta mapper: {len(theta_mapper.simulator_names)} simulators, "
          f"{theta_mapper.n_total} merged params")

    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {output_dir}")

    # Aggregate only
    if args.aggregate_only:
        for sim_name in simulators.keys():
            print(f"\nAggregating {sim_name}...")
            aggregate_splits(output_dir, sim_name)
        print("\nDone.")
        return

    # Parallel mode: submit SLURM array job via dawgz
    if args.parallel:
        from dawgz import job, schedule

        @job(array=args.array, cpus=args.cpus, gpus=args.gpus, ram=args.ram, time=args.time)
        def generate_batch_job(batch_index: int):
            generate_batch_file(
                config, simulators, prior, theta_mapper,
                output_dir, batch_index, args.batch_size,
            )

        schedule(
            generate_batch_job,
            name="Dataset generation",
            backend="slurm",
            env=[
                "source ~/.bashrc",
                f"conda activate {args.conda_env}",
            ],
        )
        print(f"\nSubmitted {args.array} SLURM jobs.")
        print("After all jobs complete, run with --aggregate-only to combine.")
        return

    # Single batch mode (for manual SLURM array jobs via --batch-index)
    if args.batch_index is not None:
        print(f"\nGenerating batch {args.batch_index} for all simulators...")
        generate_batch_file(
            config, simulators, prior, theta_mapper,
            output_dir, args.batch_index, args.batch_size,
        )
    else:
        print("Use --parallel to submit SLURM jobs, or --batch-index N for a single batch.")
        sys.exit(1)

    print("\nDataset generation complete.")


if __name__ == "__main__":
    main()


