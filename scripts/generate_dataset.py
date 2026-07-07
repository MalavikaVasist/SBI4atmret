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
    parser.add_argument("--n-batches", type=int, default=300,
                        help="Number of batch files to generate.")
    parser.add_argument("--batch-size", type=int, default=4096,
                        help="Samples per batch file.")
    parser.add_argument("--batch-index", type=int, default=None,
                        help="Generate a single batch (for parallel jobs). If None, generates all.")
    parser.add_argument("--aggregate-only", action="store_true",
                        help="Only aggregate existing batches into train/valid/test splits.")
    parser.add_argument("--parallel", action="store_true",
                        help="Submit as SLURM array job via dawgz (parallel generation).")
    parser.add_argument("--conda-env", type=str, default="WISEJ1828",
                        help="Conda environment name for SLURM jobs.")
    parser.add_argument("--cpus", type=int, default=1, help="CPUs per SLURM job.")
    parser.add_argument("--ram", type=str, default="16GB", help="RAM per SLURM job.")
    parser.add_argument("--time", type=str, default="2-00:00:00", help="Wall time per SLURM job.")
    return parser.parse_args()


def filter_nan_inf(theta, x):
    """Remove samples with NaN or Inf in the spectrum."""
    mask = ~torch.any(torch.isnan(x), dim=-1)
    theta, x = theta[mask], x[mask]
    mask2 = ~torch.any(torch.isinf(x), dim=-1)
    return theta[mask2], x[mask2]


def generate_single_batch(
    simulator,
    prior,
    batch_size: int,
):
    """
    Sample theta from prior, run simulator, return (theta, x).

    Args:
        simulator: callable that takes numpy theta (D,) and returns SimulatorOutput
        prior: distribution with .sample() method
        batch_size: number of samples

    Returns:
        (theta, x) tensors after filtering
    """
    theta_samples = prior.sample((batch_size,))

    spectra = []
    valid_thetas = []

    for i in range(batch_size):
        theta_i = theta_samples[i].numpy()
        try:
            output = simulator(theta_i)
            spectra.append(torch.from_numpy(output.spectrum).float())
            valid_thetas.append(theta_samples[i])
        except Exception:
            continue

    if not spectra:
        return torch.zeros(0, theta_samples.shape[-1]), torch.zeros(0, 1)

    theta = torch.stack(valid_thetas)
    x = torch.stack(spectra)

    theta, x = filter_nan_inf(theta, x)

    return theta, x


def generate_batch_file(
    config: BaseConfig,
    sim_name: str,
    simulator,
    prior,
    output_dir: Path,
    batch_index: int,
    batch_size: int,
):
    """Generate a single batch file for one simulator."""

    output_path = output_dir / sim_name
    output_path.mkdir(parents=True, exist_ok=True)

    file_path = output_path / f"samples_{batch_index:06d}.h5"

    if file_path.exists():
        print(f"  Skipping {file_path} (already exists)")
        return

    theta, x = generate_single_batch(simulator, prior, batch_size)

    if len(theta) == 0:
        print(f"  WARNING: No valid samples for batch {batch_index}")
        return

    # Store as H5
    def yield_batch():
        yield theta, x

    H5Dataset.store(yield_batch(), file_path, size=len(theta))
    print(f"  Saved {len(theta)} samples to {file_path}")


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

    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        # Use the first dataset path's parent as output base
        first_path = next(iter(config.dataset_config.dataset_path.values())).path
        output_dir = Path(first_path).parent
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

        @job(array=args.n_batches, cpus=args.cpus, ram=args.ram, time=args.time)
        def generate_batch_job(batch_index: int):
            for sim_name, simulator in simulators.items():
                generate_batch_file(
                    config, sim_name, simulator, prior,
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
        print(f"\nSubmitted {args.n_batches} SLURM jobs.")
        print("After all jobs complete, run with --aggregate-only to combine.")
        return

    # Single batch mode (for manual SLURM array jobs via --batch-index)
    if args.batch_index is not None:
        for sim_name, simulator in simulators.items():
            print(f"\nGenerating batch {args.batch_index} for {sim_name}...")
            generate_batch_file(
                config, sim_name, simulator, prior,
                output_dir, args.batch_index, args.batch_size,
            )
    else:
        print("Use --parallel to submit SLURM jobs, or --batch-index N for a single batch.")
        sys.exit(1)

    print("\nDataset generation complete.")


if __name__ == "__main__":
    main()


