#!/usr/bin/env python
"""
Config-driven dataset generation script.

Reads simulator configs from YAML, samples from the prior,
runs each simulator, filters invalid spectra, and stores
as H5 splits (train/valid/test).

Usage:
    python generate_dataset.py --config experiments/config_MiriGeminiHST_cloudfree.yaml
    python generate_dataset.py --config experiments/config_MiriGeminiHST_cloudfree.yaml --array-index 5
"""

import argparse
import sys
import os
from pathlib import Path
from itertools import starmap

# petitRADTRANS requires this before import
os.environ['pRT_input_data_path'] = '/media/mvasist/Elements/PhDprojects/scratch/input_data_v2.4.9/input_data'

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
    parser.add_argument("--array-index", type=int, default=None,
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
    parser.add_argument("--input-dir", type=str, default=None,
                        help="Path to folder with existing samples_*.h5 files (e.g., cloudfree). "
                             "If provided, loads theta from these, appends new cloud params "
                             "from the config's prior, and resimulates with the new simulators. "
                             "Original data is NOT modified.")
    return parser.parse_args()


def filter_nan_inf(theta, x):
    """Remove samples with NaN or Inf in the spectrum."""
    mask = ~torch.any(torch.isnan(x), dim=-1)
    theta, x = theta[mask], x[mask]
    mask2 = ~torch.any(torch.isinf(x), dim=-1)
    return theta[mask2], x[mask2]


import json


def _save_metadata(output_path: Path, sim_name: str, simulator):
    """Save a metadata.json with parameter names alongside the H5 files."""
    meta_path = output_path / "metadata.json"
    if meta_path.exists():
        return  # already saved

    metadata = {
        "sim_name": sim_name,
        "param_names": simulator.names,
        "n_params": len(simulator.names),
        "wavelength_range": [float(simulator.a), float(simulator.b)],
    }

    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)


def _load_source_param_names(input_dir: Path) -> list:
    """Load parameter names from metadata.json in the source directory."""
    meta_path = input_dir / "metadata.json"

    if not meta_path.exists():
        raise FileNotFoundError(
            f"No metadata.json found in {input_dir}. "
            f"Cannot determine source parameter names. "
            f"Re-generate the source dataset with the current script to create it."
        )

    with open(meta_path) as f:
        metadata = json.load(f)

    return metadata["param_names"]


def resimulate_batch_from_existing(
    source_file: Path,
    simulators: dict,
    theta_mapper,
    source_param_names: list,
    target_param_names: list,
    target_prior,
    output_dir: Path,
    array_index: int,
):
    """
    Load existing theta (e.g., cloudfree), append new parameters (e.g., clouds),
    and resimulate with the new (e.g., cloudy) simulators.

    The source theta is augmented with extra parameters sampled from the
    target prior. Only parameters NOT present in the source are sampled fresh.

    Args:
        source_file: H5 file with existing (theta, x) from simpler model
        simulators: {sim_name: Simulator} for the new (complex) model
        theta_mapper: BaseThetaMapper for the new model
        source_param_names: list of param names in the source theta columns
        target_param_names: list of param names for the target model's merged theta
        target_prior: prior over the target model's full parameter space
        output_dir: where to save the new H5 files
        array_index: index for naming
    """
    from lampe.data import H5Dataset

    # Load source data
    source_ds = H5Dataset(source_file, batch_size=len(H5Dataset(source_file)))
    for theta_source, x_source in source_ds:
        break  # single batch = all data

    B = theta_source.shape[0]

    # Map source param names to indices in source theta
    source_name_to_idx = {name: i for i, name in enumerate(source_param_names)}

    # Build the target merged theta:
    # - For params present in source: copy from source
    # - For params NOT in source: sample from target prior
    target_sample = target_prior.sample((B,))  # full target prior sample
    theta_merged = target_sample.clone()

    for target_idx, name in enumerate(target_param_names):
        if name in source_name_to_idx:
            source_idx = source_name_to_idx[name]
            theta_merged[:, target_idx] = theta_source[:, source_idx]

    # Split to per-simulator and simulate
    theta_dict = theta_mapper.split_theta(theta_merged)

    for sim_name, simulator in simulators.items():
        out_path = output_dir / sim_name
        out_path.mkdir(parents=True, exist_ok=True)
        file_path = out_path / f"samples_{array_index:06d}.h5"

        if file_path.exists():
            continue

        theta_inst = theta_dict[sim_name]
        spectra = []
        valid_thetas = []

        for i in range(theta_inst.shape[0]):
            try:
                output = simulator(theta_inst[i].numpy())
                spectra.append(torch.from_numpy(output.spectrum).float())
                valid_thetas.append(theta_inst[i])
            except Exception:
                continue

        if not spectra:
            print(f"  WARNING: No valid samples for {sim_name} batch {array_index}")
            continue

        theta_out = torch.stack(valid_thetas)
        x_out = torch.stack(spectra)
        theta_out, x_out = filter_nan_inf(theta_out, x_out)

        def yield_batch(t=theta_out, xx=x_out):
            yield t, xx

        H5Dataset.store(yield_batch(), file_path, size=len(theta_out))

    print(f"  Resimulated batch {array_index}: {B} source samples → new spectra")


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
    array_index: int,
    batch_size: int,
):
    """Generate a single batch file for ALL simulators (shared atmosphere).

    Samples theta once, splits per-simulator, simulates all instruments,
    saves one H5 file per simulator.
    """

    # Check if all files already exist
    all_exist = all(
        (output_dir / sim_name / f"samples_{array_index:06d}.h5").exists()
        for sim_name in simulators.keys()
    )
    if all_exist:
        print(f"  Skipping batch {array_index} (all files exist)")
        return

    # Generate for all simulators at once (shared atmosphere)
    results = generate_batch_all_simulators(
        simulators, prior, theta_mapper, batch_size
    )

    # Save per-simulator
    for sim_name, (theta, x) in results.items():
        output_path = output_dir / sim_name
        output_path.mkdir(parents=True, exist_ok=True)
        file_path = output_path / f"samples_{array_index:06d}.h5"

        if file_path.exists():
            continue

        if len(theta) == 0:
            print(f"  WARNING: No valid samples for {sim_name} batch {array_index}")
            continue

        def yield_batch(t=theta, xx=x):
            yield t, xx

        H5Dataset.store(yield_batch(), file_path, size=len(theta))

        # Save metadata (once per simulator directory)
        _save_metadata(output_path, sim_name, simulators[sim_name])

    n_valid = min(len(v[0]) for v in results.values()) if results else 0
    print(f"  Batch {array_index}: {n_valid} valid samples saved")


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

    # =========================================================
    # RESIMULATION MODE: --input-dir provided
    # Loads existing theta from source H5 files, appends new params
    # (e.g., cloud params) from this config's prior, resimulates.
    # =========================================================
    if args.input_dir:
        input_dir = Path(args.input_dir)
        source_files = sorted(input_dir.glob("samples_*.h5"))

        if not source_files:
            print(f"ERROR: No samples_*.h5 files found in {input_dir}")
            sys.exit(1)

        # Determine source param names from metadata.json
        target_param_names = config.get_parameter_names()
        source_param_names = _load_source_param_names(input_dir)

        print(f"\nResimulation mode:")
        print(f"  Source: {input_dir} ({len(source_files)} files, {len(source_param_names)} params)")
        print(f"  Source params: {source_param_names}")
        print(f"  Target: {len(target_param_names)} params")
        new_params = [n for n in target_param_names if n not in source_param_names]
        print(f"  New params to sample: {new_params}")

        if args.parallel:
            from dawgz import job, schedule

            @job(array=min(args.n_array, len(source_files)), cpus=args.cpus, gpus=args.gpus, ram=args.ram, time=args.time)
            def resim_job(array_index: int):
                if array_index < len(source_files):
                    resimulate_batch_from_existing(
                        source_file=source_files[array_index],
                        simulators=simulators,
                        theta_mapper=theta_mapper,
                        source_param_names=source_param_names,
                        target_param_names=target_param_names,
                        target_prior=prior,
                        output_dir=output_dir,
                        array_index=array_index,
                    )

            schedule(
                resim_job,
                name="Resimulation",
                backend="slurm",
                env=["source ~/.bashrc", f"conda activate {args.conda_env}"],
            )
            print(f"\nSubmitted {min(args.n_array, len(source_files))} resimulation jobs.")
            return

        elif args.array_index is not None:
            if args.array_index >= len(source_files):
                print(f"ERROR: array_index {args.array_index} >= {len(source_files)} source files")
                sys.exit(1)

            resimulate_batch_from_existing(
                source_file=source_files[args.array_index],
                simulators=simulators,
                theta_mapper=theta_mapper,
                source_param_names=source_param_names,
                target_param_names=target_param_names,
                target_prior=prior,
                output_dir=output_dir,
                array_index=args.array_index,
            )
        else:
            print("Use --parallel or --array-index with --input-dir.")
            sys.exit(1)

        print("\nResimulation complete.")
        return

    # =========================================================
    # FRESH GENERATION MODE (no --input-dir)
    # =========================================================

    # Parallel mode: submit SLURM array job via dawgz
    if args.parallel:
        from dawgz import job, schedule

        @job(array=args.array, cpus=args.cpus, gpus=args.gpus, ram=args.ram, time=args.time)
        def generate_batch_job(array_index: int):
            generate_batch_file(
                config, simulators, prior, theta_mapper,
                output_dir, array_index, args.batch_size,
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

    # Single batch mode (for manual SLURM array jobs via --array-index)
    if args.array_index is not None:
        print(f"\nGenerating batch {args.array_index} for all simulators...")
        generate_batch_file(
            config, simulators, prior, theta_mapper,
            output_dir, args.array_index, args.batch_size,
        )
    else:
        print("Use --parallel to submit SLURM jobs, or --array-index N for a single batch.")
        sys.exit(1)

    print("\nDataset generation complete.")


if __name__ == "__main__":
    main()


