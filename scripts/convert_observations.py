"""
Convert raw observation files from WISEJ1738 into uniform CSV format.

Output format for each instrument:
    wavelength[um], flux[Jy], error[Jy]

All files are scaled to D_sim = 9.9 pc using:
    factor = (D_source / D_sim)^2
    flux_scaled = flux_native * scale * factor
    error_scaled = error_native * factor

HST needs unit conversion from cW/m2/nm → Jy:
    flux_Jy = flux_cgs * 1e1 * 1e30 * (wlen_cm)^2 / c

MIRI and Gemini are already in Jy.

Simulated spectra are stored as single-row CSVs (already in scaled units from the pipeline).
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ---------- Constants ----------
c_cgs = 2.99792458e10  # speed of light in cm/s
D_source = 7.34        # parallax distance in pc
D_sim = 9.9            # simulation distance in pc
scale = 1e5            # flux scaling factor used in pipeline
factor = (D_source / D_sim) ** 2

# ---------- Paths ----------
obs_root = Path("/media/mvasist/Elements/PhDprojects/WISEJ1738/observation")
out_root = Path("/home/mvasist/Documents/SBI4atmret/testing/observations/WISEJ1738")

# Source files (from observations.py)
HST_FILE = obs_root / "HST" / "WISEJ1738_HST.txt"
MIRI_FILE = obs_root / "MIRI" / "unconvolved" / "spectrum_reprocessed231123.csv"
GEMINI_FILE = obs_root / "NIRGemini" / "spectrum_gemini.csv"
SIM_DIR = obs_root / "simulations"


def convert_hst(input_path, output_path):
    """
    HST: comma-separated, header row, columns = wavelength[um], flux[cW/m2/nm], error[cW/m2/nm]
    Convert flux/error from cW/m2/nm to Jy, then apply scale * factor.
    
    Conversion: F_Jy = F_cgs * 1e1 * 1e30 * (lambda_cm)^2 / c
    where lambda_cm = wavelength_um * 1e-4
    """
    df = pd.read_csv(input_path, header=0, delimiter=",")
    wlen = df.iloc[:, 0].values            # um
    flux_cgs = df.iloc[:, 1].values        # cW/m2/nm
    error_cgs = df.iloc[:, 2].values       # cW/m2/nm

    # Unit conversion: cW/m2/nm → Jy
    conv = 1e1 * 1e30 * (wlen * 1e-4) ** 2 / c_cgs
    flux_jy = flux_cgs * conv
    error_jy = error_cgs * conv

    # Apply scale and distance factor
    flux_scaled = flux_jy * scale * factor
    error_scaled = error_jy * factor

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame({
        "wavelength[um]": wlen,
        "flux[Jy]": flux_scaled,
        "error[Jy]": error_scaled,
    })
    out_df.to_csv(output_path, index=False, float_format='%.17g')
    print(f"  HST: {len(wlen)} points → {output_path}")


def convert_miri(input_path, output_path):
    """
    MIRI: CSV with header, columns = wavelength[um], flux[Jy], error[Jy]
    Already in Jy — just apply scale * factor.
    """
    df = pd.read_csv(input_path)
    wlen = df.iloc[:, 0].values
    flux_jy = df.iloc[:, 1].values
    error_jy = df.iloc[:, 2].values

    flux_scaled = flux_jy * scale * factor
    error_scaled = error_jy * factor

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame({
        "wavelength[um]": wlen,
        "flux[Jy]": flux_scaled,
        "error[Jy]": error_scaled,
    })
    out_df.to_csv(output_path, index=False, float_format='%.17g')
    print(f"  MIRI: {len(wlen)} points → {output_path}")


def convert_gemini(input_path, output_path):
    """
    Gemini: CSV with header row '0,1,2', columns = wavelength[um], flux[Jy], error[Jy]
    Already in Jy — just apply scale * factor.
    Uses float_precision='round_trip' to preserve exact wavelength values.
    """
    df = pd.read_csv(input_path, float_precision="round_trip")
    wlen = df.iloc[:, 0].values
    flux_jy = df.iloc[:, 1].values
    error_jy = df.iloc[:, 2].values

    flux_scaled = flux_jy * scale * factor
    error_scaled = error_jy * factor

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame({
        "wavelength[um]": wlen,
        "flux[Jy]": flux_scaled,
        "error[Jy]": error_scaled,
    })
    out_df.to_csv(output_path, index=False, float_format='%.17g')
    print(f"  Gemini: {len(wlen)} points → {output_path}")


def convert_simulations(sim_dir, output_dir):
    """
    Simulated spectra are single-row CSVs. Each column is a flux value
    in the concatenated wavelength order used by the pipeline:
        [hst+gemini sorted by wavelength] + [miri]
    
    These are already in the scaled units from the pipeline, so we just copy them.
    """
    sim_files = [
        "mostprobCF_simulation0_noisefree.csv",
        "mostprobCF_simulation1.csv",
        "mostprobCF_simulation2.csv",
        "mostprobCF_simulation3.csv",
        "mostprobCLavg_simulation0_noisefree.csv",
        "mostprobCLavg_simulation1.csv",
        "mostprobCLavg_simulation2.csv",
        "mostprobCLavg_simulation3.csv",
    ]

    output_dir.mkdir(parents=True, exist_ok=True)

    for fname in sim_files:
        src = sim_dir / fname
        if src.exists():
            dst = output_dir / fname
            df = pd.read_csv(src)
            df.to_csv(dst, index=False)
            n_cols = df.shape[1]
            print(f"  Simulation: {fname} ({n_cols} flux values) → {dst}")
        else:
            print(f"  Simulation: {fname} — NOT FOUND, skipping")


def main():
    print("Converting observations to uniform format...")
    print(f"  Distance scaling: D={D_source} pc → D_sim={D_sim} pc, factor={factor:.6f}")
    print(f"  Flux scale: {scale}")
    print()

    convert_hst(HST_FILE, out_root / "hst" / "spectrum.csv")
    convert_miri(MIRI_FILE, out_root / "miri" / "spectrum.csv")
    convert_gemini(GEMINI_FILE, out_root / "gemini" / "spectrum.csv")
    print()

    print("Converting simulations...")
    convert_simulations(SIM_DIR, out_root / "simulations")
    print()

    print("Done. Config paths should point to:")
    print(f"  hst:    testing/observations/WISEJ1738/hst/spectrum.csv")
    print(f"  gemini: testing/observations/WISEJ1738/gemini/spectrum.csv")
    print(f"  miri:   testing/observations/WISEJ1738/miri/spectrum.csv")
    print(f"  simulated: testing/observations/WISEJ1738/simulations/<filename>.csv")


if __name__ == "__main__":
    main()
