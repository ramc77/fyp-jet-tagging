#!/usr/bin/env python3
"""
Download the Top Quark Tagging Reference Dataset from Zenodo.
=============================================================
Dataset: https://zenodo.org/records/2603256
DOI: 10.5281/zenodo.2603256

This dataset contains 2M Monte Carlo simulated jets from 14 TeV pp collisions:
  - 1.2M training jets
  - 400k validation jets
  - 400k test jets

Each jet has up to 200 constituents with 4-momenta (E, px, py, pz).
Labels: is_signal_new (1 = top quark jet, 0 = QCD jet)

Total download size: ~1.6 GB (HDF5 compressed; ~17 GB uncompressed)

By Dr. Ram Chand, The Begum Nusrat Bhutto Women University, Sukkur.
"""

import os
import sys
import argparse
import urllib.request
import hashlib

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)

# Zenodo record 2603256 - Top Quark Tagging Reference Dataset
# These are the direct download links from the Zenodo API
# Note: Files are HDF5-compressed (~1.6 GB total), much smaller than
# the uncompressed ~17 GB often cited. Jet counts are identical.
FILES = {
    "train.h5": {
        "url": "https://zenodo.org/records/2603256/files/train.h5",
        "size_gb": 1.0,
    },
    "val.h5": {
        "url": "https://zenodo.org/records/2603256/files/val.h5",
        "size_gb": 0.33,
    },
    "test.h5": {
        "url": "https://zenodo.org/records/2603256/files/test.h5",
        "size_gb": 0.33,
    },
}


def download_with_progress(url, filepath, description=""):
    """Download a file with a progress bar."""
    print(f"\nDownloading {description}...")
    print(f"  URL:  {url}")
    print(f"  Dest: {filepath}")

    def progress_hook(count, block_size, total_size):
        percent = min(100, count * block_size * 100 // total_size)
        downloaded = count * block_size / (1024**3)
        total = total_size / (1024**3)
        bar = "█" * (percent // 2) + "░" * (50 - percent // 2)
        sys.stdout.write(f"\r  [{bar}] {percent}% ({downloaded:.1f}/{total:.1f} GB)")
        sys.stdout.flush()

    urllib.request.urlretrieve(url, filepath, reporthook=progress_hook)
    print("\n  Done!")


def main():
    parser = argparse.ArgumentParser(description="Download Top Tagging dataset")
    parser.add_argument(
        "-y", "--yes", action="store_true",
        help="Skip the confirmation prompt (for non-interactive use, e.g. Colab)."
    )
    args = parser.parse_args()

    print("=" * 65)
    print("Top Quark Tagging Reference Dataset Downloader")
    print("Zenodo DOI: 10.5281/zenodo.2603256")
    print("=" * 65)

    total_size = sum(f["size_gb"] for f in FILES.values())
    print(f"\nTotal download size: ~{total_size:.1f} GB")
    print(f"Download directory:  {DATA_DIR}\n")

    # Check which files already exist
    existing = []
    needed = []
    for fname, info in FILES.items():
        fpath = os.path.join(DATA_DIR, fname)
        if os.path.exists(fpath):
            size_gb = os.path.getsize(fpath) / (1024**3)
            print(f"  [EXISTS] {fname} ({size_gb:.1f} GB)")
            existing.append(fname)
        else:
            print(f"  [NEEDED] {fname} (~{info['size_gb']:.1f} GB)")
            needed.append(fname)

    if not needed:
        print("\nAll files already downloaded! Ready to proceed.")
        return

    needed_size = sum(FILES[f]["size_gb"] for f in needed)
    print(f"\nNeed to download {len(needed)} file(s), ~{needed_size:.1f} GB")

    if args.yes or not sys.stdin.isatty():
        print("\nProceeding with download (non-interactive / --yes).")
    else:
        response = input("\nProceed with download? [y/N]: ").strip().lower()
        if response != "y":
            print("Download cancelled.")
            print("\nAlternative: Download manually from https://zenodo.org/records/2603256")
            print(f"Place files in: {DATA_DIR}/")
            return

    for fname in needed:
        info = FILES[fname]
        fpath = os.path.join(DATA_DIR, fname)
        download_with_progress(info["url"], fpath, f"{fname} (~{info['size_gb']:.1f} GB)")

    print("\n" + "=" * 65)
    print("Download complete! You can now run the pipeline.")
    print("=" * 65)


if __name__ == "__main__":
    main()
