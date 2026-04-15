#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Precompute DROID trajectory lengths and report which are long enough
for a given ``dataset_fpcs`` (frames-per-clip) value.

Only reads h5 state arrays (no video decoding), so it is fast.

Usage examples
--------------
# mpk dataset (glob patterns)
python -m app.plan_common.datasets.droid_traj_stats \
    --data_path /path/to/droid \
    --mpk_manifest_patterns "**/*.h5" \
    --fps 4 \
    --dataset_fpcs 5 10 16 20

# decord dataset (CSV file)
python -m app.plan_common.datasets.droid_traj_stats \
    --data_path /path/to/droid_paths.csv \
    --fps 4 \
    --dataset_fpcs 5 10 16 20

# Save trajectory lengths to a file (can be loaded later)
python -m app.plan_common.datasets.droid_traj_stats \
    --data_path /path/to/droid_paths.csv \
    --fps 4 \
    --dataset_fpcs 5 10 16 20 \
    --output /path/to/seq_lengths.pkl
"""

import argparse
import pickle
from math import ceil
from pathlib import Path
from typing import List, Optional, Sequence

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

VFPS = 30  # DROID recording fps


def _load_mpk_paths(data_path: str, patterns: Sequence[str]) -> List[str]:
    """Glob h5 files under *data_path* matching *patterns*."""
    paths: List[str] = []
    for pattern in patterns:
        cleaned = pattern.lstrip("/")
        found = list(Path(data_path).glob(cleaned))
        if not found:
            print(f"[warn] no files found for pattern {cleaned}")
        paths.extend(str(p) for p in found)
    return paths


def _load_decord_paths(csv_path: str) -> List[str]:
    """Read trajectory directory paths from a whitespace-delimited CSV."""
    return list(pd.read_csv(csv_path, header=None, delimiter=" ").values[:, 0])


def _get_vlen_mpk(path: str) -> Optional[int]:
    """Return trajectory length from an mpk-style h5 file."""
    try:
        with h5py.File(path, "r") as f:
            return f["episode_data"]["observation"]["cartesian_position"].shape[0]
    except Exception as e:
        print(f"[warn] could not read {path}: {e}")
        return None


def _get_vlen_decord(traj_dir: str) -> Optional[int]:
    """Return trajectory length from a decord-style trajectory directory."""
    h5_path = str(Path(traj_dir) / "trajectory.h5")
    try:
        with h5py.File(h5_path, "r") as f:
            return f["observation"]["robot_state"]["cartesian_position"].shape[0]
    except Exception as e:
        print(f"[warn] could not read {h5_path}: {e}")
        return None


def _nframes_for_fpcs(dataset_fpcs: int, fps: int) -> int:
    """Minimum number of raw frames required for *dataset_fpcs* at *fps*."""
    fstp = ceil(VFPS / fps)
    return int(dataset_fpcs * fstp)


def load_seq_lengths(path: str) -> List[int]:
    """Load precomputed trajectory lengths from a pickle file.

    Args:
        path: Path to the pickle file (e.g., seq_lengths.pkl).

    Returns:
        List of trajectory lengths.
    """
    with open(path, "rb") as f:
        return pickle.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Report DROID trajectory length statistics."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Root directory (mpk) or CSV file path (decord).",
    )
    parser.add_argument(
        "--mpk_manifest_patterns",
        type=str,
        nargs="+",
        default=None,
        help="Glob patterns for mpk h5 files (e.g. '**/*.h5').",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=4,
        help="Target sampling fps (default: 4).",
    )
    parser.add_argument(
        "--dataset_fpcs",
        type=int,
        nargs="+",
        required=True,
        help="One or more frames-per-clip values to evaluate.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save trajectory lengths (pickle file, e.g., seq_lengths.pkl).",
    )
    args = parser.parse_args()

    is_mpk = args.mpk_manifest_patterns is not None
    if is_mpk:
        paths = _load_mpk_paths(args.data_path, args.mpk_manifest_patterns)
    else:
        paths = _load_decord_paths(args.data_path)

    print(f"Found {len(paths)} trajectories (mode={'mpk' if is_mpk else 'decord'})")

    get_vlen = _get_vlen_mpk if is_mpk else _get_vlen_decord
    vlens: List[int] = []
    valid_paths: List[str] = []
    for p in tqdm(paths, desc="Reading trajectory lengths"):
        vl = get_vlen(p)
        if vl is not None:
            vlens.append(vl)
            valid_paths.append(p)

    vlens_arr = np.array(vlens)
    total = len(vlens_arr)
    print(f"\nTrajectory length statistics (n={total}):")
    print(f"  min    = {vlens_arr.min()}")
    print(f"  max    = {vlens_arr.max()}")
    print(f"  median = {np.median(vlens_arr):.0f}")
    print(f"  mean   = {vlens_arr.mean():.1f}")

    print(f"\nRequired lengths per dataset_fpcs (fps={args.fps}, vfps={VFPS}):")
    print(f"{'fpcs':>6}  {'nframes':>8}  {'valid':>8}  {'total':>8}  {'pct':>7}")
    print("-" * 45)
    for fpcs in sorted(args.dataset_fpcs):
        nf = _nframes_for_fpcs(fpcs, args.fps)
        valid = int((vlens_arr >= nf).sum())
        pct = 100.0 * valid / total if total > 0 else 0.0
        print(f"{fpcs:>6}  {nf:>8}  {valid:>8}  {total:>8}  {pct:>6.1f}%")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            pickle.dump(vlens, f)
        print(f"\nSaved {len(vlens)} trajectory lengths to {output_path}")

        paths_output = output_path.with_suffix(".paths.pkl")
        with open(paths_output, "wb") as f:
            pickle.dump(valid_paths, f)
        print(f"Saved {len(valid_paths)} trajectory paths to {paths_output}")


if __name__ == "__main__":
    main()
