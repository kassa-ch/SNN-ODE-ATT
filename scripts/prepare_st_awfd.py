#!/usr/bin/env python3
"""Prepare ST-AWFD files into the SNN-ODE-ATT BTD payload.

This is a safe scaffold: raw files are not bundled with the repository. Adapt
column names after downloading the official ST-AWFD data.
"""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare ST-AWFD into {x, mask, delta_t, tau, label, sample_id} tensors."
    )
    parser.add_argument("--raw-dir", default="data/ST_AWFD/raw")
    parser.add_argument("--out-dir", default="data/ST_AWFD/processed")
    parser.add_argument("--group-id-column", default="MaterialID")
    parser.add_argument("--time-column", default="duration_ms")
    parser.add_argument("--stage-column", default="StepID")
    parser.add_argument("--label-column", default="target")
    args = parser.parse_args()
    print(
        "Preparation scaffold for ST-AWFD. Download the official data, map the "
        "columns to BTD, and save processed .pt payloads.",
        args,
    )


if __name__ == "__main__":
    main()
