#!/usr/bin/env python3
"""Prepare Bosch Production Line Performance into staged BTD format.

Bosch is not a native continuous-time curve dataset. The recommended conversion
uses ordered production stations or measurement blocks as the `T` axis.
"""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare Bosch Production Line Performance into staged BTD tensors."
    )
    parser.add_argument("--raw-dir", default="Data/Bosch_Production_Line/raw")
    parser.add_argument("--out-dir", default="Data/Bosch_Production_Line/processed")
    parser.add_argument("--sample-id-column", default="Id")
    parser.add_argument("--label-column", default="Response")
    parser.add_argument(
        "--stage-parser",
        default="station_measurement_blocks",
        help="How to group columns into the staged T axis.",
    )
    args = parser.parse_args()
    print(
        "Preparation scaffold for Bosch. Convert station/stage measurement "
        "blocks into staged BTD; do not describe it as native continuous time.",
        args,
    )


if __name__ == "__main__":
    main()
