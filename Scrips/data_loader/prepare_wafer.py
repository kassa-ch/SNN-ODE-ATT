#!/usr/bin/env python3
import argparse

def main():
    p = argparse.ArgumentParser(description="Prepare WAFER data into the SNN-ODE-ATT processed format.")
    p.add_argument("--raw-dir", default="Data/Wafer/exp4/raw")
    p.add_argument("--out-dir", default="Data/Wafer/exp4/processed")
    args = p.parse_args()
    print("Preparation stub for WAFER. Adapt column names and save processed .pt payloads.", args)

if __name__ == "__main__":
    main()
