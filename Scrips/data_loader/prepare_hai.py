#!/usr/bin/env python3
import argparse

def main():
    p = argparse.ArgumentParser(description="Prepare HAI data into the SNN-ODE-ATT processed format.")
    p.add_argument("--raw-dir", default="Data/HAI/raw")
    p.add_argument("--out-dir", default="Data/HAI/processed")
    args = p.parse_args()
    print("Preparation stub for HAI. Adapt column names and save processed .pt payloads.", args)

if __name__ == "__main__":
    main()
