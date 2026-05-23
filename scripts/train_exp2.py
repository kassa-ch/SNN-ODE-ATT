#!/usr/bin/env python3
import argparse

def main():
    p = argparse.ArgumentParser(description="Train wafer exp2 using SNN-ODE-ATT.")
    p.add_argument("--config", default="configs/experiments/wafer_exp2.yaml")
    args = p.parse_args()
    print("Training wrapper for exp2. Connect this to snnodeatt.train.trainer with local data paths.", args)

if __name__ == "__main__":
    main()
