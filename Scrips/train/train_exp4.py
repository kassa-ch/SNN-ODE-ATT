#!/usr/bin/env python3
import argparse

def main():
    p = argparse.ArgumentParser(description="Train wafer exp4 using SNN-ODE-ATT.")
    p.add_argument("--config", default="Scrips/config/experiments/wafer_exp4.yaml")
    args = p.parse_args()
    print("Training wrapper for exp4. Connect this to snnodeatt.train.trainer with local data paths.", args)

if __name__ == "__main__":
    main()
