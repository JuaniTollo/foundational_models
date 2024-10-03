#!/usr/bin/env python

import subprocess

def main():
    chronos_path = "/Users/juantollo/Documents/Foundational models/chronos-forecasting/scripts"
    
    command = [
        "python", f"{chronos_path}/evaluation/evaluate.py",
        f"{chronos_path}/evaluation/configs/in-domain.yaml",
        f"{chronos_path}/evaluation/results/chronos-t5-small-in-domain.csv",
        "--chronos-model-id", "amazon/chronos-t5-small",
        "--batch-size=32",
        "--device=cuda:0",
        "--num-samples", "20"
    ]

    # Run the command
    subprocess.run(command)

if __name__ == "__main__":
    main()
