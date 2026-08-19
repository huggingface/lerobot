"""CIG-Bench entry point intentionally requires an explicit integration backend."""

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.parse_args()
    raise SystemExit("Provide a validated simulator backend/task registry before rollout evaluation")


if __name__ == "__main__":
    main()
