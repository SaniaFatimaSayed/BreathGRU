"""
Example script for running BreathGRU inference.

Usage
-----
python run_inference.py \
    --audio example.wav \
    --model ../pretrained/breathgru.pt \
    --priors ../pretrained/duration_priors.json
"""

import argparse

from breathgru import BreathGRU


def main():
    parser = argparse.ArgumentParser(
        description="Run BreathGRU speech-breath segmentation."
    )

    parser.add_argument(
        "--audio",
        required=True,
        help="Path to input audio file.",
    )

    parser.add_argument(
        "--model",
        required=True,
        help="Path to pretrained BreathGRU model.",
    )

    parser.add_argument(
        "--priors",
        required=True,
        help="Path to duration priors JSON.",
    )

    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Device to run inference on.",
    )

    args = parser.parse_args()

    model = BreathGRU(
        checkpoint_path=args.model,
        duration_priors_path=args.priors,
        device=args.device,
    )

    segments = model.predict(args.audio)

    print("\nPredicted Segments")
    print("-" * 60)

    for label, start, end, duration in segments:
        print(
            f"{label:<8} "
            f"{start:7.2f}s -> {end:7.2f}s "
            f"({duration:5.2f}s)"
        )


if __name__ == "__main__":
    main()
