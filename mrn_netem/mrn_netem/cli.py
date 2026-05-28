"""Command line entry point for network fault helpers."""

import argparse
import json

from mrn_netem.loss_models import RandomLossModel
from mrn_netem.profile import load_network_profile


def main() -> None:
    parser = argparse.ArgumentParser(prog="mrn_netem")
    parser.add_argument("--profile", default="", help="YAML network profile to load.")
    parser.add_argument("--loss-rate", type=float, default=0.0)
    parser.add_argument("--packets", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", action="store_true", help="Print JSON output.")
    args = parser.parse_args()

    profile = load_network_profile(args.profile) if args.profile else None
    loss_rate = profile.loss_rate if profile is not None else args.loss_rate
    model = RandomLossModel(loss_rate=loss_rate, seed=args.seed)
    dropped = sum(model.mask(args.packets))
    result = {
        "dropped": dropped,
        "total": args.packets,
        "loss_rate": loss_rate,
    }
    if profile is not None:
        result["profile"] = profile.to_dict()
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(f"dropped={dropped} total={args.packets} loss_rate={loss_rate:.3f}")
        if profile is not None:
            print(
                "profile="
                f"{args.profile} model={profile.model} "
                f"latency_mean_ms={profile.latency_ms_mean:.1f} "
                f"jitter_ms={profile.jitter_ms:.1f}"
            )


if __name__ == "__main__":
    main()
