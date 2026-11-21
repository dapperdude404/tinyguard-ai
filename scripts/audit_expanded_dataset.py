"""Strict preflight validation; no training occurs until this passes."""
import argparse
import json
from smartplant.expanded_data import audit_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/expanded")
    parser.add_argument("--plan", default="data/expanded/class_plan.json")
    parser.add_argument("--groups-csv")
    parser.add_argument("--min-train", type=int, default=20)
    parser.add_argument("--min-val", type=int, default=5)
    parser.add_argument("--min-test", type=int, default=5)
    args = parser.parse_args()
    report = audit_dataset(args.dataset, args.plan,
                           (args.min_train, args.min_val, args.min_test),
                           groups_csv=args.groups_csv)
    print(json.dumps(report, indent=2))
    if args.groups_csv is None:
        print("WARNING: No plant/farm source-group provenance supplied; independent field validation NOT established.")


if __name__ == "__main__":
    main()
