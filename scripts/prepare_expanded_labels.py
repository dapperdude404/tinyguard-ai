"""Generate the reviewed 50-class catalog from the SHA-pinned legacy checkpoint.

python scripts/prepare_expanded_labels.py \
  --checkpoint models/smartplant_mobilenetv3.pt \
  --output data/expanded/class_plan.json

The data/expanded folder must contain independently licensed, labeled images.
"""
import argparse
import json
from pathlib import Path

from smartplant.expanded_labels import build_plan, pinned_legacy_checkpoint_names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="models/smartplant_mobilenetv3.pt")
    parser.add_argument("--output", default="data/expanded/class_plan.json")
    args = parser.parse_args()
    plan = build_plan(pinned_legacy_checkpoint_names(args.checkpoint))
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Planned {len(plan['classes'])} unique categories: 29 existing + 21 candidates")
    for item in plan["classes"]:
        print(f"{item['id']}: {item['name']} [{item['source']}]")
    print("Created metadata plan ONLY. No images have been acquired or model trained.")


if __name__ == "__main__":
    main()
