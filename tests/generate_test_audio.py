"""CLI for generating missing synthetic speech validation assets."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.audio_generation import MANIFEST_PATH, generate_all


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--regenerate", action="store_true",
                        help="Regenerate files previously marked synthetic; never overwrite human files")
    args = parser.parse_args()
    root = ROOT
    cases = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))["cases"]
    results = generate_all(cases, root, args.regenerate)
    existing = sum(result.status == "READY" for result in results)
    generated = sum(result.status == "GENERATED" for result in results)
    failed = sum(result.status == "TTS_GENERATION_ERROR" for result in results)
    print("=" * 40, "\n TEST AUDIO GENERATION\n", "=" * 40, sep="")
    print(f"Total Test Cases:       {len(results):3d}")
    print(f"Existing Audio:         {existing:3d}")
    print(f"Generated Audio:        {generated:3d}")
    print(f"Generation Failed:      {failed:3d}")
    print(f"Total Ready:            {existing + generated:3d}")
    for result in results:
        if result.status == "TTS_GENERATION_ERROR":
            print(f"TTS_GENERATION_ERROR {result.case_id}: {result.error}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
