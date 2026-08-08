"""CLI for generating missing synthetic speech validation assets."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.audio_generation import (
    MANIFEST_PATH,
    generate_all,
    list_windows_voices,
    remove_test_audio,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--regenerate", action="store_true",
                        help="Regenerate files previously marked synthetic; never overwrite human files")
    parser.add_argument("--list-voices", action="store_true",
                        help="List installed Windows SAPI voices and exit")
    cleanup = parser.add_mutually_exclusive_group()
    cleanup.add_argument("--remove-generated", action="store_true",
                         help="Remove generated WAVs while preserving human recordings")
    cleanup.add_argument("--remove-all", action="store_true",
                         help="Remove every manifest test WAV, including human recordings")
    args = parser.parse_args()
    if args.list_voices:
        print(list_windows_voices())
        return 0
    root = ROOT
    cases = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))["cases"]
    if args.remove_generated or args.remove_all:
        result = remove_test_audio(cases, root, include_human=args.remove_all)
        print("=" * 40, "\n TEST AUDIO CLEANUP\n", "=" * 40, sep="")
        print(f"Removed Audio:          {result['removed']:3d}")
        print(f"Preserved Human Audio:  {result['preserved']:3d}")
        print(f"Already Missing:        {result['missing']:3d}")
        return 0
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
