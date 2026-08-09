"""Validation policy for the baseline and continuously growing ASR manifest."""

from __future__ import annotations

import json
from pathlib import Path

DEFAULT_TEST_TYPES = frozenset({
    "baseline", "feature", "integration", "edge", "stress", "regression",
    "performance", "stability",
})


class ManifestPolicyError(ValueError):
    """Raised before audio work when a suite manifest is not maintainable."""


def validate_manifest(manifest: dict) -> list[dict]:
    """Return cases after enforcing baseline preservation and extension metadata."""
    cases = manifest.get("cases")
    if not isinstance(cases, list):
        raise ManifestPolicyError("manifest 'cases' must be a list")
    baseline_count = manifest.get("baseline_count", 0)
    if not isinstance(baseline_count, int) or baseline_count < 0:
        raise ManifestPolicyError("baseline_count must be a non-negative integer")
    if len(cases) < baseline_count:
        raise ManifestPolicyError(
            f"manifest removed baseline cases: expected at least {baseline_count}, got {len(cases)}")

    policy = manifest.get("growth_policy", {})
    required = tuple(policy.get("extension_required_fields", ("reason", "feature", "type")))
    allowed_types = set(policy.get("test_types", DEFAULT_TEST_TYPES))
    seen_ids, seen_audio = set(), set()
    for index, case in enumerate(cases):
        label = case.get("id") or f"case #{index + 1}"
        if not case.get("id") or case["id"] in seen_ids:
            raise ManifestPolicyError(f"duplicate or missing case id: {label}")
        if not case.get("audio") or case["audio"] in seen_audio:
            raise ManifestPolicyError(f"duplicate or missing audio path for {label}")
        seen_ids.add(case["id"])
        seen_audio.add(case["audio"])
        if index >= baseline_count:
            missing = [field for field in required if not str(case.get(field, "")).strip()]
            if missing:
                raise ManifestPolicyError(
                    f"extension {label} is missing metadata: {', '.join(missing)}")
            if case.get("type") not in allowed_types:
                raise ManifestPolicyError(
                    f"extension {label} has unsupported type: {case.get('type')!r}")
    return cases


def load_manifest(path: Path | str) -> tuple[dict, list[dict]]:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    return manifest, validate_manifest(manifest)
