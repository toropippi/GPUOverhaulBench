from __future__ import annotations


ALLOWED_STATUS = {"ok", "invalid", "failed", "unsupported"}
REQUIRED_META_FIELDS = {"id", "title", "question", "tags"}
REQUIRED_RESULT_FIELDS = {"status", "primary_metric", "unit", "parameters", "measurement", "validation"}
FORBIDDEN_RESULT_KEYS = {
    "analysis",
    "summary",
    "interpretation",
    "considerations",
    "comparisons",
    "question",
    "focus",
    "primary_metrics",
    "theoretical_reference",
}


def _ensure_no_interpretation_fields(value: object, path: str = "result") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key in FORBIDDEN_RESULT_KEYS:
                raise ValueError(
                    f"{path}.{key} is not allowed in result output; keep explanatory text in meta.json"
                )
            _ensure_no_interpretation_fields(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _ensure_no_interpretation_fields(child, f"{path}[{index}]")


def validate_meta(meta: dict, expected_id: str) -> None:
    missing = REQUIRED_META_FIELDS - set(meta)
    if missing:
        raise ValueError(f"meta.json missing fields: {sorted(missing)}")
    if meta["id"] != expected_id:
        raise ValueError(f"meta id mismatch: expected {expected_id}, got {meta['id']}")
    if not isinstance(meta["tags"], list) or not all(isinstance(tag, str) for tag in meta["tags"]):
        raise ValueError("meta.tags must be a list of strings")


def validate_result(result: dict, meta_id: str) -> None:
    _ensure_no_interpretation_fields(result)
    missing = REQUIRED_RESULT_FIELDS - set(result)
    if missing:
        raise ValueError(f"result missing fields: {sorted(missing)}")
    if result["status"] not in ALLOWED_STATUS:
        raise ValueError(f"invalid status: {result['status']}")
    if not isinstance(result["primary_metric"], str) or not result["primary_metric"]:
        raise ValueError("primary_metric must be a non-empty string")
    if not isinstance(result["unit"], str) or not result["unit"]:
        raise ValueError("unit must be a non-empty string")
    if not isinstance(result["parameters"], dict):
        raise ValueError("parameters must be an object")
    if not isinstance(result["measurement"], dict):
        raise ValueError("measurement must be an object")
    if "timing_backend" not in result["measurement"]:
        raise ValueError("measurement.timing_backend is required")
    if not isinstance(result["validation"], dict):
        raise ValueError("validation must be an object")
    if "passed" not in result["validation"]:
        raise ValueError("validation.passed is required")
    if not isinstance(result["validation"]["passed"], bool):
        raise ValueError("validation.passed must be a bool")
    if "meta" in result and result["meta"].get("id") != meta_id:
        raise ValueError("result.meta.id mismatch")
