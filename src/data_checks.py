import json
from pathlib import Path
from typing import Any, Dict


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def check_sample_clevr(samples: Dict[str, Any]) -> Dict[str, Any]:
    records = samples
    sample_count = len(records)
    fields = set()
    missing_programs = 0
    for rec in records:
        fields.update(rec.keys())
        if not rec.get("program"):
            missing_programs += 1
    return {
        "sample_count": sample_count,
        "fields": sorted(fields),
        "missing_programs": missing_programs,
    }


def check_clevrer(samples: Dict[str, Any]) -> Dict[str, Any]:
    records = samples
    sample_count = len(records)
    fields = set()
    missing_programs = 0
    for rec in records:
        fields.update(rec.keys())
        if not rec.get("program"):
            missing_programs += 1
    return {
        "sample_count": sample_count,
        "fields": sorted(fields),
        "missing_programs": missing_programs,
    }


def main() -> None:
    base = Path("datasets")
    output = {}

    clevr_path = base / "sample_clevr" / "samples.json"
    clevrer_path = base / "clevrer_counterfactual" / "samples.json"

    clevr_samples = load_json(clevr_path)
    clevrer_samples = load_json(clevrer_path)

    output["sample_clevr"] = check_sample_clevr(clevr_samples)
    output["clevrer_counterfactual"] = check_clevrer(clevrer_samples)

    results_path = Path("results")
    results_path.mkdir(parents=True, exist_ok=True)
    with (results_path / "data_checks.json").open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)


if __name__ == "__main__":
    main()
