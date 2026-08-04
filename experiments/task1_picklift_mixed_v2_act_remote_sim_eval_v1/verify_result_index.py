from __future__ import annotations

import json

from finalize_results import (
    EXPERIMENT_DIR,
    build_result_index,
    load_json,
    sha256_file,
)


def main() -> None:
    result_path = EXPERIMENT_DIR / "result_index.json"
    comparison_path = EXPERIMENT_DIR / "comparison_result.json"
    expected_result, expected_comparison = build_result_index()
    if load_json(result_path) != expected_result:
        raise RuntimeError("Result index differs from independent recomputation.")
    if load_json(comparison_path) != expected_comparison:
        raise RuntimeError("Comparison result differs from independent recomputation.")
    print(
        json.dumps(
            {
                "status": "independent_result_index_verification_passed",
                "result_index_sha256": sha256_file(result_path),
                "comparison_result_sha256": sha256_file(comparison_path),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
