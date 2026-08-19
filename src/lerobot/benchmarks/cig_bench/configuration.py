from dataclasses import dataclass


@dataclass
class CIGBenchConfig:
    source_offsets_m: tuple[float, ...] = (0.01, 0.03, 0.05)
    destination_offsets_m: tuple[float, ...] = (0.01, 0.03, 0.05)
    output_json: str = "cig_bench_results.json"
