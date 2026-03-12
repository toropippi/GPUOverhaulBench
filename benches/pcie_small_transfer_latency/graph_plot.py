from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_result(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def format_fit_label(name: str, model: dict) -> str:
    if not model.get("success"):
        return f"{name} fit (unavailable)"
    return (
        f"{name} fit "
        f"(lat={model['fixed_latency_us']:.3f} us, "
        f"bw={model['fitted_bandwidth_gib_per_s']:.3f} GiB/s, "
        f"R^2={model['r2']:.6f})"
    )


def format_fit_equation(name: str, model: dict) -> str:
    if not model.get("success"):
        return f"{name}: fit unavailable"
    return (
        f"{name}: t(us) = {model['fixed_latency_us']:.6f} + "
        f"{model['slope_us_per_byte']:.9e} * bytes"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot pcie_small_transfer_latency latest.json")
    parser.add_argument(
        "--input",
        default="results/latest.json",
        help="Input JSON path relative to this bench directory",
    )
    parser.add_argument(
        "--output",
        default="graph/transfer_time_loglog.png",
        help="Output image path relative to this bench directory",
    )
    args = parser.parse_args()

    bench_dir = Path(__file__).resolve().parent
    input_path = (bench_dir / args.input).resolve()
    output_path = (bench_dir / args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = load_result(input_path)
    measurement = payload["result"]["measurement"]
    cases = measurement["cases"]
    h2d_model = measurement["h2d_model"]
    d2h_model = measurement["d2h_model"]

    sizes = [case["size_bytes"] for case in cases]
    h2d_measured = [case["h2d"]["avg_copy_us"] for case in cases]
    d2h_measured = [case["d2h"]["avg_copy_us"] for case in cases]
    h2d_predicted = [case["h2d"]["predicted_copy_us"] for case in cases]
    d2h_predicted = [case["d2h"]["predicted_copy_us"] for case in cases]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")

    ax.plot(sizes, h2d_measured, marker="o", linewidth=2, color="#0B6E4F", label="H2D measured")
    ax.plot(sizes, d2h_measured, marker="s", linewidth=2, color="#C84C09", label="D2H measured")
    ax.plot(sizes, h2d_predicted, linestyle="--", linewidth=1.8, color="#43AA8B", label=format_fit_label("H2D", h2d_model))
    ax.plot(sizes, d2h_predicted, linestyle="--", linewidth=1.8, color="#F28F3B", label=format_fit_label("D2H", d2h_model))

    ax.set_title("Pinned CUDA copy time vs transfer size")
    ax.set_xlabel("Transfer size (bytes, log2 scale)")
    ax.set_ylabel("Average copy time (us, log scale)")
    ax.grid(True, which="both", linestyle=":", linewidth=0.8, alpha=0.7)
    ax.legend(loc="upper left", fontsize=8)
    equation_text = "\n".join(
        [
            format_fit_equation("H2D", h2d_model),
            format_fit_equation("D2H", d2h_model),
        ]
    )
    ax.text(
        0.02,
        0.70,
        equation_text,
        transform=ax.transAxes,
        fontsize=8,
        va="top",
        ha="left",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85, "edgecolor": "#777777"},
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)

    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
