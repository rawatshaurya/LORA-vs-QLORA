import os
import json
import glob
import matplotlib.pyplot as plt


def load_summaries(outputs_glob="outputs/*/run_summary.json"):
    paths = glob.glob(outputs_glob)
    summaries = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            s = json.load(f)
        summaries.append(s)
    return summaries


def main():
    summaries = load_summaries()

    if not summaries:
        print("No run_summary.json files found. Train first.")
        return

    # Keep only runs that have memory + accuracy
    rows = []
    for s in summaries:
        acc = s.get("quick_eval_exact_match")
        vram = s.get("peak_vram_reserved_mb") or s.get("peak_vram_alloc_mb")
        if acc is None or vram is None:
            continue
        rows.append((s["run_name"], bool(s.get("use_4bit")), float(vram), float(acc)))

    if not rows:
        print("Found summaries, but missing VRAM/accuracy fields. Check CUDA availability and logging.")
        return

    # Plot
    plt.figure()
    for name, use_4bit, vram, acc in rows:
        label = f"{name} ({'QLoRA' if use_4bit else 'LoRA'})"
        plt.scatter(vram, acc)
        plt.text(vram, acc, label, fontsize=9)

    plt.xlabel("Peak VRAM (MB) [reserved]")
    plt.ylabel("Quick Eval Exact Match")
    plt.title("Accuracy vs Memory: LoRA vs QLoRA")
    plt.grid(True, alpha=0.3)

    out_path = "outputs/accuracy_vs_memory.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print("Saved plot to:", out_path)


if __name__ == "__main__":
    main()
