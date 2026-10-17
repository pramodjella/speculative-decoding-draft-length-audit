"""Run the full M5 pipeline (honest baseline + 3 seeds + equivalence + analysis)
on a Modal A100.

GPU choice: A100-40GB. The 7B/1.5B bf16 pair needs ~18 GB of weights plus KV
cache, which fits comfortably. Decoding is memory-bandwidth bound, so the A100's
~1.5 TB/s HBM halves per-token latency versus an L40 (GDDR6, ~0.86 TB/s); since
Modal bills by the hour, the faster card is also the cheaper run here.

Usage:
    modal run modal_pipeline.py
Resumable: the results volume holds the per-prompt checkpoint, so re-running
continues where a timeout left off.
"""
import os
import modal

image = (
    modal.Image.debian_slim()
    .pip_install(
        "torch", "transformers==4.44.2", "accelerate", "datasets",
        "pandas", "numpy", "tqdm", "matplotlib", "tabulate",
    )
    .add_local_dir("src", "/root/project/src")
    .add_local_file("run_full_pipeline.py", "/root/project/run_full_pipeline.py")
    .add_local_file("analyze_pipeline.py", "/root/project/analyze_pipeline.py")
)

app = modal.App("adaptive-draft-m5-pipeline")
vol = modal.Volume.from_name("spec-dec-m5-results", create_if_missing=True)

# Tunables (override via env before `modal run`)
TARGET = os.environ.get("M5_TARGET", "Qwen/Qwen2.5-7B-Instruct")
DRAFT = os.environ.get("M5_DRAFT", "Qwen/Qwen2.5-1.5B-Instruct")
SEEDS = os.environ.get("M5_SEEDS", "3")
NPROMPTS = os.environ.get("M5_NPROMPTS", "60")
MAXTOK = os.environ.get("M5_MAXTOK", "64")


@app.function(
    image=image,
    gpu="A100",
    timeout=36000,            # 10h; checkpointing makes this safe
    volumes={"/root/project/results": vol},
)
def run_pipeline():
    os.chdir("/root/project")
    import sys
    sys.path.insert(0, "/root/project")
    sys.path.insert(0, "/root/project/src")

    os.environ.update(
        M5_TARGET=TARGET, M5_DRAFT=DRAFT,
        M5_SEEDS=SEEDS, M5_NPROMPTS=NPROMPTS, M5_MAXTOK=MAXTOK,
    )

    print(f"=== M5 pipeline: target={TARGET} draft={DRAFT} "
          f"seeds={SEEDS} nprompts={NPROMPTS} maxtok={MAXTOK} ===")

    import run_full_pipeline
    run_full_pipeline.main()
    vol.commit()  # persist checkpoint before analysis

    import analyze_pipeline
    analyze_pipeline.main()
    vol.commit()

    # Return key artifacts to the local caller
    out = {}
    for name in ["results/m5_metrics.csv", "results/m5_summary.csv",
                 "results/m5_beat_best_fixed.csv", "results/m5_insight.md",
                 "results/m5_equivalence.json"]:
        if os.path.exists(name):
            with open(name, "r", encoding="utf-8") as f:
                out[os.path.basename(name)] = f.read()
    return out


@app.local_entrypoint()
def main():
    print("=" * 60)
    print("  Launching M5 pipeline on Modal A100")
    print("=" * 60)
    res = run_pipeline.remote()
    os.makedirs("results", exist_ok=True)
    for fname, content in res.items():
        path = os.path.join("results", fname)
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write(content)
        print(f"  saved {path} ({len(content)} bytes)")
    if "m5_beat_best_fixed.csv" in res:
        print("\n--- Beat-the-best-fixed ---")
        print(res["m5_beat_best_fixed.csv"])
    print("\nDone. Figures remain in the Modal volume 'spec-dec-m5-results' (results/figures_m5/).")
