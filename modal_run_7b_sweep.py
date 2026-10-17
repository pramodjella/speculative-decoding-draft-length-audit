import os
import modal

# Define the Modal image with all required dependencies
image = (
    modal.Image.debian_slim()
    .pip_install(
        "torch",
        "transformers",
        "accelerate",
        "datasets",
        "pandas",
        "tqdm",
    )
    # Add project source code and sweep script to container
    .add_local_dir("src", "/root/project/src")
    .add_local_file("run_7b_physical_sweep.py", "/root/project/run_7b_physical_sweep.py")
)

app = modal.App("speculative-decoding-7b-sweep")

# Persistent volume for logging metrics and taxonomy
# Mount it to /root/project/results so the sweep script naturally writes checkpoints here.
results_volume = modal.Volume.from_name("spec-dec-7b-results-vol", create_if_missing=True)


@app.function(
    image=image,
    gpu="A100",  # Requesting a 40GB A100 GPU (approx $2.20/hr)
    timeout=36000,  # 10 hours timeout limit
    volumes={"/root/project/results": results_volume},
)
def run_sweep():
    # Switch working directory to /root/project so relative imports and file writes work
    os.chdir("/root/project")
    
    # Add project directory and src directory to Python path
    import sys
    sys.path.insert(0, "/root/project")
    sys.path.insert(0, "/root/project/src")
    
    print("Initializing speculative decoding sweep on Modal container...")
    
    # Import the sweep main function from the copied file
    import run_7b_physical_sweep
    
    print("Launching run_7b_physical_sweep.main()...")
    run_7b_physical_sweep.main()
    
    # Force commit to persistent volume to save metrics
    print("Sweep complete. Committing outputs to persistent Volume...")
    results_volume.commit()
    
    # Read files to return them directly to the local runner
    metrics_path = "results/physical_roadmap_metrics_7b_0.5b.csv"
    taxonomy_path = "results/physical_error_taxonomy_7b_0.5b.csv"
    
    metrics_data = ""
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics_data = f.read()
            
    taxonomy_data = ""
    if os.path.exists(taxonomy_path):
        with open(taxonomy_path, "r", encoding="utf-8") as f:
            taxonomy_data = f.read()
            
    return {
        "metrics_7b": metrics_data,
        "taxonomy_7b": taxonomy_data
    }


@app.local_entrypoint()
def main():
    print("==========================================================")
    print("  Launching 7B/1.5B Speculative Decoding Sweep on Modal")
    print("  GPU Requested: NVIDIA A100 (40GB)")
    print("==========================================================")
    
    try:
        # Launch the remote GPU execution
        res = run_sweep.remote()
        
        # Write files locally
        os.makedirs("results", exist_ok=True)
        
        if res.get("metrics_7b"):
            with open("results/physical_roadmap_metrics_7b.csv", "w", newline="", encoding="utf-8") as f:
                f.write(res["metrics_7b"])
            print("Successfully saved: results/physical_roadmap_metrics_7b.csv")
        else:
            print("Warning: No metrics data returned.")
            
        if res.get("taxonomy_7b"):
            with open("results/physical_error_taxonomy_7b.csv", "w", newline="", encoding="utf-8") as f:
                f.write(res["taxonomy_7b"])
            print("Successfully saved: results/physical_error_taxonomy_7b.csv")
        else:
            print("Warning: No taxonomy data returned.")
            
        print("\nSweep execution completed successfully! All files synced locally.")
        print("You can now run local analysis and update your paper figures.")
        
    except Exception as e:
        print(f"\nError occurred during Modal execution: {e}")
        print("Checkpoints are preserved in the persistent volume. Rerun to resume.")
