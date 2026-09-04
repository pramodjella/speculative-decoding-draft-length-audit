"""Compile + run overhead_microbench.cu on a Modal A100 (nvcc available there).

Runs at two 'weight' sizes to show how the launch-vs-compute-vs-transition split
shifts: ~1 GB (mimics a 0.5B draft) and ~14 GB (mimics a 7B target). Bigger model
=> more memory-bound => launch overhead a smaller fraction.
"""
import subprocess
import modal

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .add_local_file("overhead_microbench.cu", "/root/overhead_microbench.cu")
)
app = modal.App("cuda-overhead-microbench")


@app.function(image=image, gpu="A100", timeout=900)
def run():
    import os
    os.chdir("/root")
    # A100 = sm_80
    c = subprocess.run(
        ["nvcc", "-O3", "-arch=sm_80", "overhead_microbench.cu", "-o", "mb"],
        capture_output=True, text=True)
    print("nvcc stderr:", c.stderr or "(none)")
    if c.returncode != 0:
        return "COMPILE FAILED:\n" + c.stderr
    outs = []
    for wMB, K, S in [(1024, 4, 22), (14336, 4, 22)]:
        r = subprocess.run(["./mb", str(wMB), str(K), str(S)], capture_output=True, text=True)
        outs.append(f"\n########## weight={wMB} MB  K={K}  S={S} ##########\n" + r.stdout + r.stderr)
    return "\n".join(outs)


@app.local_entrypoint()
def main():
    print(run.remote())
