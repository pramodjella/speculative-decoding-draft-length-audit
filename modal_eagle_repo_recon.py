"""Recon the SafeAILab/EAGLE reference repo -> map its EAGLE-3 inference API so we
can instrument q_t (draft dist), drafted token, p_t (target dist), and acceptance
per step (the pairing vLLM's compiled path hid). Free CPU; just clones + reads source.

Dumps: the .py tree, and the source of the generation + draft-tree + verification
functions (eagenerate / topk_genrate / tree_decoding / evaluate_posterior / update_inference),
flagging lines that compute logits / sample / accept.
"""
import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .run_commands("git clone --depth 1 https://github.com/SafeAILab/EAGLE.git /root/EAGLE")
)
app = modal.App("eagle-repo-recon")


@app.function(image=image, timeout=600)
def probe():
    import os, re
    base = "/root/EAGLE"

    print("===== .py TREE =====", flush=True)
    pyfiles = []
    for root, dirs, files in os.walk(base):
        if ".git" in root:
            continue
        for fn in sorted(files):
            if fn.endswith(".py"):
                rel = os.path.relpath(os.path.join(root, fn), base)
                pyfiles.append(os.path.join(root, fn))
                print("  " + rel, flush=True)

    # functions that define the inference loop / draft / verify
    FN_NAMES = ("eagenerate", "ea_generate", "naive_generate", "topk_genrate",
                "topK_genrate", "tree_decoding", "evaluate_posterior", "update_inference",
                "generate_candidates", "initialize_tree", "forward")
    SIG_KEYS = ("logits", "topk", "argmax", "sample", "softmax", "accept", "posterior",
                "draft", "verify", "candidate", "tree")

    def dump_fn(path, name, maxlines=70):
        try:
            src = open(path).read()
        except Exception:
            return
        # find 'def <name>'
        for m in re.finditer(rf"\n(\s*)def {re.escape(name)}\s*\(", src):
            start = m.start() + 1
            indent = len(m.group(1))
            lines = src[start:].splitlines()
            body = [lines[0]]
            for ln in lines[1:]:
                if ln.strip() and (len(ln) - len(ln.lstrip())) <= indent and ln.lstrip().startswith(("def ", "class ")):
                    break
                body.append(ln)
                if len(body) >= maxlines:
                    break
            rel = os.path.relpath(path, base)
            print(f"\n{'='*70}\n### {rel} :: {name}()  ({len(body)} lines shown)\n{'='*70}", flush=True)
            for i, ln in enumerate(body):
                flag = "  <<" if any(k in ln.lower() for k in SIG_KEYS) else ""
                print(f"  {i:3d}: {ln[:150]}{flag}", flush=True)

    # focus on the model/inference files first
    priority = [p for p in pyfiles if re.search(r"(ea_model|model|cnets|utils|gen)", os.path.basename(p), re.I)]
    seen = set()
    for path in priority + pyfiles:
        if path in seen:
            continue
        seen.add(path)
        for name in FN_NAMES:
            dump_fn(path, name)


@app.local_entrypoint()
def main():
    probe.remote()
