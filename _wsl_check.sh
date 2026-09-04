#!/usr/bin/env bash
# Status + environment verification for the WSL EAGLE-3 setup.
set -u
VENV=/root/eagle-venv
LOG=/root/vllm_install.log

if pgrep -f "pip install" >/dev/null; then
  echo "PIP_STATUS=RUNNING"
else
  echo "PIP_STATUS=NOT_RUNNING"
fi
echo "====LOG_TAIL===="
tail -n 12 "$LOG" 2>/dev/null
echo "====VLLM_CHECK===="
"$VENV/bin/python" - <<'PY'
try:
    import torch
    print("torch", torch.__version__, "cuda_avail", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("device", torch.cuda.get_device_name(0))
        print("capability", torch.cuda.get_device_capability(0))  # (12,0)=Blackwell sm_120
        x = torch.randn(8, device="cuda")          # raises if no sm_120 kernels
        print("cuda_kernel_ok", float((x * 2).sum().cpu()))
except Exception as e:
    print("TORCH_OR_CUDA_FAIL:", repr(e)[:300])
try:
    import vllm
    print("vllm", vllm.__version__)
except Exception as e:
    print("VLLM_IMPORT_FAIL:", repr(e)[:300])
PY
