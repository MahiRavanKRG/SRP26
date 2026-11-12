#!/usr/bin/env bash
# ==============================================================================
# Bigmama Cluster Environment Setup Script (University of Hildesheim | ISMLL)
# ==============================================================================
set -e

echo "=== 1. Checking uv installation on Bigmama ==="
export PATH="$HOME/.local/bin:$PATH"

if ! command -v uv &> /dev/null; then
    echo "Installing uv into ~/.local/bin via wget ..."
    wget -qO- https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
fi

echo "uv version: $(uv --version)"

echo "=== 2. Setting Cache Directory ==="
export HF_HOME="$HOME/.cache/huggingface"
mkdir -p "$HF_HOME"
echo "HF_HOME set to: $HF_HOME"

echo "=== 3. Creating Virtual Environment ==="
cd ~/SRP26
uv venv .venv
source .venv/bin/activate

echo "=== 4. Installing Core Dependencies ==="
# PyTorch with CUDA 12.4 support
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Time-series Foundation Model packages & scientific tools
uv pip install timesfm chronos-forecasting tsfm-public transformers accelerate peft datasets
uv pip install pandas numpy scikit-learn tabulate scipy matplotlib

echo "=== 5. Verifying Installation ==="
python -c "
import torch
print('PyTorch Version :', torch.__version__)
print('CUDA Available  :', torch.cuda.is_available())
if torch.cuda.is_available():
    print('Device Name     :', torch.cuda.get_device_name(0))
else:
    print('Notice: Login node is CPU-only; CUDA will be active on compute nodes via sbatch/srun.')
"

python -c "
from timesfm import TimesFM_2p5_200M_torch
from chronos import ChronosBoltPipeline, ChronosPipeline
from tsfm_public import TinyTimeMixerForPrediction
print('All foundation model libraries imported successfully!')
"

echo "=========================================================="
echo " Setup complete! Ready to submit SLURM jobs via:"
echo "   sbatch submit_array_job.sh"
echo "=========================================================="
