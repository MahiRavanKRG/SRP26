#!/bin/bash
cd ~/SRP26
source .venv/bin/activate
python3 -c "
import torch
print('=== PyTorch & CUDA ===')
print('PyTorch Version:', torch.__version__)
print('CUDA Available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('Device Name:', torch.cuda.get_device_name(0))

print('\n=== Foundation Models ===')
from timesfm import TimesFM_2p5_200M_torch
print('[x] Google TimesFM 2.5 import SUCCESS')

from chronos import ChronosBoltPipeline, ChronosPipeline
print('[x] Amazon Chronos import SUCCESS')

from tsfm_public import TinyTimeMixerForPrediction
print('[x] IBM Granite TTM import SUCCESS')

print('\n=== Baseline & Diagnostics ===')
import sklearn, scipy, tabulate, matplotlib
print('[x] SciPy, Scikit-Learn, Tabulate import SUCCESS')
"

