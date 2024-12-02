# Object Tracker

Segmentation module and tracking.

---

## Requirements

This project has the following requirements:

1. **Python**: Version 3.10
2. **CUDA**: Version 12.1 (for GPU support)
3. **Dependencies**:
   - PyTorch 2.2.0 and related libraries
   - Torch Geometric and extensions
   - Git-based libraries:
     - [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)
     - [Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything)
     - [XMem](https://github.com/hkchengrex/XMem)
   - Core libraries like `numpy`, `opencv-python-headless`, `scipy`, `scikit-learn`, and `open3d`

---

## Installation

Follow these steps to install and set up the project:

### 1. Clone the Repository

Clone the project repository to your local machine:

```bash
git clone <repository-url>
cd <repository-directory>
```


### 2. Set Up a Python Environment

1. Ensure you have Conda installed on your system. If not, you can download it from [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/distribution).
2. Create a new Conda environment with Python 3.10:
```bash
conda create -n tracker python=3.10 -y
conda activate tracker
```



### 3. Install the project dependencies
If you have a CUDA environment, make sure that the variable `CUDA_HOME` is set.
To check if `CUDA_HOME` is set, run:
   ```bash
   echo $CUDA_HOME
   ```
If nothing is printed, it means `CUDA_HOME` is not set up. To temporarily set the environment variable in the current shell, run:
   ```bash
export CUDA_HOME=/path/to/cuda-11.3
   ```
Replace `/path/to/cuda-11.3` with the actual path where your CUDA toolkit is installed. You can find this path by running:
   ```bash
which nvcc
   ```
Setup the environment first:
```bash
bash setup_env.sh
```

Install GroundingDINO:
```bash
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO
pip install -e .
cd ..
```

Install SAM:
```bash
git clone https://github.com/facebookresearch/segment-anything.git
cd segment-anything
pip install -e .
cd ..
```

Install XMem:
```bash
git clone https://github.com/hkchengrex/XMem.git
cd XMem
pip install -r requirements.txt
cd ..
```

Export XMem path to your `PYTHONPATH` variable:
```bash
export PYTHONPATH=$PYTHONPATH:<path-to-folder>/object_tracker
export PYTHONPATH=$PYTHONPATH:<path-to-folder>/object_tracker/XMem
```



<!-- Install the project dependencies in editable mode:
```bash
pip install -e .
``` -->


### 4. Verify PyTorch, CUDA and Additional Dependencies
Make sure the environemtn
Test that PyTorch and CUDA are installed correctly:
```bash
python -c "import torch
print(f'Torch Version: {torch.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')
"
```
Expected output:
```bash
Torch Version: 2.2.0+cu121
CUDA Available: True
```

Test that GroundingDINO is installed correctly:
```bash
python -c "
from groundingdino.util.inference import load_model
print('GroundingDINO installed successfully.')
"
```

Test that SAM is installed correctly:
```bash
python -c "
from segment_anything import SamPredictor
print('SAM installed successfully.')
"
```


Test that XMem is installed correctly:
```bash
python -c "
from XMem.model.network import XMem
print('XMem installed successfully.')
"
```