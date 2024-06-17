# Instruction

## Install all dependencies and create env

- Use python 3.9

- If you have anaconda then create environemnt using conda

```bash
# To create conda environment run below command
conda create -n "diverse" python=3.9

# activate conda environemnt
conda activate diverse
```
+ if not create using local python installation by installing python 3.9 version

+ then create environment using below command
```bash
python -m venv venv

# activate the environment using below command
# in powershell
./venv/Scripts/Activate.ps1

# in unix system (macos, linux)
. ./venv/bin/activate
```

+ after then check if correct version 

```bash
# verify python version if it 3.9 or not 
python --version

# install requirements for project
pip install -r requirements.txt

# install extra dependencies
pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 --index-url https://download.pytorch.org/whl/cu113

```

## To Train Chico model 

## For Chico dataset

> **_NOTE:_** Train VAE First use below command to train VAE
```bash
python motion_pred/exp_vae.py --cfg chico_nsamp10
```
> **_NOTE:_** Train DLow (After VAE is trained)

```bash
python motion_pred/exp_dlow.py --cfg chico_nsamp10
```

# Test
### Visualize Motion Samples
```
python motion_pred/eval.py --cfg chico_nsamp10 --mode vis
```
Useful keyboard shortcuts for the visualization GUI:  
| Key           | Functionality |
| ------------- | ------------- |
| d             | test next motion data
| c             | save current animation as `out/video.mp4` |
| space         | stop/resume animation |
| 1             | show DLow motion samples |
| 2             | show VAE motion samples |


### Compute Metrics
```
python motion_pred/eval.py --cfg chico_nsamp10 --mode stats
```  
