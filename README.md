# Q-Mamba

Code of paper: Meta-Black-Box-Optimization through Offline Q-function Learning.


![Mamba-DAC Architecture](./src/qmamba.png)

## Preparations

### Create and activate conda environment

```bash
conda create --name q_mamba python=3.10
conda activate q_mamba
pip install -r requirments.txt
```

### Train
To quick start training, just run the following code.

```bash
# train q_mamba with conservative_reg_loss 
CUDA_VISIBLE_DEVICES=0 python run.py --train --trajectory_file_path './trajectory_files/trajectories_set_alg0/trajectory_set_0_Unit.pkl' --has_conservative_reg_loss 

# train q_mamba without conservative_reg_loss
CUDA_VISIBLE_DEVICES=0 python run.py --train --trajectory_file_path './trajectory_files/trajectories_set_alg0/trajectory_set_0_Unit.pkl' 

```


For more detailed configuration options, please refer to the argparse help.

```bash

python run.py --help

```

### Test
Run the following code:
```bash
# test q_mamba 
CUDA_VISIBLE_DEVICES=0 python run.py --test --algorithm_id 0 --load_path [MODEL_PATH] 

```


