# Q-Mamba

Code of paper: Meta-Black-Box-Optimization through Offline Q-function Learning.


![Mamba-DAC Architecture](./src/qmamba.png)

## Preparations

### Create and activate conda environment
First, create the q_mamba environment.
```bash
conda create --name q_mamba python=3.10
conda activate q_mamba
```

Second, install the mamba-ssm following the instructions on: https://github.com/state-spaces/mamba.git.

Third, install the other necessary libraries.
```bash
pip install -r requirements.txt
```

### Train
To quickly start training, 
firstly, download the training trajectories from [here](https://github.com/GMC-DRL/Q-Mamba/tree/main). The directory could be set like this basic structure:
```bash
├── /trajectory_files/
│  ├── trajectory_set_0_Unit.pkl   
│  ├── trajectory_set_1_Unit.pkl   
│  ├── trajectory_set_2_Unit.pkl                     
```
Second, run the following code.

```bash
# train q_mamba with conservative_reg_loss 
CUDA_VISIBLE_DEVICES=0 python run.py --train --trajectory_file_path './trajectory_files/trajectory_set_0_Unit.pkl' --has_conservative_reg_loss 

# train q_mamba without conservative_reg_loss
CUDA_VISIBLE_DEVICES=0 python run.py --train --trajectory_file_path './trajectory_files/trajectory_set_0_Unit.pkl' 

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


