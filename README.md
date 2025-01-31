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
Then install Pytorch with version 1.12+ and CUDA 11.6+ (see https://pytorch.org/ for more details). The cuda-toolkit is also required ``conda install nvidia::cuda-toolkit=12.1``.

Next, install the mamba-ssm using ``pip install mamba-ssm`` (see https://github.com/state-spaces/mamba.git for more details).

Finally, install the other necessary libraries.
```bash
pip install -r requirements.txt
```

Other requirements:
- Linux
- NVIDIA GPU
- PyTorch 1.12+
- CUDA 11.6+


### Train
To quickly start training, 
firstly, download the training trajectories from [here](https://github.com/GMC-DRL/Q-Mamba/tree/main). The directory could be set like this basic structure:
```bash
├── /trajectory_files/
│  ├── trajectory_set_0_Rand.pkl
│  ├── trajectory_set_0_CfgX.pkl  
│  ├── trajectory_set_0_Unit.pkl   
│  ├── trajectory_set_1_Unit.pkl   
│  ├── trajectory_set_2_Unit.pkl                     
```
Then we can train the main Q-Mamba agent using:

```bash
# train q_mamba with conservative_reg_loss 
python run.py --train --trajectory_file_path './trajectory_files/trajectory_set_0_Unit.pkl' --has_conservative_reg_loss 

# train q_mamba without conservative_reg_loss
python run.py --train --trajectory_file_path './trajectory_files/trajectory_set_0_Unit.pkl' 

```

To reproduce the experiment in Section 5.4, Table 2 in the paper, you could run the training with parameter _lambda_ = 0, 1 or 10 and _beta_ = 1 or 10 to change the two hyper-parameters. For example:

```bash

python run.py --train --trajectory_file_path './trajectory_files/trajectory_set_0_Unit.pkl' --has_conservative_reg_loss --lambda=1 --beta=1

```

For the experiment on action bins (Appendix E.1), run the following command:
```bash

python run.py --train --trajectory_file_path './trajectory_files/trajectory_set_0_Unit.pkl' --has_conservative_reg_loss --action_bins 16 --actions_dim 5

```
with different _action_bins_ = 16, 32, 64, 128, 256, or 512 and _actions_dim_ = 5, 6, 7, 8, 9 or 10 accordingly.

For the E&E data ratio experiment, modified the ``rate'' parameter in line 37 to the value in (0, 0.25, 0.5, 0.75, 1) and assign the data path in line 58. line 60, then run the training command:

```bash
python EE_data_ratio.py

```

For more detailed configuration options, please refer to the argparse help. 

```bash

python run.py --help

```

Taking the training on _Alg0_ as an example, the models in the training is saved at ./model/trajectory_set_0_Unit/YYMMDDTHHmmSS/ (where YYMMDDTHHmmSS is the time stamp of the run) and the tensorboard log is stored at ./log/trajectory_set_0_Unit/YYMMDDTHHmmSS 

### Test
To test the trained model on the BBOB testing problems (Section 5.2, Table 1):
```bash
# test q_mamba 
python run.py --test --algorithm_id 0 --load_path [MODEL_PATH] 

```


### Log
Run the following code:
```bash

tensorboard --logdir=./log/

```
