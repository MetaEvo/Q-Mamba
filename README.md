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
│  ├── trajectory_set_0_Rand.pkl
│  ├── trajectory_set_0_CfgX.pkl  
│  ├── trajectory_set_0_Unit.pkl   
│  ├── trajectory_set_1_Unit.pkl   
│  ├── trajectory_set_2_Unit.pkl                     
```
Then we can train the main Q-Mamba agent using:

```bash
# train q_mamba with conservative_reg_loss 
CUDA_VISIBLE_DEVICES=0 python run.py --train --trajectory_file_path './trajectory_files/trajectory_set_0_Unit.pkl' --has_conservative_reg_loss 

# train q_mamba without conservative_reg_loss
CUDA_VISIBLE_DEVICES=0 python run.py --train --trajectory_file_path './trajectory_files/trajectory_set_0_Unit.pkl' 

```

To reproduce the experiment in Section 5.4, Table 2 in the paper, you could run the training with parameter ``lambda'' = 0, 1 or 10 and ``beta'' = 1 or 10 to change the two hyper-parameters. For example:

```bash

CUDA_VISIBLE_DEVICES=0 python run.py --train --trajectory_file_path './trajectory_files/trajectory_set_0_Unit.pkl' --has_conservative_reg_loss --lambda=1 --beta=1

```

For the experiment on action bins (Appendix E.1), run the following command:
```bash

CUDA_VISIBLE_DEVICES=0 python run.py --train --trajectory_file_path './trajectory_files/trajectory_set_0_Unit.pkl' --has_conservative_reg_loss --action_bins 16 --actions_dim 5

```
with different ``action_bins'' = 16, 32, 64, 128, 256, or 512 and ``actions_dim'' = 5, 6, 7, 8, 9 or 10 accordingly.

For the E&E data ratio experiment, modified the ``rate'' parameter in line 37 to the value in (0, 0.25, 0.5, 0.75, 1) and assign the data path in line 58. line 60, then run the training command:

```bash
CUDA_VISIBLE_DEVICES=0 python EE_data_ratio.py

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
CUDA_VISIBLE_DEVICES=0 python run.py --test --algorithm_id 0 --load_path [MODEL_PATH] 

```

To run the Neuroevolution experiemnt (Section 5.2, Figure 2), assign the ``model_path'' variable the path of the model to be tested in _neuralevolution.py_ line 150, then run the command:
```bash
# test Neuroevolution 
CUDA_VISIBLE_DEVICES=0 python neuralevolution.py

```
Then the results will be stored in the QMamba_NE_res.npy file as a dictionary containing the best rewards of each run, each generation.


### Log
Run the following code:
```bash

tensorboard --logdir=./log/

```
