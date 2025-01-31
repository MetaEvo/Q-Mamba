import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import gymnasium as gym
import numpy as np
import torch
import warnings
from q_mamba import Q_Mamba
from dataset import My_Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle
from tensorboardX import SummaryWriter
from env.optimizer_mamba import Optimizer
import os
import time, warnings
from q_transformer import Q_Transformer
from dit import DIT

def vector2nn(x, net):
    assert len(x) == sum([param.nelement() for param in net.parameters()]), 'dim of x and net not match!'
    x_copy = copy.deepcopy(x)
    params = net.parameters()
    ptr = 0
    for v in params:
        num_of_params = v.nelement()
        temp = torch.tensor(x_copy[ptr: ptr+num_of_params],dtype=torch.float64)
        v.data = temp.reshape(v.shape)
        ptr += num_of_params
    return net

# this neural network has 2312 parameters
# the initial value range of the parameters should be (-0.1, 0.1), accoding to Kaiming_init  
class MLP_Control_Policy(nn.Module):
    def __init__(self):
        super(MLP_Control_Policy, self).__init__()
        self.ln1 = nn.Linear(27, 64)
        self.ln2 = nn.Linear(64,8)
    
    def forward(self, x):
        x = F.relu(self.ln1(x))
        x = torch.tanh(self.ln2(x)) 
        return x

class MLP_Control_Policy_Hopper(nn.Module):
    def __init__(self):
        super(MLP_Control_Policy_Hopper, self).__init__()
        self.ln1 = nn.Linear(11, 64)
        self.ln2 = nn.Linear(64,3)
    
    def forward(self, x):
        x = F.relu(self.ln1(x))
        x = torch.tanh(self.ln2(x)) 
        return x
    
class MLP_Control_Policy_HalfCheetah(nn.Module):
    def __init__(self):
        super(MLP_Control_Policy_HalfCheetah, self).__init__()
        self.ln1 = nn.Linear(17, 64)
        self.ln2 = nn.Linear(64,6)
    
    def forward(self, x):
        x = F.relu(self.ln1(x))
        x = torch.tanh(self.ln2(x)) 
        return x

class MLP_Control_Policy_Walker2d(nn.Module):
    def __init__(self):
        super(MLP_Control_Policy_Walker2d, self).__init__()
        self.ln1 = nn.Linear(17, 64)
        self.ln2 = nn.Linear(64,6)
    
    def forward(self, x):
        x = F.relu(self.ln1(x))
        x = torch.tanh(self.ln2(x)) 
        return x

class MLP_Control_Policy_Pusher(nn.Module):
    def __init__(self):
        super(MLP_Control_Policy_Pusher, self).__init__()
        self.ln1 = nn.Linear(23, 64)
        self.ln2 = nn.Linear(64,7)
    
    def forward(self, x):
        x = F.relu(self.ln1(x))
        x = torch.tanh(self.ln2(x)) 
        return x*2

class MLP_Control_Policy_Reacher(nn.Module):
    def __init__(self):
        super(MLP_Control_Policy_Reacher, self).__init__()
        self.ln1 = nn.Linear(11, 64)
        self.ln2 = nn.Linear(64,2)
    
    def forward(self, x):
        x = F.relu(self.ln1(x))
        x = torch.tanh(self.ln2(x)) 
        return x

class MLP_Control_Policy_IDP(nn.Module):
    def __init__(self):
        super(MLP_Control_Policy_IDP, self).__init__()
        self.ln1 = nn.Linear(11, 64)
        self.ln2 = nn.Linear(64,1)
    
    def forward(self, x):
        x = F.relu(self.ln1(x))
        x = torch.tanh(self.ln2(x)) 
        return x

class MLP_Control_Policy_InvertedPendulum(nn.Module):
    def __init__(self):
        super(MLP_Control_Policy_InvertedPendulum, self).__init__()
        self.ln1 = nn.Linear(4, 64)
        self.ln2 = nn.Linear(64,1)
    
    def forward(self, x):
        x = F.relu(self.ln1(x))
        x = torch.tanh(self.ln2(x)) 
        return x

class MLP_Control_Policy_Humanoid(nn.Module):
    def __init__(self):
        super(MLP_Control_Policy_Humanoid, self).__init__()
        self.ln1 = nn.Linear(376, 64)
        self.ln2 = nn.Linear(64,17)
    
    def forward(self, x):
        x = F.relu(self.ln1(x))
        x = torch.tanh(self.ln2(x)) *0.4
        return x

class NE_Problem:
    def __init__(self, gym_env='Ant-v4') -> None:
        self.gym_env = gym_env
        if self.gym_env == 'Ant-v4':
            self.dim = 2312
            self.Policy = MLP_Control_Policy
        if self.gym_env == 'Hopper-v4':
            self.dim = 963
            self.Policy = MLP_Control_Policy_Hopper
        if self.gym_env == 'HalfCheetah-v4':
            self.dim = 1542
            self.Policy = MLP_Control_Policy_HalfCheetah
        if self.gym_env == 'Walker2d-v4':
            self.dim = 1542
            self.Policy = MLP_Control_Policy_Walker2d
        if self.gym_env == 'Reacher-v4':
            self.dim = 898
            self.Policy = MLP_Control_Policy_Reacher
        if self.gym_env == "Pusher-v4":
            self.dim = 1991
            self.Policy = MLP_Control_Policy_Pusher
        if self.gym_env == "InvertedDoublePendulum-v4":
            self.dim = 833
            self.Policy = MLP_Control_Policy_IDP
        if self.gym_env == 'InvertedPendulum-v4':
            self.dim = 385
            self.Policy = MLP_Control_Policy_InvertedPendulum
        if self.gym_env == 'Humanoid-v4':
            self.dim = 25233
            self.Policy = MLP_Control_Policy_Humanoid
        self.rollout_step = 1000

    def reset(self):
        pass

    def eval(self, xx, seed=1):
        if len(xx.shape) > 1:
            xs = xx
        else:
            xs = xx[None,:]
        env = gym.make(self.gym_env)
        scores = []
        # transforme the vector representation to neural network
        for net_vec in xs:
            with torch.no_grad():
                # net = MLP_Control_Policy()
                net = self.Policy()
                net = vector2nn(net_vec, net)
                # run an episode
                terminated = False
                truncated = False
                state = torch.tensor(env.reset(seed=seed)[0],dtype=torch.float64)
                returnG = 0.
                for _ in range(self.rollout_step): # no matter what we rollout this env for 100 steps
                # while terminated==False or truncated==False: 
                    action = net(state)
                    observation, reward, _, _, _ = env.step(action.numpy())
                    returnG += reward
                    state = torch.tensor(observation,dtype=torch.float64)
                # print(returnG)
                scores.append( - returnG)
        # return the total accumulated rewards 
        # print(scores)
        if len(xx.shape) > 1:
            return np.array(scores)
        else:
            return scores[0]


import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['figure.dpi'] = 300
plt.rcParams['text.usetex'] = True
plt.rcParams.update(plt.rcParamsDefault)
plt.rc('font',family='Times New Roman')

if __name__ == '__main__':
    # net = MLP_Control_Policy_IDP()
    # total_paramas = sum([param.nelement() for param in net.parameters()])
    # print(total_paramas)
    # exit()
    # for param in net.parameters():
    #     # print(param)
    #     print(torch.mean(param))
    #     print(torch.std(param))
    gyms = ['Ant-v4', 'Hopper-v4', 'HalfCheetah-v4', 'Walker2d-v4', 'Reacher-v4', "Pusher-v4", 'InvertedDoublePendulum-v4', 'Humanoid-v4']
    # gyms = ['Reacher-v4', "Pusher-v4"]
    # gyms = ['InvertedDoublePendulum-v4']
    with open('./task_set_for_mamba_test.pkl', 'rb') as f:
        envs = pickle.load(f)
    # q_mamba_mix = Q_Mamba(from_pretrain='model/20240902T141649/model_66.pth')
    q_mamba_mix = Q_Mamba(from_pretrain='model/qmamba_best_0/epoch_15.pth')
    # q_mamba_mix = Q_Mamba(from_pretrain='model/online_qmb_best_0/epoch_0.pth')
    # q_mamba_mix = Q_Transformer(from_pretrain='model/qt_best_0/epoch_14.pth', device='cuda:0')
    # q_mamba_mix = DIT(act_num=3, from_pretrain='model/dit_best_0/epoch_0.pth')
    q_mamba_mix.eval()
    results = {}
    # pbar = tqdm(total=len(gyms)*10, desc='qmamba ol testing')
    pbar = tqdm(total=len(gyms)*10, desc='qmamba testing')
    for i in range(len(gyms)):
        res = []
        rewards = []
        env = envs[0]
        env.problem = NE_Problem(gyms[i])
        env.MaxGen = 50
        env.config.Xmax = 1
        for j in range(env.Npop):
            env.NPmax[j] = 10
            env.NPmin[j] = 10
        reward = []
        perf = 0
        for k in range(10):
            env.seed(k+1)
            # print('q_mamba_1000 run {}: reward = {}'.format(i,q_mamba_mix.rollout_trajectory(env,500)))
            # reward += q_mamba_mix.rollout_trajectory(env,500,task_id=k//8)
            rew, traj = q_mamba_mix.rollout_trajectory(env,50, req_traj=True)
            reward.append(rew)
            pbar.update()
            res.append(traj)
        # rewards.append(reward/10)
        res = np.array(res)
        print(gyms[i])
        print(np.mean(rewards))
        print(np.mean(res, 0).tolist())
        print(np.std(res, 0).tolist())
        results[gyms[i]] = {}
        results[gyms[i]]['raw'] = res
        results[gyms[i]]['final'] = [np.mean(res[:,-1]),np.std(res[:,-1]),np.mean(reward),np.std(reward)]
        results[gyms[i]]['mean'] = np.mean(res, 0)
        results[gyms[i]]['std'] = np.std(res, 0)
        np.save('QMamba_NE_resv2.npy', results, allow_pickle=True)
        
    # np.save('neu_mamba_3.npy', np.array(results),)
    pbar.close()