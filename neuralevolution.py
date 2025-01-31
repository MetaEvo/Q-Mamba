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


class MLP_Control_Policy(nn.Module):
    def __init__(self):
        super(MLP_Control_Policy, self).__init__()
        self.ln1 = nn.Linear(27, 64)
        self.ln2 = nn.Linear(64,8)
    
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


class MLP_Control_Policy_IDP(nn.Module):
    def __init__(self):
        super(MLP_Control_Policy_IDP, self).__init__()
        self.ln1 = nn.Linear(11, 64)
        self.ln2 = nn.Linear(64,1)
    
    def forward(self, x):
        x = F.relu(self.ln1(x))
        x = torch.tanh(self.ln2(x)) 
        return x


class NE_Problem:
    def __init__(self, gym_env='Ant-v4') -> None:
        self.gym_env = gym_env
        if self.gym_env == 'Ant-v4':
            self.dim = 2312
            self.Policy = MLP_Control_Policy
        if self.gym_env == 'HalfCheetah-v4':
            self.dim = 1542
            self.Policy = MLP_Control_Policy_HalfCheetah
        if self.gym_env == "Pusher-v4":
            self.dim = 1991
            self.Policy = MLP_Control_Policy_Pusher
        if self.gym_env == "InvertedDoublePendulum-v4":
            self.dim = 833
            self.Policy = MLP_Control_Policy_IDP
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
                for _ in range(self.rollout_step): 
                    action = net(state)
                    observation, reward, _, _, _ = env.step(action.numpy())
                    returnG += reward
                    state = torch.tensor(observation,dtype=torch.float64)

                scores.append( - returnG)

        if len(xx.shape) > 1:
            return np.array(scores)
        else:
            return scores[0]

import numpy as np

if __name__ == '__main__':
    gyms = ['Ant-v4', 'HalfCheetah-v4', "Pusher-v4", 'InvertedDoublePendulum-v4']
    with open('./task_set_for_mamba_test.pkl', 'rb') as f:
        envs = pickle.load(f)

    q_mamba_mix = Q_Mamba(from_pretrain=model_path)

    q_mamba_mix.eval()
    results = {}
    pbar = tqdm(total=len(gyms)*10, desc='qmamba testing')
    for i in range(len(gyms)):
        res = []
        rewards = []
        env = envs[0]
        env.problem = NE_Problem(gyms[i])
        env.MaxGen = 50
        env.config.Xmax = 0.1
        for j in range(env.Npop):
            env.NPmax[j] = 10
            env.NPmin[j] = 10
        reward = []
        perf = 0
        for k in range(10):
            env.seed(k+1)
            rew, traj = q_mamba_mix.rollout_trajectory(env,50, req_traj=True)
            reward.append(rew)
            pbar.update()
            res.append(traj)
        res = np.array(res)
        print(gyms[i])
        print(np.mean(reward))
        print(np.mean(res, 0).tolist())
        print(np.std(res, 0).tolist())
        results[gyms[i]] = {}
        results[gyms[i]]['raw'] = res
        results[gyms[i]]['final'] = [np.mean(res[:,-1]),np.std(res[:,-1]),np.mean(reward),np.std(reward)]
        results[gyms[i]]['mean'] = np.mean(res, 0)
        results[gyms[i]]['std'] = np.std(res, 0)
        np.save('QMamba_NE_resv2.npy', results, allow_pickle=True)
        
    pbar.close()
