from sqlite3 import NotSupportedError
import time
import copy
import gym
from gym import spaces
import numpy as np
from scipy.spatial.distance import cdist
from components.operators import *
from components.Population import Population


# operators that can act as te first operators 
de_mu = [best2, rand2, current2rand, current2best, rand2best, ]
de_cr = [binomial, exponential]

pso_ud = [currentxbest3, ]

ga_cr = [sbx, mpx]
ga_mu = [gaussian, polynomial, ]

operator_mapping = {
    'DE_Mutation': de_mu,
    'PSO_Operator': pso_ud,
    'GA_Crossover': ga_cr,

    'DE_Crossover': de_cr,
    'GA_Mutation': ga_mu,
}

sampling = ['gaussian', 'sobol', 'lhs', 'halton', 'uniform']

grouping = ['rank', 'nearest', 'rand']

arch_replace = ['oldest', 'worst', 'rand']

pop_reduce = ['linear', 'nonlinear', None]

selection_mode = {
    'DE_Mutation': ['direct', 'crowding', ],
    'PSO_Operator': ['inherit', ],
    'GA_Crossover': ['rank', 'tournament', 'roulette', ],
}

class Optimizer(gym.Env):
    def __init__(self, config, problem, seed=None, import_setting=None, strategy_mode='RL', algorithm_class=None) -> None:

        self.config = config
        self.problem = problem
        # self.MaxFEs = self.config.MaxFEs
        self.MaxGen = self.config.MaxGen
        self.skip_step = config.skip_step

        # ---------------------------- init some variables may needed ---------------------------- #
        self.op_strategy = None
        self.global_strategy = {}
        self.bound_strategy = []
        self.select_strategies = []
        self.regroup_strategy = None
        self.restart_strategies = []
        # self.comm_strategy = None
        self.peid = []

        # ---------------------------- init or import the operators ---------------------------- #
        self.strategy_mode = strategy_mode

        if import_setting is None:
            self.initialize(seed, algorithm_class)
            if self.strategy_mode == 'Given':
                raise NotSupportedError
        else:
            self.import_settings(import_setting)

        # ---------------------------- random number generators ---------------------------- #
        self.rng_seed = None
        self.rng = None
        self.trng = None

    def initialize(self, seed=None, algorithm_class=None):
        if seed is not None:  # seed for generation, not for running
            np.random.seed(seed)

        # -------------------------------- parameters -------------------------------- #
        self.Npop = np.random.choice(self.config.Npop, p=self.config.Npop_prob)
        self.regroup = np.random.choice(self.config.regroup)
        self.NPmax = [np.random.choice(self.config.NPmax) for _ in range(self.Npop)]
        self.NPmin = [np.random.choice(self.config.NPmin) for _ in range(self.Npop)]
        self.NA = np.random.choice(self.config.NA)
        self.Vmax = np.random.choice(self.config.Vmax)
        self.sample = np.random.choice(sampling)
        self.arch_replace = np.random.choice(arch_replace)
        self.enable_bc = np.random.choice([True, False], p=[0.7, 0.3])
        self.pop_reduce = []
        for i in range(self.Npop):
            self.pop_reduce.append(np.random.choice(pop_reduce))
        self.init_grouping = np.random.choice(grouping)

        # -------------------------------- components -------------------------------- #
        self.op_description = []
        self.op_features = None
        self.ops = [[] for _ in range(self.Npop)]
        self.Comms = []
        self.boundings = []
        self.n_component = 0

        # ********** op for each sub pop ********** #
        for i in range(self.Npop):
            # .......... number of ops for the sub pop .......... #
            # opm = np.random.choice(self.config.opm)
            # .......... the first op .......... #
            if algorithm_class == 'DE':
                step_1_ops = de_mu
            elif algorithm_class == 'PSO':
                step_1_ops = pso_ud
            elif algorithm_class == 'GA':
                step_1_ops = ga_cr
            else:
                step_1_ops = de_mu + pso_ud + ga_cr
            op, cls = self.rand_op(step_1_ops)
            self.ops[i] = [op, ]
            self.op_description.append(self.ops[i][-1].description)
            self.peid.append(i*4)
            self.n_component += 1

            if cls == 'DE_Mutation':
                self.ops[i].append(self.rand_op(de_cr)[0])
                self.peid.append(i*4 + 1)
                self.n_component += 1
                self.op_description.append(self.ops[i][-1].description)
            elif cls == 'GA_Crossover':
                self.ops[i].append(self.rand_op(ga_mu)[0])
                self.peid.append(i*4 + 1)
                self.n_component += 1
                self.op_description.append(self.ops[i][-1].description)
            else:  # PSO
                pass

            # ********** bound control ********** #
            # self.bound_id = self.n_component
            bc = Bounding()
            if not self.enable_bc:
                bc.control_param[0]['default'] = np.random.choice(bc.control_param[0]['range'])
            self.boundings.append(bc)
            self.op_description.append(self.boundings[-1].description)
            self.peid.append(i*4+2)
            self.n_component += 1
            # ********** selection ********** #
            # self.select_id = self.n_component
            self.select_strategies.append(Selection(np.random.choice(selection_mode[cls])))
            # self.select_strategy = {'select': }
            # self.op_description.append(self.selection.description)
            # self.n_component += 1
            if self.Npop > 1 and self.regroup < 1:
                self.Comms.append(Comm())
                self.op_description.append(self.Comms[-1].description)
                self.peid.append(i*4+3)
                self.n_component += 1

                self.restart_strategies.append(Restart(np.random.choice(Restart.conditions)))

        # ********** regrouping & communication ********** #
        self.reg_id = None
        if self.regroup > 0 and self.Npop > 1:
            self.regrouping = Regroup()
            self.reg_id = self.n_component
            self.op_description.append(self.regrouping.description)
            self.peid.append(13)
            self.n_component += 1

        # -------------------------------- ob space -------------------------------- #
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.n_component, self.config.maxAct, ),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(0, 100, shape=(self.n_component, self.config.maxAct * 2))
        self.peid = torch.tensor(self.peid, dtype=torch.long)

    def import_settings(self, setting_dict):
        self.Npop = setting_dict['Npop']
        self.regroup = setting_dict['regroup']
        self.NPmax = setting_dict['NPmax']
        self.NPmin = setting_dict['NPmin']
        self.NA = setting_dict['NA']
        self.Vmax = setting_dict['Vmax']
        self.sample = setting_dict['sample']
        self.arch_replace = setting_dict['arch_replace']
        # self.pop_reduce = setting_dict['pop_reduce']
        self.pop_reduce = []
        self.init_grouping = setting_dict['grouping'] if setting_dict['grouping'] is not None else grouping[0]

        self.op_description = []
        self.op_features = None
        self.ops = [[] for _ in range(self.Npop)]
        self.Comms = []
        self.Comms_strategy = []
        self.boundings = []
        self.n_component = 0

        if self.strategy_mode == 'Given':
            self.op_strategy = [[]] * self.Npop
            self.global_strategy = {'message': {}}
            for key in setting_dict['global_strategy'].keys():
                strategy = setting_dict['global_strategy'][key]
                self.global_strategy[key] = eval(strategy['class'])(*strategy['args'])

        # ********** op for each sub pop ********** #
        for i in range(self.Npop):
            subpop_dict = setting_dict['subpops'][i]
            # .......... number of ops for the sub pop .......... #
            ops = subpop_dict['ops']
            opm = len(ops)
            for j in range(opm):
                op_dict = ops[j]
                if op_dict['class'] == 'Multi_op':
                    op_list = []
                    for sub_op in op_dict['op_list']:
                        op_list.append(eval(sub_op['class'])(*sub_op['args']))
                    op = Multi_op(op_list)
                else:
                    op = eval(op_dict['class'])(*op_dict['args'])  # Todo: Multi-op ?

                self.ops[i].append(op)
                self.peid.append(i*3 + j)
                self.n_component += 1
                self.op_description.append(op.description)
                if self.strategy_mode == 'Given':
                    strategy = {}
                    if op_dict['class'] == 'Multi_op':
                        # print(op_dict['op_select'])
                        # print(op_dict['op_select']['class'], op_dict['op_select']['args'])
                        strategy['op_select'] = eval(op_dict['op_select']['class'])(*op_dict['op_select']['args'])
                        strategy['op_list'] = []
                        for sub_op in op_dict['op_list']:
                            strategy['op_list'].append(self.strategy_unzip(sub_op['param_ada']))
                    else:
                        strategy = self.strategy_unzip(op_dict['param_ada'])
                    self.op_strategy[i].append(strategy)
            # ********** bound control ********** #
            # self.bound_id = self.n_component
            self.boundings.append(Bounding())
            self.bound_strategy.append({'bound': subpop_dict['bounding']})
            self.peid.append(12)
            # self.op_description.append(self.bounding.description)
            self.n_component += 1
            # ********** selection ********** #
            # self.select_id = self.n_component
            # self.selection = Selection()
            # self.op_description.append(self.selection.description)
            # self.select_strategy = {'select': setting_dict['selection']}
            self.select_strategies.append(Selection(subpop_dict['selection']))
            # self.selection.control_param[0]['default'] = setting_dict['selection']
            # self.n_component += 1

            # ********** reduce ********** #
            self.pop_reduce.append(subpop_dict['pop_reduce'])
            
            # ********** communication ********** #
            if subpop_dict['comm'] is not None:
                self.Comms.append(Comm())
                self.Comms_strategy.append(subpop_dict['comm'])
                self.n_component += 1


        # ********** regrouping ********** #
        self.reg_id = None
        if setting_dict['regrouping'] is not None:
            self.regrouping = Regroup()
            self.reg_id = self.n_component
            # self.op_description.append(self.regrouping.description)
            self.peid.append(13)
            self.n_component += 1
            if self.strategy_mode == 'Given':
                self.regroup_strategy = setting_dict['regrouping']

        # -------------------------------- ob space -------------------------------- #
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.n_component, self.config.maxAct, ),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(0, 100, shape=(self.n_component, self.config.maxAct * 2))
        self.peid = torch.tensor(self.peid, dtype=torch.long)

    def strategy_unzip(self, param_ada):
        strategy = {}
        for param in param_ada.keys():

            if isinstance(param_ada[param], dict):  # use self-adaptive method
                ada = eval(param_ada[param]['class'])(*param_ada[param]['args'])
            elif param_ada[param] in param_ada.keys():  # use the same value as another param
                ada = param_ada[param]
            elif param_ada[param] in self.global_strategy.keys():  # use a global shared self-adaptive method
                ada = param_ada[param]
            else:  # use the spercified value, could be float, int, or other user spercified types
                ada = param_ada[param]

            strategy[param] = ada
        return strategy

    def set_op_feature(self, features):
        self.op_features = torch.tensor(features)
        if self.op_features.shape[0] < self.config.maxCom:  # mask 
            mask = torch.zeros(self.config.maxCom - self.op_features.shape[0], self.op_features.shape[-1])
            self.op_features = torch.concat((self.op_features, mask), 0)

    def seed(self, seed=None):
        self.rng_seed = seed
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        torch.manual_seed(seed)
        self.trng = torch.random.get_rng_state()

    def reset(self):
        self.problem.reset()
        self.population = Population(self.problem.dim,
                                     NPmax=self.NPmax, 
                                     NPmin=self.NPmin, 
                                     NA=self.NA,
                                     Xmax=self.config.Xmax, 
                                     Vmax=self.Vmax, 
                                     sample=self.sample,
                                     multiPop=self.Npop, 
                                     regroup=self.regroup,
                                     arch_replace=self.arch_replace,
                                     pop_reduce=self.pop_reduce,
                                     rng=self.rng,
                                     seed=self.rng_seed,
                                     )
        self.population.initialize_costs(self.problem)
        self.population.grouping(self.init_grouping, rng=self.rng)
        self.population.update_lbest()
        self.gbest = self.pre_gb = self.init_gb = self.population.gbest  # For reward
        self.stag_count = 0
        self.FEs = self.population.NP
        self.step_count = 0
        if not self.strategy_mode == 'RL':
            self.self_strategy_reset()
        return self.get_state()
            
    def rand_single_op(self, step_op):
        op = np.random.choice(step_op)
        base = op.__base__.__name__
        use_qbest = united_qbest = use_archive = False
        archive_id = united_id = []
        if op.allow_qbest:
            use_qbest=np.random.randint(2)
            united_qbest = False
            if use_qbest:
                united_qbest=np.random.randint(2)
        if op.allow_archive:
            use_archive=np.random.randint(2)
            if use_archive and op.Nxn > 0:
                aid = np.random.choice(op.Nxn, size=min(2, op.Nxn), replace=False)
                archive_id=[aid[0]]
                united_id=[aid[-1]]
        return op(use_qbest, united_qbest, use_archive, archive_id, united_id), base

    def rand_multi_op(self, step_op):
        ops = []
        ban = []
        op, base = self.rand_single_op(step_op)
        while base == 'PSO_Operator':
            op, base = self.rand_single_op(step_op)
        # base = op.__base__.__name__
        ops.append(op)
        ban.append(op.__class__.__name__)
        step_op = operator_mapping[base]
        nop = min(len(step_op), np.random.choice([2, 3]))
        for i in range(nop-1):
            op, _ = self.rand_single_op(step_op)
            while op.__class__.__name__ in ban:
                op, _ = self.rand_single_op(step_op)
            ops.append(op)
            ban.append(op.__class__.__name__)
        return Multi_op(ops), base
    
    def rand_op(self, step_op):
        # op_type= np.random.choice(2, p=[0.75, 0.25])
        op_type= 0
        if op_type == 0:
            return self.rand_single_op(step_op)
        else:
            return self.rand_multi_op(step_op)
        
    def cal_feature(self, group, cost, gbest, gbest_solution, cbest, cbest_solution):
        features = [] # 9

        # features.append(gbest / self.population.init_gb)
        # features.append(cbest / self.population.init_gb)
        # features.append(np.mean(cost / self.population.init_gb))
        # features.append(np.std(cost / self.population.init_gb))

        gbest_ = np.log10(max(1e-8, gbest) + 0)
        cbest_ = np.log10(max(1e-8, cbest) + 0)
        cost[cost < 1e-8] = 1e-8
        cost_ = np.log10(cost + 0)
        init_max = np.log10(self.population.init_max + 0)
        features.append(gbest_ / init_max)
        features.append(cbest_ / init_max)
        features.append(np.mean(cost_ / init_max))
        features.append(np.std(cost_ / init_max))

        dist = np.sqrt(np.sum((group[None,:,:] - group[:,None,:]) ** 2, -1))
        features.append(np.max(dist) / (self.population.Xmax - self.population.Xmin) / np.sqrt(self.problem.dim))
        top10 = np.argsort(cost)[:int(max(1, 0.1*len(cost)))]
        dist10 = np.sqrt(np.sum((group[top10][None,:,:] - group[top10][:,None,:]) ** 2, -1))
        features.append((np.mean(dist10) - np.mean(dist)) / (self.population.Xmax - self.population.Xmin) / np.sqrt(self.problem.dim))

        # FDC
        d_lbest = np.sqrt(np.sum((group - gbest_solution) ** 2, -1))
        c_lbest = cost - gbest
        features.append(np.mean((c_lbest - np.mean(c_lbest)) * (d_lbest - np.mean(d_lbest))) / (np.std(c_lbest) * np.std(d_lbest) + 0.00001))
        d_cbest = np.sqrt(np.sum((group - cbest_solution) ** 2, -1))
        c_cbest = cost - cbest
        features.append(np.mean((c_cbest - np.mean(c_cbest)) * (d_cbest - np.mean(d_cbest))) / (np.std(c_cbest) * np.std(d_cbest)+ 0.00001))

        # features.append((self.MaxFEs - self.FEs) / self.MaxFEs)
        # features = []
        features.append((self.MaxGen - self.step_count) / self.MaxGen)
        
        features = torch.tensor(features)
        
        # if torch.sum(torch.isnan(features)) > 0:
        #     print('='*25 + ' nan features ' + '='*25)
        #     print()
        #     print(np.where(np.isnan(features)))
        #     print(np.sum(np.isnan(group)) > 0)
        #     print(np.sum(np.isnan(cost)) > 0)
        #     print(np.max(cost))
        #     print(group[np.argmax(cost)])


        return features

    def get_state(self):
        states = []
        for i, ops in enumerate(self.ops):
            subpop = self.population.get_subpops(i)
            # print(i, np.max(subpop['cost']), self.boundings[i].control_param[0]['default'])
            local_state = self.cal_feature(subpop['group'], 
                                           subpop['cost'], 
                                           subpop['lbest'], 
                                           subpop['lbest_solutions'], 
                                           subpop['cbest'], 
                                           subpop['cbest_solutions'])
            for io, op in enumerate(ops):
                if self.config.morphological:
                    states.append(torch.concat((op.feature, local_state), -1))
                else:
                    states.append(local_state)

            if self.config.morphological:
                states.append(torch.concat((self.boundings[i].feature, local_state), -1))
            else:
                states.append(local_state)

            if len(self.Comms) > 0:
                if self.config.morphological:
                    states.append(torch.concat((self.Comms[i].feature, local_state), -1))
                else:
                    states.append(local_state)

        # Global features for bounding, selection, regrouping and communication
        global_state = self.cal_feature(np.concatenate(self.population.group),
                                        np.concatenate(self.population.cost), 
                                        self.population.gbest, 
                                        self.population.gbest_solution, 
                                        self.population.cbest, 
                                        self.population.cbest_solution)

        # states.append(torch.concat((self.selection.feature, global_state), -1))
        # states.append(global_state)
        # states.append(global_state)
        if self.reg_id is not None:
            if self.config.morphological:
                states.append(torch.concat((self.regrouping.feature, global_state), -1))
            else:
                states.append(global_state)
                
        states = torch.stack(states)
        if states.shape[0] < self.config.maxCom:  # mask 
            mask = torch.zeros(self.config.maxCom - states.shape[0], states.shape[-1])
            states = torch.concat((states, mask), 0)
        return states

    def cal_symbol_feature(self, group, cost, gbest, gbest_solution, cbest, cbest_solution):
        def dist(x,y):
            return np.sqrt(np.sum((x-y)**2,axis=-1))
        fea_1=(cost-gbest)/(self.population.init_max-gbest+1e-8)
        fea_1=np.mean(fea_1)
        
        distances = cdist(group, group, metric='euclidean')
        np.fill_diagonal(distances, 0)
        mean_distance = np.mean(distances)
        fea_2=mean_distance/np.sqrt((self.config.Xmax*2)**2*self.problem.dim)

        fit=np.zeros_like(cost)
        fit[:group.shape[0]//2]=self.population.init_max
        fit[group.shape[0]//2:]=self.gbest
        maxstd=np.std(fit)
        fea_3=np.std(cost)/(maxstd+1e-8)

        fea_4=(self.MaxGen-self.step_count)/self.MaxGen

        fea_5=self.stag_count/self.MaxGen
        
        fea_6=dist(group,cbest_solution[None,:])/np.sqrt((self.config.Xmax*2)**2*self.problem.dim)
        fea_6=np.mean(fea_6)

        fea_7=(cost-cbest)/(self.population.init_max-gbest+1e-8)
        fea_7=np.mean(fea_7)

        fea_8=dist(group,gbest_solution[None,:])/np.sqrt((self.config.Xmax*2)**2*self.problem.dim)
        fea_8=np.mean(fea_8)

        fea_9=0
        if self.gbest<self.pre_gb:
            fea_9=1
        feature=np.array([fea_1,fea_2,fea_3,fea_4,fea_5,fea_6,fea_7,fea_8,fea_9])
        return feature

    def get_symbol_state(self):
        global_state = self.cal_symbol_feature(np.concatenate(self.population.group),
                                        np.concatenate(self.population.cost), 
                                        self.population.gbest, 
                                        self.population.gbest_solution, 
                                        self.population.cbest, 
                                        self.population.cbest_solution)

        return global_state

    def get_reward(self):
        return (self.pre_gb - self.gbest) / self.init_gb

    def get_config_space(self, name_only=False):  # for SMAC3
        space = {}
        for ip in range(self.Npop):
            for io, op in enumerate(self.ops[ip]):
                name = f'{ip}_{io}_'
                if isinstance(op, Multi_op):
                    space[name + 'multi_op'] = np.arange(len(op.ops)).tolist()
                    for iso, subop in enumerate(op.ops): 
                        if name_only:
                            space[name+f'{iso}'] = subop
                            continue
                        for param in subop.control_param:
                            if param['type'] == 'float':
                                space[name+f'{iso}_'+param['name']] = (float(param['range'][0]), float(param['range'][1]))
                            else:
                                space[name+f'{iso}_'+param['name']] = np.arange(len(param['range'])).tolist()
                else:
                    if name_only:
                        space[name] = op
                        continue
                    for param in op.control_param:
                        if param['type'] == 'float':
                            space[name+param['name']] = (float(param['range'][0]), float(param['range'][1]))
                        else:
                            space[name+param['name']] = np.arange(len(param['range'])).tolist()
            if self.enable_bc:
                space[f'bound_{ip}'] = self.boundings[ip] if name_only else np.arange(len(self.boundings[ip].control_param[0]['range'])).tolist()
        # space['select'] = np.arange(len(self.selection.control_param[0]['range'])).tolist()
        if self.reg_id is None:
            # Communication
            if self.Npop > 1: 
                for ip in range(self.Npop):
                    space[f'comm_{ip}'] = self.Comms[ip] if name_only else np.arange(self.Npop).tolist()
        if self.reg_id is not None:
            space[f'regroup'] = self.regrouping if name_only else np.arange(self.regrouping.nop).tolist()   
        return space
    
    def set_config_space(self, config, op_index=False):  # for SMAC3
        for ip in range(self.Npop):
            for io, op in enumerate(self.ops[ip]):
                name = f'{ip}_{io}_'
                if isinstance(op, Multi_op):
                    op.default_op = config[name + 'multi_op']
                    for iso, subop in enumerate(op.ops):
                        if op_index:
                            op.ops[iso] = copy.deepcopy(config[op.ops[iso].op_id])
                            continue
                        for param in subop.control_param:
                            if param['type'] == 'float':
                                param['default'] = config[name+f'{iso}_'+param['name']]
                            else:
                                param['default'] = param['range'][config[name+f'{iso}_'+param['name']]]
                else:
                    if op_index:
                        self.ops[ip][io] = copy.deepcopy(config[self.ops[ip][io].op_id])
                        continue
                    for param in op.control_param:
                        if param['type'] == 'float':
                            param['default'] = config[name+param['name']]
                        else:
                            param['default'] = param['range'][config[name+param['name']]]
            if self.enable_bc:
                if op_index:
                    self.boundings[ip] = copy.deepcopy(config[self.boundings.op_id])
                else:
                    self.boundings[ip].control_param[0]['default'] = self.boundings[ip].control_param[0]['range'][config[f'bound_{ip}']]

            if self.reg_id is None and self.Npop > 1:
                if op_index:
                    self.Comms[ip] = copy.deepcopy(config[self.comm.op_id])
                else:
                    self.Comms[ip].default_action = config[f'comm_{ip}']

        # self.selection.control_param[0]['default'] = config['select']
        if self.reg_id is not None:
            if op_index:
                self.regrouping = copy.deepcopy(config[self.regrouping.op_id])
            else:
                self.regrouping.default_action = self.regrouping.ops[config['regroup']]

    def given_multi_op_action(self, multi_op, strategy, size, ratio, rng=None):
        if rng is None:
            rng = np.random

        def manage_adaptive(method, sha_id=None):
            if isinstance(method, SHA):  # use successful history adaption
                act = method.get(size, ids=sha_id, rng=rng)
            elif isinstance(method, jDE):  # use adaption proposed in jDE
                act = method.get(size)
            elif isinstance(method, Linear):  # use linearly changing value(s)
                act = method.get(ratio, size)
            elif isinstance(method, Bound_rand):  # use random value(s)
                act = method.get(size, rng=rng)
            else:
                raise NotSupportedError
            return act
        
        actions = {}
        op_selection = strategy['op_select'].get(size, rng=rng)
        actions['op_select'] = op_selection.copy()
        actions['op_list'] = []
        for io, op in enumerate(multi_op.ops):
            actions['op_list'].append({})
            for param in op.control_param:
                name = param['name']
                param_ada = strategy['op_list'][io][name]
                if isinstance(param_ada, Param_adaption):
                    act = manage_adaptive(param_ada)

                elif param_ada in actions['op_list'][-1].keys():  # use the same value as another param
                    act = actions['op_list'][-1][param_ada].copy()

                elif param_ada in self.global_strategy.keys():  # use a global shared self-adaptive method
                    if param_ada in self.global_strategy['message'].keys():  # use the shared values which have already been generated
                        act = self.global_strategy['message'][param_ada].copy()
                    else:
                        sha_id = None  # use the same successful history adaption memory indices, such as F anc Cr usually share the same indices
                        if isinstance(self.global_strategy[param_ada], SHA):
                            if 'SHA_id' not in self.global_strategy['message'].keys():
                                sha_id = self.global_strategy[param_ada].get_ids(size, rng=rng)
                                self.global_strategy['message']['SHA_id'] = sha_id
                            else:
                                sha_id = self.global_strategy['message']['SHA_id']
                        act = manage_adaptive(self.global_strategy[param_ada], sha_id)

                else:  # use the spercified value, could be float, int, or other user spercified types
                    act = param_ada
                if isinstance(act, np.ndarray):
                    act[op_selection != io] = 0
                if isinstance(param_ada, Param_adaption):
                    param_ada.history_value = act.copy()
                actions['op_list'][-1][name] = act
        return actions

    def given_action(self, op, strategy, size, ratio, rng=None):
        if rng is None:
            rng = np.random

        def manage_adaptive(method, sha_id=None):
            if isinstance(method, SHA):  # use successful history adaption
                act = method.get(size, ids=sha_id, rng=rng)
            elif isinstance(method, jDE):  # use adaption proposed in jDE
                act = method.get(size)
            elif isinstance(method, Linear):  # use linearly changing value(s)
                act = method.get(ratio, size)
            elif isinstance(method, Bound_rand):  # use random value(s)
                act = method.get(size, rng=rng)
            elif isinstance(method, DMS):  # use DMS-PSO
                act = method.get(ratio, size)
            else:
                raise NotSupportedError
            return act

        actions = {}
        for param in op.control_param:
            name = param['name']
            param_ada = strategy[name]
            if isinstance(param_ada, Param_adaption):
                act = manage_adaptive(param_ada)

            elif param_ada in strategy.keys():  # use the same value as another param
                act = actions[param_ada]

            elif param_ada in self.global_strategy.keys():  # use a global shared self-adaptive method
                if param_ada in self.global_strategy['message'].keys():  # use the shared values which have already been generated
                    act = self.global_strategy['message'][param_ada]
                else:
                    sha_id = None  # use the same successful history adaption memory indices, such as F anc Cr usually share the same indices
                    if isinstance(self.global_strategy[param_ada], SHA):
                        if 'SHA_id' not in self.global_strategy['message'].keys():
                            sha_id = self.global_strategy[param_ada].get_ids(size, rng=rng)
                            self.global_strategy['message']['SHA_id'] = sha_id
                        else:
                            sha_id = self.global_strategy['message']['SHA_id']
                    act = manage_adaptive(self.global_strategy[param_ada], sha_id)

            else:  # use the spercified value, could be float, int, or other user spercified types
                act = param_ada

            actions[name] = act
        return actions

    def self_strategy_reset(self):
        for ip in range(self.Npop):
            for io, op in enumerate(self.ops[ip]):
                if isinstance(self.ops[ip][io], Multi_op):
                    # self.ops[ip][io].reset()
                    if self.op_strategy is not None:
                        self.op_strategy[ip][io]['op_select'].reset()
                        for sop in self.op_strategy[ip][io]['op_list']:
                            for stra in sop.values():
                                if isinstance(stra, Param_adaption):
                                    stra.reset()
                elif self.op_strategy is not None:
                    for stra in self.op_strategy[ip][io].values():
                        if isinstance(stra, Param_adaption):
                            stra.reset()

    def self_strategy_update(self, old_population):
        # update local strategy for each operator
        subpops = self.population.get_subpops()
        old_subpops = old_population.get_subpops()
        for ip in range(self.Npop):
            for io, op in enumerate(self.ops[ip]):
                if isinstance(self.ops[ip][io], Multi_op):
                    # self.ops[ip][io].update(old_subpops[ip]['cost'], subpops[ip]['cost'], self.step_count/self.MaxGen)  # default strategy
                    if self.op_strategy is not None:  # given strategy
                        self.op_strategy[ip][io]['op_select'].update(old_subpops[ip]['cost'], subpops[ip]['cost'], self.step_count/self.MaxGen)
                        for sop in self.op_strategy[ip][io]['op_list']:
                            for stra in sop.values():
                                if isinstance(stra, Param_adaption):
                                    stra.update(old_subpops[ip]['cost'], subpops[ip]['cost'], self.step_count/self.MaxGen)
                elif self.op_strategy is not None:
                    for stra in self.op_strategy[ip][io].values():
                        if isinstance(stra, Param_adaption):
                            stra.update(old_subpops[ip]['cost'], subpops[ip]['cost'], self.step_count/self.MaxGen)

        # update global strategies
        for k in self.global_strategy.keys():
            if k == 'message':
                self.global_strategy[k] = {}
                continue
            self.global_strategy[k].update(old_population.cost, self.population.cost, self.step_count/self.MaxGen)

    def step(self, logits):
        rewards = 0
        state,reward,is_end,info = self.one_step(logits)
        rewards += reward
        if self.skip_step < 2:
            return state,rewards,is_end,info
        for t in range(1, self.skip_step):
            _,reward,is_end,_ = self.one_step([None]*logits.shape[0], had_action=info['had_action'])
            rewards += reward
        return self.get_state(),rewards,is_end,info

    def one_step(self, logits, had_action=None):
        torch.random.set_rng_state(self.trng)
        pre_state = self.get_symbol_state()
        subpops = self.population.get_subpops()
        old_population = copy.deepcopy(self.population)

        if had_action is None:
            had_action = [None for _ in range(self.n_component)]
        had_action_rec = [None for _ in range(self.n_component)]
        # reproduction
        trails = [[] for _ in range(self.Npop)]
        new_xs, new_ys = [], []
        i_component = 0
        action_values = [[] for _ in range(self.n_component)]
        logp_values = 0
        entropys = []

        # bound_action, select_action = None, self.select_strategy

        # if self.strategy_mode == 'Given':
        #     bound_action = self.bound_strategy
        #     # select_action = self.select_strategy
        # elif self.strategy_mode == 'RL':
        #     bound_action, action_values[self.bound_id], bound_logp, entropy = self.bounding.action_interpret(logits[self.bound_id], softmax=self.config.softmax)
        #     logp_values += np.sum(bound_logp)
        #     entropys += entropy

            # select_action, action_values[self.select_id], select_logp, entropy = self.selection.action_interpret(logits[self.select_id], softmax=self.config.softmax)
            # logp_values += select_logp
            # entropys += entropy
        # else:
        #     pass

        for ip in range(self.Npop):
            for io, op in enumerate(self.ops[ip]):
                subpops[ip]['ratio'] = self.step_count/self.MaxGen
                # subpops[ip]['problem'] = self.problem
                if self.strategy_mode == 'Given':
                    strategy = self.op_strategy[ip][io]
                    if isinstance(op, Multi_op):
                        had_action = self.given_multi_op_action(op, strategy, subpops[ip]['NP'], self.step_count/self.MaxGen, rng=self.rng)
                    else:
                        had_action = self.given_action(op, strategy, subpops[ip]['NP'], self.step_count/self.MaxGen, rng=self.rng)
                    res = op(None, subpops[ip], had_action=had_action, softmax=self.config.softmax, rng=self.rng)
                elif self.strategy_mode == 'RL':
                    res = op(logits[i_component], subpops[ip], softmax=self.config.softmax, rng=self.rng, had_action=had_action[i_component])
                    # print(ip, io, i_component, op)
                    if had_action[i_component] is None:
                        action_values[i_component] = res['actions']
                        logp_values += np.sum(res['logp'])
                        entropys += res['entropy']
                        had_action_rec[i_component] = res['had_action']
                else:
                    res = op(None, subpops[ip], softmax=self.config.softmax, rng=self.rng)
                    if had_action[i_component] is None:
                        action_values[i_component] = res['actions']
                trail = res['result']
                subpops[ip]['trail'] = trail
                i_component += 1
            # bounding
            if self.strategy_mode == 'Given' and self.enable_bc:
                strategy = self.bound_strategy[ip]
                res = self.boundings[ip](None, subpops[ip], had_action=strategy, softmax=self.config.softmax, rng=self.rng)
            elif self.strategy_mode == 'RL' and self.enable_bc:
                res = self.boundings[ip](logits[i_component], subpops[ip], softmax=self.config.softmax, rng=self.rng, had_action=had_action[i_component])
                if had_action[i_component] is None:
                    action_values[i_component] = res['actions']
                    logp_values += np.sum(res['logp'])
                    entropys += res['entropy']
                    had_action_rec[i_component] = res['had_action']
            else:
                res = self.boundings[ip](None, subpops[ip], softmax=self.config.softmax, rng=self.rng, default=not self.enable_bc)
                if had_action[i_component] is None and self.enable_bc:
                    action_values[i_component] = res['actions']
            trail = res['result']
            trails[ip].append(trail)
            subpops[ip]['trail'] = trail
            i_component += 1

            # evluation
            subpops[ip]['trail_cost'] = self.problem.eval(subpops[ip]['trail'])
            self.FEs += subpops[ip]['trail'].shape[0]
            
            # selection
            new_x, new_y, replaced_x, replaced_y = self.select_strategies[ip](subpops[ip], rng=self.rng)
            new_xs.append(new_x)
            new_ys.append(new_y)
            for x, y in zip(replaced_x, replaced_y):
                self.population.update_archive(x, y, rng=self.rng)

        # i_component += 1  # skip the bounding which have been afore processed
        self.population.update_subpop(new_xs, new_ys)

        # update self-adaptive strategies
        if not self.strategy_mode == 'RL':
            self.self_strategy_update(old_population)

        if self.reg_id is None:
            # Communication
            if self.Npop > 1: 
                for ip in range(self.Npop):
                    if self.strategy_mode == 'Given':
                        strategy = self.Comms_strategy[ip]
                        res = self.Comms[ip](None, ip, self.population, had_action=strategy, softmax=self.config.softmax, rng=self.rng)
                    elif self.strategy_mode == 'RL':
                        res = self.Comms[ip](logits[i_component], ip, self.population, softmax=self.config.softmax, rng=self.rng, had_action=had_action[i_component])
                        # print(ip, io, i_component, op)
                        if had_action[i_component] is None:
                            action_values[i_component] = res['actions']
                            logp_values += np.sum(res['logp'])
                            entropys += res['entropy']
                            had_action_rec[i_component] = res['had_action']
                    else:
                        res = self.Comms[ip](None, ip, self.population, softmax=self.config.softmax, rng=self.rng)
                        if had_action[i_component] is None:
                            action_values[i_component] = res['actions']
                    self.population = res['result']
                    i_component += 1
            self.population.update_lbest()

            # Restart
            if len(self.restart_strategies) > 0:
                subpops = self.population.get_subpops()
                new_xs, new_ys = [], []
                for ip in range(self.Npop):
                    if self.restart_strategies[ip](subpops[ip], rng=self.rng):
                        new_x = self.population.initialize_group(self.sample, subpops[ip]['NP'], rng=self.rng, seed=self.rng_seed)
                        new_y = self.problem.eval(new_x)
                        self.FEs += subpops[ip]['NP']
                        new_xs.append(new_x)
                        new_ys.append(new_y)
                    else:
                        new_xs.append(subpops[ip]['group'])
                        new_ys.append(subpops[ip]['cost'])
                self.population.update_subpop(new_xs, new_ys, ignore_vel=True)

        else: # self.reg_id is not None
            # regrouping
            if (self.step_count + 1) % self.regroup == 0:
                if self.strategy_mode == 'RL':
                    res = self.regrouping(logits[self.reg_id], self.population, softmax=self.config.softmax, rng=self.rng, had_action=had_action[self.reg_id])
                    if had_action[self.reg_id] is None:
                        action_values[self.reg_id] = res['actions']
                        entropys += res['entropy']
                        logp_values += np.sum(res['logp'])
                        had_action_rec[self.reg_id] = res['had_action']
                elif self.strategy_mode == 'Given':
                    res = self.regrouping(None, self.population, softmax=self.config.softmax, rng=self.rng, had_action=self.regroup_strategy)
                else:
                    res = self.regrouping(None, self.population, softmax=self.config.softmax, rng=self.rng)
                    if had_action[self.reg_id] is None:
                        action_values[self.reg_id] = res['actions']
                self.population = res['result']
                i_component += 1
        self.step_count += 1
        # pop size reduction
        # self.population.reduction(self.FEs / self.MaxFEs)
        self.population.reduction(self.step_count / self.MaxGen, rng=self.rng)

        self.pre_gb = self.gbest
        if self.gbest > self.population.gbest:
            self.gbest = min(self.gbest, self.population.gbest)
            self.stag_count = 0
        else:
            self.stag_count += 1

        # print(np.max(np.abs(self.population.group)))
        info = {}
        info['action_values'] = action_values
        info['logp'] = logp_values
        info['entropy'] = entropys
        info['gbest_val'] = self.population.gbest
        info['gbest_sol'] = self.population.gbest_solution
        info['init_gb'] = self.init_gb
        info['had_action'] = had_action_rec
        info['pre_state'] = pre_state
        info['nex_state'] = self.get_symbol_state()
        # is_done = self.FEs >= self.MaxFEs or self.population.gbest <= 1e-8
        is_done = self.step_count >= self.MaxGen or self.population.gbest <= 1e-8
        self.trng = torch.random.get_rng_state()

        return self.get_state(), self.get_reward(), is_done, info

    def action_interpret(self, logits, fixed_action):
        i_component = 0
        logp_values = 0
        entropys = []

        for ip in range(self.Npop):
            for op in self.ops[ip]:
                _, _, logp, entropy = op.action_interpret(logits[i_component], softmax=self.config.softmax, fixed_action=fixed_action[i_component])
                i_component += 1
                logp_values += np.sum(logp)
                entropys += entropy

            _, _, bound_logp, entropy = self.boundings[ip].action_interpret(logits[i_component], softmax=self.config.softmax, fixed_action=fixed_action[i_component])
            logp_values += np.sum(bound_logp)
            entropys += entropy
            i_component += 1

        # _, _, select_logp, entropy = self.selection.action_interpret(logits[self.select_id], softmax=self.config.softmax, fixed_action=fixed_action[self.select_id])
        # logp_values += select_logp
        # entropys += entropy
        # i_component += 1
        if self.reg_id is None:
            # Communication
            if self.Npop > 1: 
                for ip in range(self.Npop):
                    _, comm_logp, entropy = self.Comms[ip].get_op(logits[i_component], softmax=self.config.softmax, fixed_action=fixed_action[i_component])
                    logp_values += np.sum(comm_logp)
                    entropys += entropy
                    i_component += 1

        elif len(fixed_action[self.reg_id]) > 0:
            _, logp, entropy = self.regrouping.get_op(logits[self.reg_id], softmax=self.config.softmax, fixed_action=fixed_action[self.reg_id])
            logp_values += float(logp)
            entropys += entropy
        
        return logp_values, entropys
            
