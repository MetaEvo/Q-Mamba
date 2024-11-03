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
        self.rng_seed = 0
        self.rng = np.random.RandomState(seed)
        self.trng = torch.random.get_rng_state()

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
        return self.get_symbol_state()
            
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

        return torch.from_numpy(global_state)

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

    # def mamba_action_interpret(self, logits, bins=16):
    #     logits = logits.cpu().numpy()
    #     space = self.get_config_space()
    #     alg_actions = {}
    #     action = []
    #     for i, subspace in enumerate(space.keys()):
    #         rge = space[subspace]
    #         if isinstance(rge[0], float):
    #             ac = np.argmax(logits[i])
    #             alg_actions[subspace] = (ac / (bins - 1)) * (rge[-1] - rge[0]) + rge[0]
    #             action.append(ac)
    #         else:
    #             na = len(rge)
    #             ac = np.argmax(logits[i][:na])
    #             alg_actions[subspace] = ac
    #             action.append(ac)
    #     return action, alg_actions
    def mamba_action_interpret(self, alg_actions, bins=16):
        space = self.get_config_space()
        for i, subspace in enumerate(space.keys()):
            rge = space[subspace]
            if isinstance(rge[0], float):
                ac = alg_actions[subspace].item()
                alg_actions[subspace] = (ac / (bins - 1)) * (rge[-1] - rge[0]) + rge[0]
            # else:
            #     ac = alg_actions[subspace]
            #     alg_actions[subspace] = ac
        return alg_actions

    def mamba_step(self, logits):
        torch.random.set_rng_state(self.trng)
        alg_actions = self.mamba_action_interpret(logits)
        self.set_config_space(alg_actions)
        pre_state = self.get_symbol_state()
        subpops = self.population.get_subpops()
        # reproduction
        trails = [[] for _ in range(self.Npop)]
        new_xs, new_ys = [], []
        i_component = 0
        for ip in range(self.Npop):
            for io, op in enumerate(self.ops[ip]):
                res = op(None, subpops[ip], softmax=self.config.softmax, rng=self.rng, random=False)
                trail = res['result']
                subpops[ip]['trail'] = trail
                i_component += 1
            # bounding
            res = self.boundings[ip](None, subpops[ip], softmax=self.config.softmax, rng=self.rng, default=True)
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

        self.population.update_subpop(new_xs, new_ys)

        if self.reg_id is None:
            # Communication
            if self.Npop > 1: 
                for ip in range(self.Npop):
                    res = self.Comms[ip](None, ip, self.population, softmax=self.config.softmax, rng=self.rng, random=False)
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
                res = self.regrouping(None, self.population, softmax=self.config.softmax, rng=self.rng, random=False)
                self.population = res['result']
                i_component += 1
        self.step_count += 1
        self.population.reduction(self.step_count / self.MaxGen, rng=self.rng)

        self.pre_gb = self.gbest
        if self.gbest > self.population.gbest:
            self.gbest = min(self.gbest, self.population.gbest)
            self.stag_count = 0
        else:
            self.stag_count += 1

        # print(np.max(np.abs(self.population.group)))
        info = {}
        # info['action_values'] = action_values
        info['gbest_val'] = self.population.gbest
        info['gbest_sol'] = self.population.gbest_solution
        info['init_gb'] = self.init_gb
        info['pre_state'] = pre_state
        info['nex_state'] = self.get_symbol_state()
        # is_done = self.FEs >= self.MaxFEs or self.population.gbest <= 1e-8
        is_done = self.step_count >= self.MaxGen or self.population.gbest <= 1e-8
        self.trng = torch.random.get_rng_state()

        return self.get_symbol_state(), self.get_reward(), is_done, info
