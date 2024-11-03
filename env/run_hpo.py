import torch
from config import get_options
import numpy as np
import xgboost as xgb
import copy
from utils.utils import *
import pickle

class Problem_hpo(object):
    def __init__(self,bst_surrogate,dim,y_min,y_max) -> None:
        self.bst_surrogate=bst_surrogate
        self.y_min=y_min
        self.y_max=y_max
        self.dim=dim

    def func(self,position):
        x_q = xgb.DMatrix(position.reshape(-1,self.dim))
        new_y = self.bst_surrogate.predict(x_q)
        
        return -self.normalize(new_y)

    def normalize(self, y):
        # if y_min is None:
        #     return (y-np.min(y))/(np.max(y)-np.min(y))
        # else:
        return np.clip((y-self.y_min)/(self.y_max-self.y_min),0,1)

if __name__ == '__main__':
    opts=get_options()
    opts.population_size=5
    opts.max_fes=500
    opts.max_x=1
    opts.min_x=0
    opts.require_baseline=False
    opts.device = torch.device("cpu")
    tokenizer=MyTokenizer()
    model=LSTM(opts,tokenizer)

    # load_path=r'/home/chenjiacheng/L2E/L2E_0816_cec/full_model/0917_madde_gap_deltaxrandx_linearcritic_20230917T153806/epoch-90.pt'
    # load_path=r'/home/chenjiacheng/L2E/L2E_0816_cec/full_model/0917_madde_newbr_deltaxrandx_linearcritic_20230917T153215/epoch-80.pt'
    # load_path=r'/home/chenjiacheng/L2E/L2E_0816_cec/full_model/0917_madde_gap_deltaxrandx_linearcritic_newexpr4_20230917T232200/epoch-98.pt'
    load_path=r'/home/chenjiacheng/L2E/L2E_0816_cec/outputs/epoch-92.pt'
    
    # load_path=r'/home/chenjiacheng/L2E/L2E_0816_newexpr/lamda_model/lamda150.pt'

    # opts.fea_mode='no_opt'
    # load_path=r'/home/chenjiacheng/L2E/L2E_0816_cec/outputs/fea_model/xy.pt'
    # load_path=r'/home/chenjiacheng/L2E/L2E_0816_cec/outputs/reduced_set.pt'
    # load_path=r'/home/chenjiacheng/L2E/L2E_0816_cec/outputs/epoch-48.pt'
    # load_path=r'/home/chenjiacheng/L2E/L2E_0816_newexpr/new_fea_model/no_fit.pt'

    # load_path=r'lamda_model/lamda150.pt'
    runner=trainer(model,opts)
    runner.load(load_path)

    # load data
    meta_train_data,meta_vali_data,meta_test_data,bo_initializations,surrogates_stats=get_data(root_dir="HPO-B-main/hpob-data/", mode="v3-test", surrogates_dir="HPO-B-main/saved-surrogates/")
    save_dict={}
    env=L2E_env(dim=opts.dim,ps=opts.population_size,problem=None,max_x=opts.max_x,min_x=opts.min_x,max_fes=opts.max_fes,boarder_method='clipping')
    total_cost=[]

    collect_all=[]
    for search_space_id in meta_test_data.keys():
        save_dict[search_space_id]={}
        for dataset_id in meta_test_data[search_space_id].keys():
            collect_one_pro=[]
            bst_model,y_min,y_max=get_bst(surrogates_dir='HPO-B-main/saved-surrogates/',search_space_id=search_space_id,dataset_id=dataset_id,surrogates_stats=surrogates_stats)
            X = np.array(meta_test_data[search_space_id][dataset_id]["X"])
            y = np.array(meta_test_data[search_space_id][dataset_id]["y"])
            dim = X.shape[1]
            p=Problem_hpo(bst_surrogate=bst_model,dim=dim,y_min=y_min,y_max=y_max)
            gbests=[]
            # ts=time.time()
            for run in range(5):
                np.random.seed(run)
                action={'problem':p}
                env.step(action)
                pop=env.reset()
                is_done=False
                cost=[pop.gbest_cost]
                while not is_done:
                    pop_feature = [pop.feature_encoding(opts.fea_mode)]
                    pop_feature=torch.FloatTensor(pop_feature)
                    seq,const_seq,log_prob=runner.actor(pop_feature)
                    expr=[]
                    pre,c_pre=get_prefix_with_consts(seq[0],const_seq[0],0)
                    str_expr=[tokenizer.decode(pre[i]) for i in range(len(pre))]
                    success,infix=prefix_to_infix(str_expr,c_pre,tokenizer)
                    assert success, 'fail to construct the update function'
                    expr.append(infix)
                    # print(f'x+{infix}')
                    action={'base_population':copy.deepcopy(pop),'expr':expr[0],'skip_step':opts.skip_step,'select':opts.stu_select}
                    pop,_,is_done,_=env.step(action)
                    cost.append(pop.gbest_cost)
                collect_one_pro.append(cost)
                gbests.append(pop.gbest_cost)
            collect_all.append(gbests)
            # te=time.time()
            # print(f'time:{(te-ts)/5}')
            # torch.rand()
            save_dict[search_space_id][dataset_id]=collect_one_pro
            print(f'search:{search_space_id}, dataset:{dataset_id}, mean:{np.mean(gbests)}')
            total_cost.append(np.mean(gbests))
    print(f'avg:{np.mean(total_cost)}')
    collect_all=np.stack(collect_all)
    print(f'collect_all:{np.mean(collect_all,0)}')
    print(np.mean(collect_all,0).shape)
    print(f'std:{np.std(np.mean(collect_all,0))}')
    print(load_path)
    # print(opts.fea_mode)
    # save_dir='save_data/'
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)
    # with open(save_dir+'epoch92_skip5_newhpo_data.pkl','wb') as f:
    #     print('saving data...')
    #     pickle.dump(save_dict,f,-1)
    #     print('saved')
