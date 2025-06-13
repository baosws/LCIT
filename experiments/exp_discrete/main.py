import random
import sys
sys.path.append('')
from src.utils.utils import Timer
from argparse import ArgumentParser
from itertools import product
import yaml
from src.citests import LCIT # GCIT, LCIT, SCIT, KCIT, FisherZ, CCIT
from src.data.data_gen import simulate
import pandas as pd
from torch.cuda import device_count
from tqdm import tqdm, trange
from experiments.exp_discrete.viz import viz
# import neptune.new as neptune
# from neptune.new.types import File
# from dotenv import load_dotenv
from multiprocessing import Pool

methods_dict = dict(
    # SCIT=SCIT,
    # GCIT=GCIT,
    # KCIT=KCIT,
    LCIT=LCIT,
    # CCIT=CCIT,
    # FisherZ=FisherZ
)

def process(task):
    task_id, (dataset, (func, params)) = task
    X, Y, Z = dataset['data']
    indep = dataset['indep']
    Linearity = ['Non-linear', 'Linear'][dataset['Linearity']]
    gpus = [task_id % max(device_count(), 1)]
    with Timer() as t:
        p_value = func(X, Y, Z, gpus=gpus, **params)
    res = dict(
        N=dataset['N'],
        d=dataset['d'],
        X_cat=dataset['X_cat'],
        Y_cat=dataset['Y_cat'],
        Z_cats=dataset['Z_cats'],
        Method=func.__name__,
        Linearity=Linearity,
        P_value=p_value,
        GT=indep,
        randm_state=dataset['random_state'],
        Time=t.elapsed
    )
    return res

if __name__ == '__main__':
    # CONFIG------------------------------------------------------------
    parser = ArgumentParser()
    parser.add_argument('--methods', type=str, nargs='+', default=['LCIT'])
    parser.add_argument('--n_jobs', type=int, default=1)
    args = parser.parse_args()
    methods = args.methods

    EXP_NUM = 'exp7'
    config_path = f'experiments/{EXP_NUM}/config'
    with open(f'{config_path}/common.yml', 'r') as f:
        common_params = yaml.safe_load(f)
        print(f'{common_params = }')
    method_params = {}
    for method in methods:
        with open(f'{config_path}/{method}.yml', 'r') as f:
            method_params[method] = common_params.copy()
            params = yaml.safe_load(f) or {}
            method_params[method].update(params)
            print(f'{method}: {params}')

    with open(f'{config_path}/data.yml', 'r') as f:
        data_config = yaml.safe_load(f)
    T = data_config['T']
    d = data_config['d']
    alpha = data_config['alpha']
    
    # LOGGER------------------------------------------------------------
    # load_dotenv('config/.env')
    # run = neptune.init(
    #     project='baoduong/LCIT',
    #     name=EXP_NUM,
    #     description='Mixed-type data',
    #     source_files=[
    #         *(f'src/citests/{method}.py' for method in methods),
    #         'src/modules/*.py',
    #         'src/data/*.py',
    #         f'experiments/{EXP_NUM}/main.py'
    #     ]
    # )
    # run['methods'] = methods
    # run['params/common'] = common_params
    # run['params/methods'] = method_params
    # run['params/data'] = data_config

    # DATA GEN----------------------------------------------------------
    datasets = []
    res = []
    for Linearity in data_config['Linearity']:
        for N in data_config['N']:
            for d in data_config['d']:
                for X_cat in data_config['X_cat']:
                    for Y_cat in data_config['Y_cat']:
                        for Z_cats in data_config['Z_cats']:
                            for i in range(T):
                                X, Y, Z, indep = simulate(N=N, d=d, X_cat=X_cat, Y_cat=Y_cat, Z_cats=Z_cats, indep=i < T // 2, linear_only=Linearity, random_state=i)
                                datasets.append(dict(
                                    data=(X, Y, Z),
                                    N=N,
                                    d=d,
                                    X_cat=X_cat,
                                    Y_cat=Y_cat,
                                    Z_cats=Z_cats,
                                    indep=indep,
                                    Linearity=Linearity,
                                    random_state=i
                                ))

    # RUN---------------------------------------------------------------
    tasks = list(enumerate(product(datasets, [(methods_dict[method], method_params[method]) for method in methods])))
    if args.n_jobs > 1:
        with Pool(args.n_jobs) as p:
            res = list(tqdm(p.imap_unordered(process, tasks, chunksize=1), total=len(tasks)))
    else:
        res = list(map(process, tqdm(tasks)))

    # VISUALIZE---------------------------------------------------------
    df = pd.DataFrame(res)
    path = f'experiments/{EXP_NUM}/results/result.csv'
    df.to_csv(path)
    # run['result'].upload(path)
    g = viz(path, alpha=0.05)
    # run['plots'].upload(File.as_image(g.fig))