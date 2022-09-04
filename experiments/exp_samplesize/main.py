import sys

sys.path.append('')
from argparse import ArgumentParser
from itertools import product
from multiprocessing import Pool

import pandas as pd
import yaml
from experiments.exp_samplesize.viz import viz
from src.citests import CCIT, KCIT, LCIT, SCIT
from src.data.data_gen import simulate
from torch.cuda import device_count
from tqdm import tqdm

methods_dict = dict(
    SCIT=SCIT,
    KCIT=KCIT,
    LCIT=LCIT,
    CCIT=CCIT,
)

def process(task):
    task_id, (dataset, (func, params)) = task
    X, Y, Z = dataset['data']
    indep = dataset['indep']
    Linearity = ['Non-linear', 'Linear'][dataset['Linearity']]
    N = dataset['N']
    gpus = [task_id % max(device_count(), 1)]
    
    p_value = func(X, Y, Z, gpus=gpus, **params)
    res = dict(
        N=N,
        Method=func.__name__,
        Linearity=Linearity,
        P_value=p_value,
        GT=indep,
    )
    return res

if __name__ == '__main__':
    # CONFIG------------------------------------------------------------
    parser = ArgumentParser()
    parser.add_argument('--methods', type=str, nargs='+', default=['LCIT'])
    parser.add_argument('--n_jobs', type=int, default=1)
    args = parser.parse_args()
    methods = args.methods

    EXP_NUM = 'exp_samplesize'
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

    # DATA GEN----------------------------------------------------------
    datasets = []
    res = []
    for Linearity in data_config['Linearity']:
            for N in data_config['N']:
                for i in range(T):
                    X, Y, Z, indep = simulate(N, d, indep=i < T // 2, linear_only=Linearity, random_state=i)
                    datasets.append(dict(
                        data=(X, Y, Z),
                        N=N,
                        indep=indep,
                        Linearity=Linearity,
                    ))

    # RUN---------------------------------------------------------------
    tasks = list(enumerate(product(datasets, [(methods_dict[method], method_params[method]) for method in methods])))
    if args.n_jobs > 1:
        with Pool(args.n_jobs) as p:
            res = list(tqdm(p.imap_unordered(process, tasks, chunksize=8), total=len(tasks)))
    else:
        res = list(map(process, tqdm(tasks)))

    # VISUALIZE---------------------------------------------------------
    df = pd.DataFrame(res)
    path = f'experiments/{EXP_NUM}/results/result.csv'
    df.to_csv(path)
    viz(path, alpha=0.05)
