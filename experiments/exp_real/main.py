import random
import sys

sys.path.append('')
from argparse import ArgumentParser
from itertools import combinations, product
from multiprocessing import Pool

import numpy as np
import pandas as pd
import yaml
from experiments.exp_real.viz import viz
from src.citests import CCIT, KCIT, LCIT, SCIT
from src.utils.utils import Timer
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
    
    gpus = [task_id % max(device_count(), 1)]
    with Timer() as t:
        p_value = func(X, Y, Z, gpus=gpus, **params)
    res = dict(
        Dataset=dataset['Dataset'],
        Method=func.__name__,
        P_value=p_value,
        GT=indep,
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

    EXP_NUM = 'exp5'
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
    alpha = data_config['alpha']

    # DATA GEN----------------------------------------------------------
    datasets = []
    for dataset in data_config['datasets']:
        df = pd.read_csv(f'data/{dataset}.csv')
        A = pd.read_csv(f'data/{dataset}_dag.csv', index_col='source').values.astype(bool)
        n_nodes = df.shape[1]
        data = df.values[:data_config['N']]
        rng = np.random.RandomState(0)
        indeps = []
        deps = []
        for i, j in combinations(range(n_nodes), 2):
            if A[i, j] or A[j, i]:
                for _ in range(5):
                    K = rng.choice([k for k in range(n_nodes) if i != k != j], size=n_nodes - 4, replace=False)
                    X = data[:, i]
                    Y = data[:, j]
                    Z = data[:, K]
                    deps.append(dict(
                        Dataset=dataset,
                        data=(X, Y, Z),
                        indep=0,
                    ))
            else:
                par_X = set(np.nonzero(A[:, i])[0])
                par_Y = set(np.nonzero(A[:, j])[0])
                K = list(par_X | par_Y)
                if K:
                    X = data[:, i]
                    Y = data[:, j]
                    Z = data[:, K]
                    indeps.append(dict(
                        Dataset=dataset,
                        data=(X, Y, Z),
                        indep=1,
                    ))
        s = min(len(indeps), len(deps), data_config['n'])
        random.seed(dataset)
        d_all = list(random.sample(indeps, k=s) + random.sample(deps, k=s))
        print(f'{len(indeps) = }, {len(deps) = }, {data_config["n"] = }, {s = }')
        datasets.extend(d_all)
    
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