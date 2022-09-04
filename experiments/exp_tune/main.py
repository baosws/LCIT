import sys

sys.path.append('')
from argparse import ArgumentParser
from functools import partial
from multiprocessing import Pool

import numpy as np
import optuna
import pandas as pd
import torch
import yaml
from src.citests.LCIT import DualCNF
from src.data.data_gen import simulate
from src.utils.utils import Timer, strip_outliers
from torch.cuda import device_count
from tqdm import tqdm

optuna.logging.set_verbosity(optuna.logging.ERROR)

def objective(trial, X, Y, Z, gpus):
    N, dz = Z.shape
    model = DualCNF(
        dz=dz,
        n_components=int(2 ** trial.suggest_discrete_uniform('log_n_components', 3, 8, 1)),
        hidden_sizes=int(2 ** trial.suggest_discrete_uniform('log_hidden_size', 2, 8, 1)),
        verbose=False
    )
    logs = model.fit(X, Y, Z, max_epochs=100, gpus=gpus, verbose=False)
    val_loss = logs['val_loss']

    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()
    return -val_loss

def process(task):
    dataset_id, dataset = task
    print(f'Starting dataset {dataset_id}')
    X, Y, Z = dataset['data']
    indep = dataset['indep']
    Linearity = ['Non-linear', 'Linear'][dataset['Linearity']]
    d = dataset['d']
    N, dz = Z.shape
    X, Y, Z = map(lambda x: (x - np.mean(x)) / np.std(x), (X, Y, Z))
    X, Y, Z = map(strip_outliers, (X, Y, Z))
    X, Y, Z = map(lambda x: torch.tensor(x, dtype=torch.float32).view(N, -1), (X, Y, Z))

    gpus = [dataset_id % max(device_count(), 1)]
    with Timer() as t:
        study = optuna.create_study(direction="maximize")
        study.optimize(partial(objective, X=X, Y=Y, Z=Z, gpus=gpus), n_trials=20)

    res = dict(
        d=d,
        indep=indep,
        Linearity=Linearity,
        **study.best_params,
        value=study.best_value,
        Time=t.elapsed
    )
    print(f'Done dataset {dataset_id}')
    return res

if __name__ == '__main__':
    # CONFIG------------------------------------------------------------
    parser = ArgumentParser()
    parser.add_argument('--n_jobs', type=int, default=1)
    args = parser.parse_args()

    EXP_NUM = 'exp_tune'
    config_path = f'experiments/{EXP_NUM}/config'
    with open(f'{config_path}/data.yml', 'r') as f:
        data_config = yaml.safe_load(f)
    T = data_config['T']
    N = data_config['N']

    # DATA GEN----------------------------------------------------------
    datasets = []
    res = []
    for Linearity in data_config['Linearity']:
            for d in data_config['d']:
                for i in range(T):
                    X, Y, Z, indep = simulate(N, d, indep=i < T // 2, linear_only=Linearity, random_state=i)
                    datasets.append(dict(
                        data=(X, Y, Z),
                        d=d,
                        indep=indep,
                        Linearity=Linearity,
                    ))

    # RUN---------------------------------------------------------------
    tasks = list(enumerate(datasets))
    if args.n_jobs > 1:
        with Pool(args.n_jobs) as p:
            res = list(tqdm(p.imap_unordered(process, tasks, chunksize=4), total=len(tasks)))
    else:
        res = list(map(process, tqdm(tasks)))

    # VISUALIZE---------------------------------------------------------
    df = pd.DataFrame(res)
    path = f'experiments/{EXP_NUM}/results/result.csv'
    df.to_csv(path)
