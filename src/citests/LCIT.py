import logging
from typing import Any, List
import warnings
import numpy as np, pandas as pd
from sklearn.metrics import pairwise_distances
from scipy.stats import norm
from torch.distributions.normal import Normal
from sklearn.preprocessing import LabelEncoder
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from src.utils.utils import strip_outliers

logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
logging.basicConfig()
logger = logging.getLogger(__name__)

def MLP(in_features: int, out_features: int, hidden_sizes: List[int]=None, norm=True, activator: Any=nn.ReLU, activator_params: dict=None):
    activator_params = activator_params or {}
    if isinstance(hidden_sizes, int):
        hidden_sizes = [hidden_sizes]
    hidden_sizes = [in_features] + (hidden_sizes or []) + [out_features]

    layers = []
    for i in range(len(hidden_sizes) - 1):
        layers.extend([nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]), activator(**activator_params)])
        if norm:
            layers.append(nn.BatchNorm1d(hidden_sizes[i + 1]))
    
    return nn.Sequential(*layers)

class UniformFlow(nn.Module):
    def __init__(self, dz, n_components, hidden_sizes, activator=nn.LeakyReLU, activator_params=None):
        super().__init__()
        activator_params = activator_params or {}
        self.z_to_mu = MLP(dz, n_components, hidden_sizes=hidden_sizes, norm=True, activator=activator, activator_params=activator_params)
        self.z_to_logstd = MLP(dz, n_components, hidden_sizes=hidden_sizes, norm=True, activator=activator, activator_params=activator_params)
        self.z_to_w = nn.Sequential(
            *MLP(dz, n_components, hidden_sizes=hidden_sizes, norm=True, activator=activator, activator_params=activator_params),
            nn.Softmax(dim=1)
        )

    def forward(self, X, Z):
        X = X.view(X.shape[0], -1) # N x 1
        mu = self.z_to_mu(Z) # N x k
        std = self.z_to_logstd(Z).exp() # N x k
        std = torch.clip(std, 1e-6, None)
        w = self.z_to_w(Z) # N x k
        dist = Normal(mu, std) # N x k
        e = (dist.cdf(X) * w).sum(axis=1, keepdims=True)
        log_de_dx = (dist.log_prob(X).exp() * w).sum(axis=1, keepdims=True).log()
        return e, log_de_dx

class DQFlow(nn.Module):
    def __init__(self, dz, n_components, hidden_sizes, activator=nn.LeakyReLU, activator_params=None):
        super().__init__()
        activator_params = activator_params or {}
        self.z_to_mu = MLP(dz, n_components, hidden_sizes=hidden_sizes, norm=True, activator=activator, activator_params=activator_params)
        self.z_to_logstd = MLP(dz, n_components, hidden_sizes=hidden_sizes, norm=True, activator=activator, activator_params=activator_params)
        self.z_to_w = nn.Sequential(
            *MLP(dz, n_components, hidden_sizes=hidden_sizes, norm=True, activator=activator, activator_params=activator_params),
            nn.Softmax(dim=1)
        )

    def forward(self, X, Z):
        X = X.view(X.shape[0], -1) # N x 1
        Z = torch.column_stack((X, Z))
        dist_eps = Normal(0, 1)
        e = torch.randn_like(X)
        mu = self.z_to_mu(Z) # N x k
        std = self.z_to_logstd(Z).exp() # N x k
        std = torch.clip(std, 1e-6, None)
        w = self.z_to_w(Z) # N x k
        dist = Normal(mu, std) # N x k
        u = (dist.cdf(e) * w).sum(axis=1, keepdims=True)
        log_du_de = (dist.log_prob(e).exp() * w).sum(axis=1, keepdims=True).log()
        log_qu = dist_eps.log_prob(e) - log_du_de
        return u, log_qu

class LogitTransform(nn.Module):
    '''
    https://github.com/TinyVolt/normalizing-flows/blob/main/1d_composing_flows/model.py
    '''
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha

    def forward(self, X, Z):
        X_new = self.alpha * 0.5 + (1 - self.alpha) * X
        e = torch.log(X_new) - torch.log(1 - X_new)
        log_de_dx = torch.log(torch.FloatTensor([1 - self.alpha])) - torch.log(X_new) - torch.log(1 - X_new)
        return e, log_de_dx

class ComposableFlow(nn.Module):
    def __init__(self, flows):
        super().__init__()
        self.flows = nn.ModuleList(flows)
    
    def forward(self, X, Z):
        e, log_px = X, 0
        for flow in self.flows:
            e, log_de_dx = flow(e, Z)
            log_px = log_px + log_de_dx
        return e, log_px

EPS = 1e-6
class CCNF(LightningModule):
    def __init__(self, dz, X_cat, Y_cat, n_components, hidden_sizes, lr, weight_decay, verbose):
        super().__init__()
        self.save_hyperparameters()
        self.validation_outputs = []

        self.x_cnf = UniformFlow(dz=dz, n_components=n_components, hidden_sizes=hidden_sizes, activator_params=dict(inplace=True))
        self.y_cnf = UniformFlow(dz=dz, n_components=n_components, hidden_sizes=hidden_sizes, activator_params=dict(inplace=True))

        if X_cat:
            self.x_dequants = DQFlow(dz=dz + 1, n_components=n_components, hidden_sizes=hidden_sizes, activator_params=dict(inplace=True))
        if Y_cat:
            self.y_dequants = DQFlow(dz=dz + 1, n_components=n_components, hidden_sizes=hidden_sizes, activator_params=dict(inplace=True))

    def loss(self, X, Y, Z):
        loss = 0
        if self.hparams.X_cat:
            ux, log_qu = self.x_dequants(X, Z)
            X = X + ux
            loss -= log_qu
        if self.hparams.Y_cat:
            uy, log_qy = self.y_dequants(Y, Z)
            Y = Y + uy
            loss -= log_qy
        ex, log_px = self.x_cnf(X, Z)
        ey, log_py = self.y_cnf(Y, Z)

        loss += log_px + log_py
        loss = -torch.mean(loss)
        return loss

    def training_step(self, batch, batch_idx):
        X, Y, Z = batch
        loss = self.loss(X, Y, Z)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        X, Y, Z = batch
        loss = self.loss(X, Y, Z)
        self.validation_outputs.append(loss)
        return loss
    
    def on_validation_epoch_end(self):
        avg_loss = torch.stack(self.validation_outputs).mean()
        self.log('val_loss', avg_loss)
        
    def transform(self, X, Y, Z):
        self.eval()
        if self.hparams.X_cat:
            ux, log_pux = self.x_dequants(X, Z)
            X = X + ux
        if self.hparams.Y_cat:
            uy, log_puy = self.y_dequants(X, Z)
            Y = Y + uy
        ex, log_px = self.x_cnf(X, Z)
        ey, log_py = self.y_cnf(Y, Z)

        ex = ex.detach().cpu().numpy()
        ey = ey.detach().cpu().numpy()

        return ex, ey

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, factor=.1, patience=10)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "train_loss"}

    def fit(self, X, Y, Z, max_epochs, verbose, gpus=None, callbacks=None):
        gpus = gpus or 0
        callbacks = callbacks or []
        N, dz = Z.shape
        train_size = int(N * 0.7)
        valid_size = N - train_size
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            train, valid = random_split(TensorDataset(X, Y, Z), lengths=[train_size, valid_size])
            train_dataloader = DataLoader(train, batch_size=64, shuffle=True)
            val_dataloader = DataLoader(valid, batch_size=64)
            early_stopping = EarlyStopping(
                mode='min',
                monitor='val_loss',
                patience=10,
                # check_on_train_epoch_end=True,
                verbose=verbose
            )
            callbacks.append(early_stopping)
            trainer = Trainer(
                accelerator="gpu" if torch.cuda.device_count() else "cpu",
                max_epochs=max_epochs,
                logger=verbose,
                enable_checkpointing=verbose,
                enable_progress_bar=verbose,
                enable_model_summary=verbose,
                deterministic=True,
                callbacks=callbacks,
                detect_anomaly=verbose,
                gradient_clip_val=1,
                gradient_clip_algorithm="value"
            )
            trainer.fit(model=self, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
            logs, = trainer.validate(model=self, dataloaders=val_dataloader, verbose=verbose)
        
        return logs

def infer_types(X, Y, Z, rng):
    X_cat = X.dtype == 'object'
    Y_cat = Y.dtype == 'object'
    df = pd.DataFrame(np.column_stack((X, Y, Z)))
    cat_cols = df.infer_objects().select_dtypes(include='object').columns
    N, d = df.shape
    assert N == X.shape[0]

    for col in cat_cols:
        if col in [0, 1, '0', '1']:
            df[col] = LabelEncoder().fit_transform(df[col])

    assert N == df.shape[0]
    X = df[0].infer_objects().values
    Y = df[1].infer_objects().values
    df = 1 * pd.get_dummies(df, columns=[col for col in cat_cols if col not in [0, 1, '0', '1']])
    assert df.columns[0] == 0
    assert df.columns[1] == 1
    Z = df.iloc[:, 2:].infer_objects().values

    return (X, X_cat), (Y, Y_cat), Z

def LCIT(X, Y, Z, n_bootstraps=0, normalize=True, n_components=32, hidden_sizes=4, lr=5e-3, weight_decay=5e-5, max_epochs=100, random_state=0, gpus=0, return_latents=False, verbose=False, **kwargs):
    logger.setLevel(logging.DEBUG if verbose else logging.ERROR)
    rng = np.random.RandomState(random_state)
    if random_state is not None:
        torch.random.fork_rng(enabled=True)
        torch.random.manual_seed(random_state)
    (X, X_cat), (Y, Y_cat), Z = infer_types(X, Y, Z, rng)
    print(f'Inferred: {X_cat = }, {Y_cat = }')

    N, dz = Z.shape
    if normalize:
        if not X_cat:
            X = (X - np.mean(X)) / np.std(X)
            X = strip_outliers(X)
        if not Y_cat:
            Y = (Y - np.mean(Y)) / np.std(Y)
            Y = strip_outliers(Y)
        Z = (Z - np.mean(Z)) / np.std(Z)
        Z = strip_outliers(Z)
    X, Y, Z = map(lambda x: torch.tensor(x, dtype=torch.float32).view(N, -1), (X, Y, Z))

    model = CCNF(dz=dz, X_cat=X_cat, Y_cat=Y_cat, n_components=n_components, hidden_sizes=hidden_sizes, lr=lr, weight_decay=weight_decay, verbose=verbose)
    model.fit(X, Y, Z, max_epochs=max_epochs, verbose=verbose, gpus=gpus)
    
    e_x, e_y = model.transform(X, Y, Z)
    e_x, e_y = map(lambda x: np.clip(x, EPS, 1 - EPS), (e_x, e_y))

    if n_bootstraps:
        A = pairwise_distances(e_x)
        B = pairwise_distances(e_y)
        A = A - np.mean(A, axis=0, keepdims=True) - np.mean(A, axis=1, keepdims=True) + np.mean(A)
        B = B - np.mean(B, axis=0, keepdims=True) - np.mean(B, axis=1, keepdims=True) + np.mean(B)
        s0 = np.mean(A * B)

        c = 0
        rng = np.random.RandomState(random_state)
        for i in range(n_bootstraps):
            idx = rng.permutation(N)
            B_perm = B[idx[:, None], idx[None, :]]
            s_perm = np.mean(A * B_perm)
            c += s0 <= s_perm

        p_value = c / n_bootstraps
    else:
        e_x, e_y = map(np.squeeze, (e_x, e_y))
        e_x, e_y = map(norm.ppf, (e_x, e_y))
        r = np.corrcoef(e_x, e_y)[0, 1]
        assert r == r
        r = np.clip(r, -1 + EPS, 1 - EPS)
        z_stat = 0.5 * np.sqrt(N - 3) * np.log1p(2 * r / (1 - r))
        assert z_stat == z_stat
        p_value = 2 * (1 - norm.cdf(abs(z_stat)))
        assert p_value == p_value

    if return_latents:
        return e_x, e_y, p_value

    return p_value