import logging
import warnings

import numpy as np
import torch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from scipy.stats import norm
from src.utils.utils import strip_outliers
from torch import nn
from torch.distributions.normal import Normal
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset, random_split

logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
logging.basicConfig()
logger = logging.getLogger(__name__)

EPS = 1e-6

def MLP(in_features: int, out_features: int, hidden_sizes: None, norm=True, activator=nn.ReLU, activator_params=None):
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
        std = torch.clip(std, min=1e-6, max=None)
        w = self.z_to_w(Z) # N x k

        dist = Normal(mu, std) # N x k
        e = (dist.cdf(X) * w).sum(axis=1, keepdims=True)
        p_hat = (dist.log_prob(X).exp() * w).sum(axis=1, keepdims=True)
        p_hat = torch.clip(p_hat, min=1e-24, max=None)
        log_de_dx = p_hat.log()
        
        return e, log_de_dx

class DualCNF(LightningModule):
    def __init__(self, dz, n_components, hidden_sizes, lr, weight_decay, verbose):
        super().__init__()
        self.save_hyperparameters()

        self.x_cnf = UniformFlow(dz=dz, n_components=n_components, hidden_sizes=hidden_sizes, activator_params=dict(inplace=True))
        self.y_cnf = UniformFlow(dz=dz, n_components=n_components, hidden_sizes=hidden_sizes, activator_params=dict(inplace=True))

    def loss(self, X, Y, Z):
        ex, log_px = self.x_cnf(X, Z)
        ey, log_py = self.y_cnf(Y, Z)

        loss = -torch.mean(log_px + log_py)
        return loss

    def training_step(self, batch, batch_idx):
        X, Y, Z = batch
        loss = self.loss(X, Y, Z)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        X, Y, Z = batch
        loss = self.loss(X, Y, Z)

        return loss
    
    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack(outputs).mean()
        self.log('val_loss', avg_loss)
        
    def transform(self, X, Y, Z):
        self.eval()
        ex, log_px = self.x_cnf(X, Z)
        ey, log_py = self.y_cnf(Y, Z)

        ex = ex.detach().cpu().numpy()
        ey = ey.detach().cpu().numpy()

        return ex, ey

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, factor=.1, patience=10, verbose=self.hparams.verbose)
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
                verbose=verbose
            )
            callbacks.append(early_stopping)
            trainer = Trainer(
                accelerator="gpu" if torch.cuda.device_count() else "cpu",
                gpus=gpus if torch.cuda.device_count() else 0,
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

def LCIT(X, Y, Z, normalize=True, n_components=32, hidden_sizes=4, lr=5e-3, weight_decay=5e-5, max_epochs=100, random_state=0, gpus=0, return_latents=False, verbose=False, **kwargs):
    logger.setLevel(logging.DEBUG if verbose else logging.ERROR)
    if random_state is not None:
        torch.random.fork_rng(enabled=True)
        torch.random.manual_seed(random_state)

    N, dz = Z.shape
    if normalize:
        X, Y, Z = map(lambda x: (x - np.mean(x)) / np.std(x), (X, Y, Z))
        X, Y, Z = map(strip_outliers, (X, Y, Z))
    X, Y, Z = map(lambda x: torch.tensor(x, dtype=torch.float32).view(N, -1), (X, Y, Z))

    model = DualCNF(dz, n_components=n_components, hidden_sizes=hidden_sizes, lr=lr, weight_decay=weight_decay, verbose=verbose)
    model.fit(X, Y, Z, max_epochs=max_epochs, verbose=verbose, gpus=gpus)
    
    e_x, e_y = model.transform(X, Y, Z)
    e_x, e_y = map(lambda x: np.clip(x, EPS, 1 - EPS), (e_x, e_y))
    e_x, e_y = map(np.squeeze, (e_x, e_y))
    e_x, e_y = map(norm.ppf, (e_x, e_y))

    r = np.corrcoef(e_x, e_y)[0, 1]
    r = np.clip(r, -1 + EPS, 1 - EPS)
    stat = 0.5 * np.sqrt(N - 3) * np.log1p(2 * r / (1 - r))
    p_value = 2 * (1 - norm.cdf(abs(stat)))

    if return_latents:
        return e_x, e_y, p_value

    return p_value