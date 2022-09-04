# Latent representation based Conditional Independence Test
This is the implementation of our paper: Bao Duong and Thin Nguyen. Conditional Independence Testing via Latent Representation Learning. In IEEE International Conference on Data Mining (ICDM), 2022.

## Dependencies

- Common packages: numpy, pandas, scikit-learn, seaborn, scipy, tqdm, yaml.
- For LCIT: PyTorch 1.12, PyTorch-Lightning 1.5.3.
- (Optional) For KCIT: causal-learn 0.1.2.3 (https://github.com/cmu-phil/causal-learn).
- (Optional) For CCIT: CCIT 0.4 (https://github.com/rajatsen91/CCIT).

## Demo:

```python
import numpy as np, seaborn as sns, matplotlib.pyplot as plt
from src.citests import LCIT
np.random.seed(0)

# X <- Z -> Y
N, d = 200, 3
Z = np.random.uniform(-2, 2, size=(N, d))
X = Z @ np.random.randn(d) + np.random.uniform(-0.2, 0.2, size=N)
Y = Z @ np.random.randn(d) + np.random.uniform(-0.2, 0.2, size=N)

e_x, e_y, p_value = LCIT(X, Y, Z, return_latents=True)
sns.jointplot(x=e_x, y=e_y)
plt.show()
print(f'{p_value = :.2f}') # 0.51
if p_value > 0.05:
    print('[Correct] Failed to reject H0 (X _||_ Y | Z)')
else:
    print('[Incorrect] Reject H0 (X _||_ Y | Z)')

# X -> Z -> Y <- X
N, d = 1000, 10
X = np.random.uniform(-0.2, 0.2, size=N)
Z = np.outer(X, np.random.randn(d) * 5) + np.random.uniform(-0.2, 0.2, size=(N, d))
Y = X * np.random.randn() + Z @ np.random.randn(d) + np.random.uniform(-0.2, 0.2, size=N)

p_value = LCIT(X, Y, Z, n_components=16, hidden_sizes=[4, 4])
print(f'{p_value = :.2f}') # 0.00
if p_value > 0.1:
    print('[Incorrect] Failed to reject H0 (X _||_ Y | Z)')
else:
    print('[Correct] Reject H0 (X _||_ Y | Z)')
```

## Running experiments

For example, to run the "Effect of sample size" experiment (Figure 4 in the paper):
```
python experiments/exp_samplesize/main.py --methods LCIT SCIT --n_jobs=8
```
where available `methods` are CCIT, KCIT, SCIT, and LCIT; `n_jobs` is the number of parallel jobs to run.

Modifiable configurations are stored in `experiments/exp_*/config/`, and result dataframes are stored in `experiments/exp_*/results/` after the command is finished.

## Citation
