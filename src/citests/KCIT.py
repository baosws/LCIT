from causallearn.utils.cit import kci
import pandas as pd, numpy as np

# causal-learn==0.1.2.3
def KCIT(X, Y, Z, **kargs):
    data = np.column_stack((X, Y, Z))
    dz = Z.shape[1]
    p_value = kci(data, 0, 1, list(range(2, 2 + dz)))
    return p_value