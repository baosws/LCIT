from CCIT.CCIT import CCIT as CCIT_
import numpy as np

def CCIT(X, Y, Z, normalize=True, nthread=1, **kwargs):
    N, dz = Z.shape
    X = X.reshape(N, -1)
    Y = Y.reshape(N, -1)
    if normalize:
        X, Y, Z = map(lambda x: (x - np.mean(x)) / np.std(x), (X, Y, Z))
    p_value = CCIT_(X, Y, Z, nthread=nthread)
    return p_value