import numpy as np

def SCIT(X, Y, Z, n_bootstraps=100, gamma=2, normalize=True, vectorized_permutation=False, random_state=None, **kwargs):
    rd = np.random.RandomState(random_state)
    if normalize:
        X, Y, Z = map(lambda x: (x - np.mean(x)) / np.std(x), (X, Y, Z))
    N, d = Z.shape
    M = Z @ np.linalg.inv(Z.T @ Z) @ Z.T if d else np.zeros((N, N))
    H = np.eye(N) - M
    L = np.mean(np.exp(-gamma * np.square(H @ (X - Y))))
    if vectorized_permutation:
        idx = np.argsort(rd.randn(n_bootstraps, N), axis=1)
        Lp = np.mean(np.exp(-gamma * np.square(H @ X - H[idx] @ Y)), axis=1)
        p_value = np.mean(L <= Lp)
    else:
        c = 0
        for _ in range(n_bootstraps):
            idx = rd.permutation(N)
            Lp = np.mean(np.exp(-gamma * np.square(H @ X - H[idx, :] @ Y)))
            c += L <= Lp
        p_value = c / n_bootstraps
    return p_value