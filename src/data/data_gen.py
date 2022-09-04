import numpy as np
from itertools import product

def linear(h, rng):
    h = h / np.std(h)
    a = rng.uniform(0.2, 2)
    return a * h

def square(X, rng=None):
    X = X / np.std(X)
    return X ** 2

def cube(X, rng=None):
    X = X / np.std(X)
    return X ** 3

def inverse(X, rng=None):
    X = X / np.std(X)
    return 1 / (X - np.min(X) + 1)

def nexp(X, rng=None):
    X = X / np.std(X)
    return np.exp(-X)

def constant(X, rng=None):
    return 1

def identity(X, rng=None):
    return X

def tanh(X, rng=None):
    X = X / np.std(X)
    return np.tanh(X)

def cos(X, rng=None):
    X = X / np.std(X)
    return np.cos(X)

def sigmoid(X, rng=None):
    X = X / np.std(X)
    return 1 / (1 + np.exp(-X))

def simulate(N, d, indep=None, linear_only=None, noise_coeff=5, random_state=None):
    rng = np.random.RandomState(random_state)
    if linear_only is None:
        linear_only = rng.randint(2)
    def gaussian_noise(*size):
        return rng.randn(*size) * noise_coeff
    def uniform_noise(*size):
        return rng.uniform(-1, 1, size=size) * noise_coeff
    def laplace_noise(*size):
        return rng.laplace(size=size) * noise_coeff
        
    funcs = [linear] if linear_only else [linear, cube, tanh, square, nexp, inverse, sigmoid]
    funcs = list(product(funcs, repeat=2))
    f1, f2 = funcs[rng.choice(range(len(funcs)))]

    noises = [uniform_noise, gaussian_noise, laplace_noise]
    noise = noises[rng.randint(len(noises))]
    X = 2 * noise(N)
    a, b = rng.uniform(-1, 1, size=(2, d))
    c = rng.uniform(1, 2)
    Z = f1(np.outer(X, a) + noise(N, d), rng=rng)
    if indep is None:
        indep = rng.randint(2)
    Y = f2(Z @ b + (1 - indep) * c * X + noise(N), rng=rng)
    return X, Y, Z, int(indep)