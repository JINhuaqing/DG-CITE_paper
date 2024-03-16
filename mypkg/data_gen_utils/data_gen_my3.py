# generate data with my own setting
# compare with Lei, I 
# - change the error dist to 
# - change the tau fun a bit 
import scipy.stats as ss
import numpy as np
from easydict import EasyDict as edict

def Xfun(n, d, rho):
    """
    Generate a random matrix X based on the specified parameters.

    Parameters:
    - n (int): Number of rows in the matrix.
    - d (int): Number of columns in the matrix.
    - rho (float): Correlation coefficient.

    Returns:
    - X (ndarray): Random matrix of shape (n, d) based on the specified parameters.
    """

    if rho > 0:
        X = ss.norm.rvs(size=(n, d))
        fac = ss.norm.rvs(size=(n, 1))
        X = X * np.sqrt(1-rho) + fac * np.sqrt(rho)
        X = ss.norm.cdf(X)
    elif rho == 0:
        X = ss.uniform.rvs(size=(n, d))
    
    return X


def taufun(X):
    """
    Calculate the tauv value based on the given input X.

    Parameters:
    X (numpy.ndarray): Input array of shape (n, d), where n is the number of samples.

    Returns:
    numpy.ndarray: Array of tauv values calculated based on the input X.
    """
    _f = lambda x: 2/(1+np.exp(-60*(x-0.5)));
    _f2 = lambda x: 4/(1+(x-0.5)**2) + 1
    _f3 = lambda x: np.exp(1+(x-0.5)**3) + 1
    
    _, d = X.shape
    d1 = d2 = int(d/4)
    d3 = d - d1 - d2
    vec1 = np.zeros(d)
    vec1[:d1] = np.linspace(1, 10, d1) * ((1)**np.arange(d1))
    vec1 = vec1/np.abs(vec1).sum()

    vec2 = np.zeros(d)
    vec2[:d2] = np.linspace(1, 10, d2) * ((1)**np.arange(d2))
    vec2 = vec2/np.abs(vec2).sum()

    vec3 = np.zeros(d)
    vec3[:d3] = np.linspace(1, 10, d3) * ((1)**np.arange(d3))
    vec3 = vec3/np.abs(vec3).sum()
    
    tauv = _f(X@vec1) * _f2(X@vec2) - _f3(X@vec3);
    return tauv

def psfun(X):
    """
    Calculate the probability density function of a beta distribution.

    Parameters:
    X (numpy.ndarray): Input array of shape (n, m).

    Returns:
    numpy.ndarray: Array of shape (n,) containing the probability density values.
    """
    return (1+ss.beta.cdf(X[:, 0], a=2, b=4))/4

def sdfun(X):
    """
    Calculate the similarity scores for the given input data.

    Parameters:
    X (numpy.ndarray): Input data.

    Returns:
    numpy.ndarray: Similarity scores.
    """
    d = X.shape[1]
    sds = np.ones(X.shape[0])
    Xnorm2 = np.linalg.norm(X, 2, axis=1)**2
    idxs1 = np.bitwise_and(Xnorm2<d/3, Xnorm2>d/6)
    idxs2 = np.bitwise_and(Xnorm2<d/6, Xnorm2>-1)
    sds[idxs1] = 2
    sds[idxs2] = 3
    return sds

def errdist(n, typ="norm"):
    if typ.lower().startswith("norm"):
        return ss.norm.rvs(size=n)
    elif typ.lower().startswith("t"):
        rvs = ss.t.rvs(3, size=n)
        return rvs/rvs.std()

def get_simu_data(n, d, rho=0, is_condition=False, err_type="norm"):
    """
    Generate simulated data for a causal inference problem.

    Parameters:
    n (int): Number of samples.
    d (int): Number of features.
    rho (float, optional): Correlation coefficient between features. Defaults to 0.
    is_condition (bool, optional): Flag indicating whether the data should be conditioned on a fixed value of X. Defaults to False.

    Returns:
    dataset (edict): A dictionary containing the generated data.
        - X (ndarray): The feature matrix of shape (n, d).
        - ps (ndarray): The propensity scores of shape (n,).
        - Y (ndarray): The outcome variable of shape (n,).
        - Y1 (ndarray): The potential outcome under treatment of shape (n,).
        - T (ndarray): The treatment assignment indicator of shape (n,).
    """
    if is_condition:
        X = np.ones((n, 1)) * Xfun(1, d, rho)
    else:
        X = Xfun(n, d, rho)
    tau = taufun(X)
    std = sdfun(X)
    ps = psfun(X)
    
    Y0 = np.zeros(n)
    Y1 = tau + std*errdist(n, err_type)
    T = ss.uniform.rvs(size=n) < ps;
    Y = Y0.copy()
    Y[T] = Y1[T]
    
    dataset = edict()
    dataset.X = X
    dataset.ps = ps
    dataset.Y = Y
    dataset.Y1 = Y1
    dataset.T = T
    dataset.tau = tau
    return dataset
