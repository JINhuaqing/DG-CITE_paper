# generate data based on (Lei, 2021, JRSSB)
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
    elif rho == 0:
        X = ss.norm.rvs(size=(n, d))
    return X

def taufun(X, beta):
    """
    Calculate the tauv value based on the given input X.

    Parameters:
    X (numpy.ndarray): Input array of shape (n, d), where n is the number of samples.
    beta (numpy.ndarray): d-vector

    Returns:
    numpy.ndarray: Array of tauv values calculated based on the input X.
    """
    return X @ beta
def psfun(X):
    """
    Calculate the probability density function of a beta distribution.

    Parameters:
    X (numpy.ndarray): Input array of shape (n, m).

    Returns:
    numpy.ndarray: Array of shape (n,) containing the probability density values.
    """
    return (1+ss.beta.cdf(np.abs(X[:, 0]), a=2, b=4))/4

def sdfun(X, is_homo=True):
    """
    Calculate the similarity scores for the given input data.

    Parameters:
    X (numpy.ndarray): Input data.
    is_homo (bool, optional): Flag indicating whether the data is homogeneous. Defaults to True.

    Returns:
    numpy.ndarray: Similarity scores.
    """
    if is_homo:
        sds = np.ones(X.shape[0])
    else:
        sds = np.abs(np.log(np.abs(X[:, 0])+1e-9))
    return sds

def errdist(n):
    return ss.norm.rvs(size=n)

def get_simu_data(n, d, beta, rho=0, is_homo=True, inp=0, is_condition=False):
    """
    Generate simulated data for a causal inference problem.

    Parameters:
    n (int): Number of samples.
    d (int): Number of features.
    rho (float, optional): Correlation coefficient between features. Defaults to 0.
    is_homo (bool, optional): Flag indicating whether the standard deviation of the error term is homogeneous across samples. Defaults to True.
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
    tau = taufun(X, beta) + inp
    std = sdfun(X, is_homo)
    ps = psfun(X)
    
    Y0 = np.zeros(n)
    Y1 = tau + std*errdist(n)
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
    dataset.beta = beta
    dataset.inp = inp
    return dataset
