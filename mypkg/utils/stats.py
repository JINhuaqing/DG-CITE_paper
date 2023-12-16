import numpy as np
import scipy.stats as ss

def get_pdf_at_xx(Y1, Y2):
    """
    Calculate the probability density function (PDF) for two sets of data, Y1 and Y2, at a given range of values.

    Parameters:
    Y1 (array-like): First set of data.
    Y2 (array-like): Second set of data.

    Returns:
    xx (ndarray): Array of values within the range of Y1 and Y2.
    p_pdf1 (ndarray): PDF values for Y1 at each value in xx.
    p_pdf2 (ndarray): PDF values for Y2 at each value in xx.
    """
    pdf1 = ss.gaussian_kde(Y1)
    pdf2 = ss.gaussian_kde(Y2)
    l1, u1 = np.quantile(Y1, [0.01, 0.99])
    l2, u2 = np.quantile(Y2, [0.01, 0.99])
    xmin = min(l1, l2)
    xmax = max(u1, u2)
    xx = np.linspace(xmin, xmax, 100)
    p_pdf1 = pdf1(xx);
    p_pdf2 = pdf2(xx);
    return xx, p_pdf1, p_pdf2
def get_kl(Y1, Y2):
    """
    Calculate the Kullback-Leibler (KL) divergence between two data spses.

    Parameters:
    Y1 (array-like): The first data sps.
    Y2 (array-like): The second sps .

    Returns:
    tuple: A tuple containing the KL divergence from Y1 to Y2 and the KL divergence from Y2 to Y1.
    """
    xx, p1, p2 = get_pdf_at_xx(Y1, Y2)
    kl_div_12 = np.sum(np.where(p1 != 0, p1 * np.log(p1 / p2), 0))
    kl_div_21 = np.sum(np.where(p2 != 0, p2 * np.log(p2 / p1), 0))
    return kl_div_12, kl_div_21


# from https://stackoverflow.com/questions/21844024/weighted-percentile-using-numpy
def weighted_quantile(values, quantiles, sample_weight=None, 
                      values_sorted=False, old_style=False):
    """ Very close to numpy.percentile, but supports weights.
    NOTE: quantiles should be in [0, 1]!
    :param values: numpy.array with data
    :param quantiles: array-like with many quantiles needed
    :param sample_weight: array-like of the same length as `array`
    :param values_sorted: bool, if True, then will avoid sorting of
        initial array
    :param old_style: if True, will correct output to be consistent
        with numpy.percentile.
    :return: numpy.array with computed quantiles.
    """
    values = np.array(values)
    quantiles = np.array(quantiles)
    if sample_weight is None:
        sample_weight = np.ones(len(values))
    sample_weight = np.array(sample_weight)
    assert np.all(quantiles >= 0) and np.all(quantiles <= 1), \
        'quantiles should be in [0, 1]'

    if not values_sorted:
        sorter = np.argsort(values)
        values = values[sorter]
        sample_weight = sample_weight[sorter]

    weighted_quantiles = np.cumsum(sample_weight) - 0.5 * sample_weight
    if old_style:
        # To be convenient with numpy.percentile
        weighted_quantiles -= weighted_quantiles[0]
        weighted_quantiles /= weighted_quantiles[-1]
    else:
        weighted_quantiles /= np.sum(sample_weight)
    return np.interp(quantiles, weighted_quantiles, values)