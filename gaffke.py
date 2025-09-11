import numpy as np

def gaffke_CI(x, alpha=0.05, B=100000, side="lower", bounds=(0, 1)):
    """
    Gaffke bound as described in Learned-Miller and Thomas (https://arxiv.org/pdf/1905.06208.pdf)

    Parameters:
    - x: 1D array of iid random samples from a distribution with known bounds
    - alpha: significance level (e.g., 0.05 for a 95% confidence bound)
    - B: number of Monte Carlo iterations
    - side: 'upper' or 'lower' for one-sided bound
    - bounds: tuple (min, max) of the population range (default is (0, 1))

    Returns:
    - One-sided (1 - alpha) confidence bound on the population mean
    """
    x = np.asarray(x)
    if np.isnan(x).any():
        raise ValueError("Samples have NA/NaN values")

    x = (x - bounds[0]) / (bounds[1] - bounds[0])

    if side == "lower":
        x = 1 - x

    n = len(x)
    z = np.sort(x)
    s = np.concatenate((np.diff(z), [1 - z[-1]]))

    ms = np.empty(B)
    for i in range(B):
        u = np.sort(np.random.uniform(0, 1, n))
        ms[i] = 1 - np.sum(u * s)

    ms_alpha = np.quantile(ms, 1 - alpha)
    ms_alpha = ms_alpha * (bounds[1] - bounds[0]) + bounds[0]

    if side == "lower":
        return 1 - ms_alpha
    elif side == "upper":
        return ms_alpha
    else:
        raise ValueError("side must be 'upper' or 'lower'")