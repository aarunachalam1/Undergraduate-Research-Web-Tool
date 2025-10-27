import numpy as np


def gaffke_ci_upper(x, conf=0.95, MC=100_000, U=1.):

    """
    Gaffke upper bound as described in Learned-Miller and Thomas (https://arxiv.org/pdf/1905.06208.pdf)

    Parameters:
        - x: 1D array of iid random samples from a distribution with known upper bound
             of the population range, i.e., all samples must be <= U.
        - conf: confidence level (e.g., 0.95 for a 95% confidence bound)
        - MC: number of Monte Carlo iterations
        - U: the max value the support set has (default is  1)

    Returns:
        - the upper bound of the mean
    """
    x = np.asarray(x)

    n = len(x)
    z = np.sort(x)

    zplus = np.append(z, U)
    s = zplus[1:] - zplus[:-1]

    u = np.sort(np.random.uniform(0, 1, (MC, n)), axis=1)
    ms = U - np.dot(u, s)
    ms_alpha = np.quantile(ms, conf)

    return ms_alpha

def gaffke_ci_lower(x, conf=0.95, MC=100000, L=0.):
    x = -np.asarray(x)
    U = -L
    return - gaffke_ci_upper(x, conf=conf, MC=MC, U=U)


def gaffke_CI(x, conf=0.95, B=100000, side="lower", extrema=0.):
    if side == "upper":
        return gaffke_ci_upper(x, conf=conf, MC=B, U=extrema)
    elif side == "lower":
        return gaffke_ci_lower(x, conf=conf, MC=B, L=extrema)
    else:
        raise ValueError("side must be 'upper' or 'lower'")


if __name__ == "__main__":

    sample = [0, 3, 4,5 ,6]

    for _ in range(20):
        print(gaffke_CI(sample, alpha=0.05, B=100000, side="lower", bounds=(0, 1)))