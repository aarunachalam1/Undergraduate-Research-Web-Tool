import math
import numpy as np

from scipy.stats import t

def student_t(x, alpha=0.05, side="lower"):
    """
    Two-sided Student-t confidence interval for the mean.

    Parameters:
        x : list or np.ndarray
            Sample values (1D).
        alpha : float
            Significance level (0.05 for 95% CI).

    Returns:
        (lower, upper) : tuple of floats
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 2:
        raise ValueError("Need at least 2 samples for Student-t CI")

    mean = np.mean(x)
    s = np.std(x, ddof=1)              # unbiased sample std
    t_crit = t.ppf(1 - alpha / 2, df=n - 1)
    margin = t_crit * s / np.sqrt(n)

    if side == "lower":
        return mean - margin
    elif side == "upper":
        return mean + margin
    else:
        raise ValueError("side must be 'lower' or 'upper'")

def anderson_bound(x, alpha, side="lower"):
    n = len(x)
    i = np.arange(1, n + 1)
    u_DKW = np.maximum(0, i / n - np.sqrt(np.log(2 / alpha) / (2 * n)))

    zu_i = np.sort(x)
    zu_iplus1 = np.append(zu_i, 1)[1:]
    zl_i = np.flip(1 - zu_i)
    zl_iplus1 = np.append(zl_i, 1)[1:]
    u = 1 - np.sum(u_DKW * (zu_iplus1 - zu_i))
    l = np.sum(u_DKW * (zl_iplus1 - zl_i))
    if side == "lower":
        return l
    elif side == "upper":
        return u
    else:
        raise ValueError("side must be 'lower' or 'upper'")

def hoeffding_bound(x, alpha=0.05, side="upper", bounds=(0,1)):
    """One-sided Hoeffding bound for the mean with known support [a,b].
       For side='upper': returns upper bound on mean.
       For side='lower': returns lower bound on mean.
    """
    a, b = bounds
    x = np.asarray(x, dtype=float)

    if not np.all((x >= a) & (x <= b)): raise ValueError(f"All samples must lie in [{a},{b}]")

    n = x.size
    if n == 0: raise ValueError("empty sample")

    width = b - a
    if width <= 0: raise ValueError("bounds must have b>a")
    mean = x.mean()

    # Hoeffding: P( m - E[m] >= t ) <= exp(-2 n t^2 / width^2)
    # -> t = sqrt( width^2 * log(1/alpha) / (2n) )
    t = math.sqrt((width**2) * math.log(1.0/alpha) / (2.0 * n))

    if side == "upper":
        return float(min(b, mean + t))
    elif side == "lower":
        return float(max(a, mean - t))
    else:
        raise ValueError("side must be 'upper' or 'lower'")