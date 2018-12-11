import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
import numpy as np
import scipy.stats as stats

def CPT(cpts, truncate=False):
    # truncate: set elements below zero to zero
    if truncate:
        cpts[cpts<0]=0
    return cpts.mean(axis=1)


def HPT(hpts):
    return hpts.mean(axis=1)


def MPT(mpts):
    # logarithm of geometric mean
    return np.log(stats.gmean(mpts,axis=1))


def composite_pain_sensitivity(CPT, HPT, MPT):
    # calculates composite pain sensitivity score as in:
    # Zunhammer, M. et al.(2016) Combined glutamate and glutamine levels in pain-processing brain regions are
    # associated with individual pain sensitivity, Pain, 157(10), pp.2248-2256.
    # DOI:10.1097/j.pain.0000000000000634
    return 0


