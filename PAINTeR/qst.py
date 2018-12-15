import numpy as np
import scipy.stats as stats

def CPT(cpts, truncate=False):
    # truncate: set elements below zero to zero
    if truncate:
        cpts[cpts < 0] = -0.1
    return np.nanmean(cpts, axis=1)


def HPT(hpts):
    return hpts.mean(axis=1)


def MPT(mpts):
    # logarithm of geometric mean
    return np.log(stats.gmean(mpts,axis=1))


def WDT(wdts):
    return np.log(np.mean(wdts, axis=1))

def CDT(wdts):
    return np.log(np.mean(wdts, axis=1))

def MDT(mdts):
    return np.log(stats.gmean(mdts, axis=1))



def composite_pain_sensitivity(CPT, HPT, MPT):
    # calculates composite pain sensitivity score as in:
    # Zunhammer, M. et al.(2016) Combined glutamate and glutamine levels in pain-processing brain regions are
    # associated with individual pain sensitivity, Pain, 157(10), pp.2248-2256.
    # DOI:10.1097/j.pain.0000000000000634
    return (+ (CPT-np.nanmean(CPT))/np.nanstd(CPT)\
           - (HPT-np.nanmean(HPT))/np.nanstd(HPT)\
           - (MPT-np.nanmean(MPT))/np.nanstd(MPT) ) / 3



def composite_pain_sensitivity_PAINTeR(CPT, HPT, MPT, CPT_ex, HPT_ex, MPT_ex):
    # calculates composite pain sensitivity score, based on the passed example sample (Bochum sample expected)
    # Zunhammer, M. et al.(2016) Combined glutamate and glutamine levels in pain-processing brain regions are
    # associated with individual pain sensitivity, Pain, 157(10), pp.2248-2256.
    # DOI:10.1097/j.pain.0000000000000634
    return (+ (CPT - np.nanmean(CPT_ex)) / np.nanstd(CPT_ex) \
            - (HPT - np.nanmean(HPT_ex)) / np.nanstd(HPT_ex) \
            - (MPT - np.nanmean(MPT_ex)) / np.nanstd(MPT_ex)) / 3

