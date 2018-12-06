#!/usr/bin/env python
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
import numpy as np
import pandas as pd

import PAINTeR.load_project_data as load
import PAINTeR.train as train
import PAINTeR.models as models
from sklearn.metrics import r2_score, explained_variance_score, mean_squared_error
from sklearn import preprocessing
import nilearn.plotting as plotting
from nilearn.connectome import vec_to_sym_matrix
from sklearn.preprocessing import StandardScaler


########################################################################################################################
# parameters
########################################################################################################################

scrub_threshold = 0.1 # equal to 0.2 mm, for some reason there is a scaling, TODO: find out why
mean_FD_threshold = 0.1 # equal to 0.2 mm, for some reason there is a scaling, TODO: find out why
perc_scrub_threshold = 22 # equal to 30%?

scrubbing = True

########################################################################################################################
# scrub function
########################################################################################################################

def scrub(ts, fd, frames_before=0, frames_after=0):

    frames_out = np.argwhere(fd>scrub_threshold).flatten().tolist()
    extra_indices=[]
    for i in frames_out:
        # remove preceding frames
        if i > 0:
            count = 1
            while count <= frames_before:
                extra_indices.append(i - count)
                count += 1

        # remove following frames
        count = 1
        while count <= frames_after:
            if i+count < len(fd):  # do not censor unexistent data
                extra_indices.append(i + count)
            count += 1

    indices_out = list(set(frames_out) | set(extra_indices))
    indices_out.sort()

    return np.delete(ts, indices_out, axis=0)



########################################################################################################################
# input files
########################################################################################################################

bochum_ts_files =[
"/Users/tspisak/res/PAINTeR/bochum/regional_timeseries_renamed/bochum-004_pumi-0.tsv",
"/Users/tspisak/res/PAINTeR/bochum/regional_timeseries_renamed/bochum-005_pumi-1_ex.tsv",
"/Users/tspisak/res/PAINTeR/bochum/regional_timeseries_renamed/bochum-006_pumi-2.tsv",
"/Users/tspisak/res/PAINTeR/bochum/regional_timeseries_renamed/bochum-007_pumi-3.tsv",
"/Users/tspisak/res/PAINTeR/bochum/regional_timeseries_renamed/bochum-008_pumi-4.tsv",
"/Users/tspisak/res/PAINTeR/bochum/regional_timeseries_renamed/bochum-009_pumi-5.tsv",
"/Users/tspisak/res/PAINTeR/bochum/regional_timeseries_renamed/bochum-010_pumi-6.tsv",
"/Users/tspisak/res/PAINTeR/bochum/regional_timeseries_renamed/bochum-011_pumi-7.tsv",
"/Users/tspisak/res/PAINTeR/bochum/regional_timeseries_renamed/bochum-012_pumi-8.tsv",
"/Users/tspisak/res/PAINTeR/bochum/regional_timeseries_renamed/bochum-013_pumi-9.tsv",
"/Users/tspisak/res/PAINTeR/bochum/regional_timeseries_renamed/bochum-014_pumi-10_ex.tsv",
"/Users/tspisak/res/PAINTeR/bochum/regional_timeseries_renamed/bochum-015_pumi-11.tsv",
"/Users/tspisak/res/PAINTeR/bochum/regional_timeseries_renamed/bochum-016_pumi-12.tsv",
"/Users/tspisak/res/PAINTeR/bochum/regional_timeseries_renamed/bochum-017_pumi-13.tsv",
"/Users/tspisak/res/PAINTeR/bochum/regional_timeseries_renamed/bochum-018_pumi-14.tsv",
"/Users/tspisak/res/PAINTeR/bochum/regional_timeseries_renamed/bochum-019_pumi-15.tsv",
"/Users/tspisak/res/PAINTeR/bochum/regional_timeseries_renamed/bochum-020_pumi-16.tsv",
"/Users/tspisak/res/PAINTeR/bochum/regional_timeseries_renamed/bochum-023_pumi-17.tsv",
"/Users/tspisak/res/PAINTeR/bochum/regional_timeseries_renamed/bochum-024_pumi-18.tsv",
"/Users/tspisak/res/PAINTeR/bochum/regional_timeseries_renamed/bochum-025_pumi-19.tsv",
"/Users/tspisak/res/PAINTeR/bochum/regional_timeseries_renamed/bochum-026_pumi-20.tsv",
"/Users/tspisak/res/PAINTeR/bochum/regional_timeseries_renamed/bochum-027_pumi-21.tsv",
"/Users/tspisak/res/PAINTeR/bochum/regional_timeseries_renamed/bochum-028_pumi-22.tsv",
"/Users/tspisak/res/PAINTeR/bochum/regional_timeseries_renamed/bochum-029_pumi-23_ex.tsv",
"/Users/tspisak/res/PAINTeR/bochum/regional_timeseries_renamed/bochum-030_pumi-24.tsv",
"/Users/tspisak/res/PAINTeR/bochum/regional_timeseries_renamed/bochum-031_pumi-25.tsv",
"/Users/tspisak/res/PAINTeR/bochum/regional_timeseries_renamed/bochum-032_pumi-26.tsv",
"/Users/tspisak/res/PAINTeR/bochum/regional_timeseries_renamed/bochum-033_pumi-27.tsv",
"/Users/tspisak/res/PAINTeR/bochum/regional_timeseries_renamed/bochum-034_pumi-28_ex.tsv",
"/Users/tspisak/res/PAINTeR/bochum/regional_timeseries_renamed/bochum-035_pumi-29.tsv",
"/Users/tspisak/res/PAINTeR/bochum/regional_timeseries_renamed/bochum-036_pumi-30.tsv",
"/Users/tspisak/res/PAINTeR/bochum/regional_timeseries_renamed/bochum-037_pumi-31.tsv",
"/Users/tspisak/res/PAINTeR/bochum/regional_timeseries_renamed/bochum-038_pumi-32.tsv",
"/Users/tspisak/res/PAINTeR/bochum/regional_timeseries_renamed/bochum-039_pumi-33.tsv",
"/Users/tspisak/res/PAINTeR/bochum/regional_timeseries_renamed/bochum-040_pumi-34_ex.tsv",
"/Users/tspisak/res/PAINTeR/bochum/regional_timeseries_renamed/bochum-041_pumi-35.tsv",
"/Users/tspisak/res/PAINTeR/bochum/regional_timeseries_renamed/bochum-042_pumi-36.tsv",
"/Users/tspisak/res/PAINTeR/bochum/regional_timeseries_renamed/bochum-043_pumi-37.tsv",
"/Users/tspisak/res/PAINTeR/bochum/regional_timeseries_renamed/bochum-044_pumi-38_e.tsv",
"/Users/tspisak/res/PAINTeR/bochum/regional_timeseries_renamed/bochum-045_pumi-39.tsv",
"/Users/tspisak/res/PAINTeR/bochum/regional_timeseries_renamed/bochum-046_pumi-40.tsv"
]

bochum_fd_files = [
"/Users/tspisak/res/PAINTeR/bochum/FD_renamed/FD_bochum-004_pumi-0.txt",
"/Users/tspisak/res/PAINTeR/bochum/FD_renamed/FD_bochum-005_pumi-1.txt",
"/Users/tspisak/res/PAINTeR/bochum/FD_renamed/FD_bochum-006_pumi-2.txt",
"/Users/tspisak/res/PAINTeR/bochum/FD_renamed/FD_bochum-007_pumi-3.txt",
"/Users/tspisak/res/PAINTeR/bochum/FD_renamed/FD_bochum-008_pumi-4.txt",
"/Users/tspisak/res/PAINTeR/bochum/FD_renamed/FD_bochum-009_pumi-5.txt",
"/Users/tspisak/res/PAINTeR/bochum/FD_renamed/FD_bochum-010_pumi-6.txt",
"/Users/tspisak/res/PAINTeR/bochum/FD_renamed/FD_bochum-011_pumi-7.txt",
"/Users/tspisak/res/PAINTeR/bochum/FD_renamed/FD_bochum-012_pumi-8.txt",
"/Users/tspisak/res/PAINTeR/bochum/FD_renamed/FD_bochum-013_pumi-9.txt",
"/Users/tspisak/res/PAINTeR/bochum/FD_renamed/FD_bochum-014_pumi-10.txt",
"/Users/tspisak/res/PAINTeR/bochum/FD_renamed/FD_bochum-015_pumi-11.txt",
"/Users/tspisak/res/PAINTeR/bochum/FD_renamed/FD_bochum-016_pumi-12.txt",
"/Users/tspisak/res/PAINTeR/bochum/FD_renamed/FD_bochum-017_pumi-13.txt",
"/Users/tspisak/res/PAINTeR/bochum/FD_renamed/FD_bochum-018_pumi-14.txt",
"/Users/tspisak/res/PAINTeR/bochum/FD_renamed/FD_bochum-019_pumi-15.txt",
"/Users/tspisak/res/PAINTeR/bochum/FD_renamed/FD_bochum-020_pumi-16.txt",
"/Users/tspisak/res/PAINTeR/bochum/FD_renamed/FD_bochum-023_pumi-17.txt",
"/Users/tspisak/res/PAINTeR/bochum/FD_renamed/FD_bochum-024_pumi-18.txt",
"/Users/tspisak/res/PAINTeR/bochum/FD_renamed/FD_bochum-025_pumi-19.txt",
"/Users/tspisak/res/PAINTeR/bochum/FD_renamed/FD_bochum-026_pumi-20.txt",
"/Users/tspisak/res/PAINTeR/bochum/FD_renamed/FD_bochum-027_pumi-21.txt",
"/Users/tspisak/res/PAINTeR/bochum/FD_renamed/FD_bochum-028_pumi-22.txt",
"/Users/tspisak/res/PAINTeR/bochum/FD_renamed/FD_bochum-029_pumi-23.txt",
"/Users/tspisak/res/PAINTeR/bochum/FD_renamed/FD_bochum-030_pumi-24.txt",
"/Users/tspisak/res/PAINTeR/bochum/FD_renamed/FD_bochum-031_pumi-25.txt",
"/Users/tspisak/res/PAINTeR/bochum/FD_renamed/FD_bochum-032_pumi-26.txt",
"/Users/tspisak/res/PAINTeR/bochum/FD_renamed/FD_bochum-033_pumi-27.txt",
"/Users/tspisak/res/PAINTeR/bochum/FD_renamed/FD_bochum-034_pumi-28.txt",
"/Users/tspisak/res/PAINTeR/bochum/FD_renamed/FD_bochum-035_pumi-29.txt",
"/Users/tspisak/res/PAINTeR/bochum/FD_renamed/FD_bochum-036_pumi-30.txt",
"/Users/tspisak/res/PAINTeR/bochum/FD_renamed/FD_bochum-037_pumi-31.txt",
"/Users/tspisak/res/PAINTeR/bochum/FD_renamed/FD_bochum-038_pumi-32.txt",
"/Users/tspisak/res/PAINTeR/bochum/FD_renamed/FD_bochum-039_pumi-33.txt",
"/Users/tspisak/res/PAINTeR/bochum/FD_renamed/FD_bochum-040_pumi-34.txt",
"/Users/tspisak/res/PAINTeR/bochum/FD_renamed/FD_bochum-041_pumi-35.txt",
"/Users/tspisak/res/PAINTeR/bochum/FD_renamed/FD_bochum-042_pumi-36.txt",
"/Users/tspisak/res/PAINTeR/bochum/FD_renamed/FD_bochum-043_pumi-37.txt",
"/Users/tspisak/res/PAINTeR/bochum/FD_renamed/FD_bochum-044_pumi-38.txt",
"/Users/tspisak/res/PAINTeR/bochum/FD_renamed/FD_bochum-045_pumi-39.txt",
"/Users/tspisak/res/PAINTeR/bochum/FD_renamed/FD_bochum-046_pumi-40.txt"
]

bochum_table = "/Users/tspisak/data/PAINTeR/TABLES/bochum_sample.csv"

essen_ts_files = [
"/Users/tspisak/res/PAINTeR/essen/regional_timeseries_renamed_1mm/essen-001_pumi-44_pumi+045.tsv",
"/Users/tspisak/res/PAINTeR/essen/regional_timeseries_renamed_1mm/essen-002_pumi-43_pumi+044.tsv",
"/Users/tspisak/res/PAINTeR/essen/regional_timeseries_renamed_1mm/essen-003_pumi-42_pumi+043.tsv",
"/Users/tspisak/res/PAINTeR/essen/regional_timeseries_renamed_1mm/essen-004_pumi-41_pumi+042.tsv",
"/Users/tspisak/res/PAINTeR/essen/regional_timeseries_renamed_1mm/essen-005_pumi-18_pumi+019_ab1.tsv",
#"/Users/tspisak/res/PAINTeR/essen/regional_timeseries_renamed_1mm/essen-005_pumi-40_pumi+041_ab2.tsv",
"/Users/tspisak/res/PAINTeR/essen/regional_timeseries_renamed_1mm/essen-006_pumi-39_pumi+040.tsv",
"/Users/tspisak/res/PAINTeR/essen/regional_timeseries_renamed_1mm/essen-007_pumi-38_pumi+039.tsv",
"/Users/tspisak/res/PAINTeR/essen/regional_timeseries_renamed_1mm/essen-008_pumi-37_pumi+038.tsv",
"/Users/tspisak/res/PAINTeR/essen/regional_timeseries_renamed_1mm/essen-009_pumi-36_pumi+037.tsv",
"/Users/tspisak/res/PAINTeR/essen/regional_timeseries_renamed_1mm/essen-010_pumi-35_pumi+036.tsv",
"/Users/tspisak/res/PAINTeR/essen/regional_timeseries_renamed_1mm/essen-011_pumi-33_pumi+034.tsv",
"/Users/tspisak/res/PAINTeR/essen/regional_timeseries_renamed_1mm/essen-012_pumi-34_pumi+035_ex.tsv",
"/Users/tspisak/res/PAINTeR/essen/regional_timeseries_renamed_1mm/essen-013_pumi-32_pumi+033.tsv",
"/Users/tspisak/res/PAINTeR/essen/regional_timeseries_renamed_1mm/essen-014_pumi-31_pumi+032.tsv",
"/Users/tspisak/res/PAINTeR/essen/regional_timeseries_renamed_1mm/essen-015_pumi-30_pumi+031.tsv",
"/Users/tspisak/res/PAINTeR/essen/regional_timeseries_renamed_1mm/essen-016_pumi-28_pumi+029_ex.tsv",
"/Users/tspisak/res/PAINTeR/essen/regional_timeseries_renamed_1mm/essen-017_pumi-29_pumi+030.tsv",
"/Users/tspisak/res/PAINTeR/essen/regional_timeseries_renamed_1mm/essen-018_pumi-27_pumi+028.tsv",
"/Users/tspisak/res/PAINTeR/essen/regional_timeseries_renamed_1mm/essen-019_pumi-26_pumi+027_e.tsv",
"/Users/tspisak/res/PAINTeR/essen/regional_timeseries_renamed_1mm/essen-020_pumi-25_pumi+026.tsv",
"/Users/tspisak/res/PAINTeR/essen/regional_timeseries_renamed_1mm/essen-021_pumi-24_pumi+025_ex.tsv",
"/Users/tspisak/res/PAINTeR/essen/regional_timeseries_renamed_1mm/essen-022_pumi-23_pumi+024.tsv",
"/Users/tspisak/res/PAINTeR/essen/regional_timeseries_renamed_1mm/essen-023_pumi-14_pumi+015.tsv",
"/Users/tspisak/res/PAINTeR/essen/regional_timeseries_renamed_1mm/essen-024_pumi-22_pumi+023.tsv",
"/Users/tspisak/res/PAINTeR/essen/regional_timeseries_renamed_1mm/essen-025_pumi-21_pumi+022.tsv",
"/Users/tspisak/res/PAINTeR/essen/regional_timeseries_renamed_1mm/essen-026_pumi-20_pumi+021_e.tsv",
"/Users/tspisak/res/PAINTeR/essen/regional_timeseries_renamed_1mm/essen-027_pumi-19_pumi+020.tsv",
"/Users/tspisak/res/PAINTeR/essen/regional_timeseries_renamed_1mm/essen-028_pumi-17_pumi+018_ex.tsv",
"/Users/tspisak/res/PAINTeR/essen/regional_timeseries_renamed_1mm/essen-029_pumi-16_pumi+017.tsv",
"/Users/tspisak/res/PAINTeR/essen/regional_timeseries_renamed_1mm/essen-030_pumi-13_pumi+014.tsv",
"/Users/tspisak/res/PAINTeR/essen/regional_timeseries_renamed_1mm/essen-031_pumi-15_pumi+016.tsv",
"/Users/tspisak/res/PAINTeR/essen/regional_timeseries_renamed_1mm/essen-032_pumi-11_pumi+012.tsv",
"/Users/tspisak/res/PAINTeR/essen/regional_timeseries_renamed_1mm/essen-033_pumi-12_pumi+013_ex.tsv",
"/Users/tspisak/res/PAINTeR/essen/regional_timeseries_renamed_1mm/essen-034_pumi-10_pumi+011.tsv",
"/Users/tspisak/res/PAINTeR/essen/regional_timeseries_renamed_1mm/essen-035_pumi-9_pumi+010.tsv",
"/Users/tspisak/res/PAINTeR/essen/regional_timeseries_renamed_1mm/essen-036_pumi-8_pumi+009_e.tsv",
"/Users/tspisak/res/PAINTeR/essen/regional_timeseries_renamed_1mm/essen-037_pumi-7_pumi+008_e.tsv",
"/Users/tspisak/res/PAINTeR/essen/regional_timeseries_renamed_1mm/essen-038_pumi-6_pumi+007.tsv",
"/Users/tspisak/res/PAINTeR/essen/regional_timeseries_renamed_1mm/essen-039_pumi-5_pumi+006.tsv",
"/Users/tspisak/res/PAINTeR/essen/regional_timeseries_renamed_1mm/essen-040_pumi-2_pumi+003.tsv",
"/Users/tspisak/res/PAINTeR/essen/regional_timeseries_renamed_1mm/essen-041_pumi-1_pumi+002.tsv",
"/Users/tspisak/res/PAINTeR/essen/regional_timeseries_renamed_1mm/essen-042_pumi-4_pumi+005.tsv",
"/Users/tspisak/res/PAINTeR/essen/regional_timeseries_renamed_1mm/essen-043_pumi-0_pumi+001.tsv",
"/Users/tspisak/res/PAINTeR/essen/regional_timeseries_renamed_1mm/essen-044_pumi-3_pumi+004.tsv",
"/Users/tspisak/res/PAINTeR/essen/regional_timeseries_renamed_1mm/essen-045_pumi-l0_pumi+045.tsv",
"/Users/tspisak/res/PAINTeR/essen/regional_timeseries_renamed_1mm/essen-046_pumi-l1_pumi+046.tsv",
"/Users/tspisak/res/PAINTeR/essen/regional_timeseries_renamed_1mm/essen-047_pumi-l2_pumi+047_ex.tsv",
"/Users/tspisak/res/PAINTeR/essen/regional_timeseries_renamed_1mm/essen-048_pumi-l3_pumi+048_e.tsv",
"/Users/tspisak/res/PAINTeR/essen/regional_timeseries_renamed_1mm/essen-049_pumi-l4_pumi+049.tsv"
]

essen_fd_files = [
"/Users/tspisak/res/PAINTeR/essen/FD_renamed/FD_essen-001_pumi-44_pumi+045.txt",
"/Users/tspisak/res/PAINTeR/essen/FD_renamed/FD_essen-002_pumi-43_pumi+044.txt",
"/Users/tspisak/res/PAINTeR/essen/FD_renamed/FD_essen-003_pumi-42_pumi+043.txt",
"/Users/tspisak/res/PAINTeR/essen/FD_renamed/FD_essen-004_pumi-41_pumi+042.txt",
"/Users/tspisak/res/PAINTeR/essen/FD_renamed/FD_essen-005_pumi-18_pumi+019.txt",
#"/Users/tspisak/res/PAINTeR/essen/FD_renamed/FD_essen-005_pumi-40_pumi+041.txt",
"/Users/tspisak/res/PAINTeR/essen/FD_renamed/FD_essen-006_pumi-39_pumi+040.txt",
"/Users/tspisak/res/PAINTeR/essen/FD_renamed/FD_essen-007_pumi-38_pumi+039.txt",
"/Users/tspisak/res/PAINTeR/essen/FD_renamed/FD_essen-008_pumi-37_pumi+038.txt",
"/Users/tspisak/res/PAINTeR/essen/FD_renamed/FD_essen-009_pumi-36_pumi+037.txt",
"/Users/tspisak/res/PAINTeR/essen/FD_renamed/FD_essen-010_pumi-35_pumi+036.txt",
"/Users/tspisak/res/PAINTeR/essen/FD_renamed/FD_essen-011_pumi-33_pumi+034.txt",
"/Users/tspisak/res/PAINTeR/essen/FD_renamed/FD_essen-012_pumi-34_pumi+035.txt",
"/Users/tspisak/res/PAINTeR/essen/FD_renamed/FD_essen-013_pumi-32_pumi+033.txt",
"/Users/tspisak/res/PAINTeR/essen/FD_renamed/FD_essen-014_pumi-31_pumi+032.txt",
"/Users/tspisak/res/PAINTeR/essen/FD_renamed/FD_essen-015_pumi-30_pumi+031.txt",
"/Users/tspisak/res/PAINTeR/essen/FD_renamed/FD_essen-016_pumi-28_pumi+029.txt",
"/Users/tspisak/res/PAINTeR/essen/FD_renamed/FD_essen-017_pumi-29_pumi+030.txt",
"/Users/tspisak/res/PAINTeR/essen/FD_renamed/FD_essen-018_pumi-27_pumi+028.txt",
"/Users/tspisak/res/PAINTeR/essen/FD_renamed/FD_essen-019_pumi-26_pumi+027.txt",
"/Users/tspisak/res/PAINTeR/essen/FD_renamed/FD_essen-020_pumi-25_pumi+026.txt",
"/Users/tspisak/res/PAINTeR/essen/FD_renamed/FD_essen-021_pumi-24_pumi+025.txt",
"/Users/tspisak/res/PAINTeR/essen/FD_renamed/FD_essen-022_pumi-23_pumi+024.txt",
"/Users/tspisak/res/PAINTeR/essen/FD_renamed/FD_essen-023_pumi-14_pumi+015.txt",
"/Users/tspisak/res/PAINTeR/essen/FD_renamed/FD_essen-024_pumi-22_pumi+023.txt",
"/Users/tspisak/res/PAINTeR/essen/FD_renamed/FD_essen-025_pumi-21_pumi+022.txt",
"/Users/tspisak/res/PAINTeR/essen/FD_renamed/FD_essen-026_pumi-20_pumi+021.txt",
"/Users/tspisak/res/PAINTeR/essen/FD_renamed/FD_essen-027_pumi-19_pumi+020.txt",
"/Users/tspisak/res/PAINTeR/essen/FD_renamed/FD_essen-028_pumi-17_pumi+018.txt",
"/Users/tspisak/res/PAINTeR/essen/FD_renamed/FD_essen-029_pumi-16_pumi+017.txt",
"/Users/tspisak/res/PAINTeR/essen/FD_renamed/FD_essen-030_pumi-13_pumi+014.txt",
"/Users/tspisak/res/PAINTeR/essen/FD_renamed/FD_essen-031_pumi-15_pumi+016.txt",
"/Users/tspisak/res/PAINTeR/essen/FD_renamed/FD_essen-032_pumi-11_pumi+012.txt",
"/Users/tspisak/res/PAINTeR/essen/FD_renamed/FD_essen-033_pumi-12_pumi+013.txt",
"/Users/tspisak/res/PAINTeR/essen/FD_renamed/FD_essen-034_pumi-10_pumi+011.txt",
"/Users/tspisak/res/PAINTeR/essen/FD_renamed/FD_essen-035_pumi-9_pumi+010.txt",
"/Users/tspisak/res/PAINTeR/essen/FD_renamed/FD_essen-036_pumi-8_pumi+009.txt",
"/Users/tspisak/res/PAINTeR/essen/FD_renamed/FD_essen-037_pumi-7_pumi+008.txt",
"/Users/tspisak/res/PAINTeR/essen/FD_renamed/FD_essen-038_pumi-6_pumi+007.txt",
"/Users/tspisak/res/PAINTeR/essen/FD_renamed/FD_essen-039_pumi-5_pumi+006.txt",
"/Users/tspisak/res/PAINTeR/essen/FD_renamed/FD_essen-040_pumi-2_pumi+003.txt",
"/Users/tspisak/res/PAINTeR/essen/FD_renamed/FD_essen-041_pumi-1_pumi+002.txt",
"/Users/tspisak/res/PAINTeR/essen/FD_renamed/FD_essen-042_pumi-4_pumi+005.txt",
"/Users/tspisak/res/PAINTeR/essen/FD_renamed/FD_essen-043_pumi-0_pumi+001.txt",
"/Users/tspisak/res/PAINTeR/essen/FD_renamed/FD_essen-044_pumi-3_pumi+004.txt",
"/Users/tspisak/res/PAINTeR/essen/FD_renamed/FD_essen-045_pumi-l0_pumi+045.txt",
"/Users/tspisak/res/PAINTeR/essen/FD_renamed/FD_essen-046_pumi-l1_pumi+046.txt",
"/Users/tspisak/res/PAINTeR/essen/FD_renamed/FD_essen-047_pumi-l2_pumi+047.txt",
"/Users/tspisak/res/PAINTeR/essen/FD_renamed/FD_essen-048_pumi-l3_pumi+048.txt",
"/Users/tspisak/res/PAINTeR/essen/FD_renamed/FD_essen-049_pumi-l4_pumi+049.txt"
]

essen_table = "/Users/tspisak/data/PAINTeR/TABLES/essen_sample_partial.csv"

########################################################################################################################
# load fd
########################################################################################################################

bochum_fd = []
bochum_mean_FD = []
bochum_median_FD = []
bochum_max_FD = []
bochum_perc_scrub = []
for f in bochum_fd_files:
    fd = pd.read_csv(f, sep="\t").values
    bochum_fd.append(fd.ravel())
    bochum_mean_FD.append(fd.mean())
    bochum_median_FD.append(np.median(fd))
    bochum_max_FD.append(fd.max())
    bochum_perc_scrub.append(100*len(fd[fd>scrub_threshold])/len(fd))

essen_fd = []
essen_mean_FD = []
essen_median_FD = []
essen_max_FD = []
essen_perc_scrub = []
for f in essen_fd_files:
    fd = pd.read_csv(f, sep="\t").values
    essen_fd.append(fd.ravel())
    essen_mean_FD.append(fd.mean())
    essen_median_FD.append(np.median(fd))
    essen_max_FD.append(fd.max())
    essen_perc_scrub.append(100 * len(fd[fd > scrub_threshold]) / len(fd))

########################################################################################################################
# load QST and other data
########################################################################################################################

bochum_data = pd.read_csv(bochum_table, sep=",")
exclude_bochum = [1,2,3, 21, 22] # original excludes (pilots and missing data)
for id in exclude_bochum:
    bochum_data = bochum_data[bochum_data["ID"] != id]
bochum_data["meanFD"]=bochum_mean_FD
bochum_data["medianFD"]=bochum_median_FD
bochum_data["maxFD"]=bochum_max_FD
bochum_data["perc_scrubbed"]=bochum_perc_scrub
bochum_data["timesereries_file"]=bochum_ts_files
bochum_data["fd_file"]=bochum_fd_files

essen_data = pd.read_csv(essen_table, sep=",")

essen_data["meanFD"]=essen_mean_FD
essen_data["medianFD"]=essen_median_FD
essen_data["maxFD"]=essen_max_FD
essen_data["perc_scrubbed"]=essen_perc_scrub
essen_data["timesereries_file"]=essen_ts_files
essen_data["fd_file"]=essen_fd_files

########################################################################################################################
# Exclude
########################################################################################################################

bochum_data = bochum_data[bochum_data["meanFD"] < mean_FD_threshold]
essen_data = essen_data[essen_data["meanFD"] < mean_FD_threshold]

bochum_data = bochum_data[bochum_data["perc_scrubbed"] < perc_scrub_threshold]
essen_data = essen_data[essen_data["perc_scrubbed"] < perc_scrub_threshold]



# save current data frame
bochum_data.to_csv("bochum.csv")
essen_data.to_csv("essen.csv")


########################################################################################################################
# load timeseries
########################################################################################################################

bochum_timeseries = []
for i, f in enumerate(bochum_data["timesereries_file"].values.ravel()):
    ts = pd.read_csv(f, sep="\t").values


    if scrubbing:
        fd = pd.read_csv(bochum_data["fd_file"].values.ravel()[i]).values.ravel().tolist()
        fd = [0] + fd
        ts = scrub(ts, np.array(fd)) # ts[np.array(fd) < scrub_threshold, ]


    # standardise timeseries
    ts = StandardScaler().fit_transform(ts)
    bochum_timeseries.append(ts)

essen_timeseries = []
for i, f in enumerate(essen_data["timesereries_file"].values.ravel()):
    ts = pd.read_csv(f, sep="\t").values

    if scrubbing:
        fd = pd.read_csv(essen_data["fd_file"].values.ravel()[i]).values.ravel().tolist()
        fd = [0] + fd
        ts = scrub(ts, np.array(fd)) # ts[np.array(fd) < scrub_threshold, ]



    # standardise timeseries
    ts = StandardScaler().fit_transform(ts)
    essen_timeseries.append(ts)

########################################################################################################################
# conn matrix
########################################################################################################################

X_bochum, cm_bochum = load.compute_connectivity(bochum_timeseries, kind="partial correlation", discard_diagonal=True)
X_essen, cm_essen = load.compute_connectivity(essen_timeseries, kind="partial correlation", discard_diagonal=True)

#mat=cm_bochum.mean_
#mat[range(mat.shape[0]), range(mat.shape[0])] = 0
#mat[mat<-0.5]=-0.5
#mat[mat>0.5]=0.5
#plotting.plot_matrix(mat)
#plotting.show()

#mat=cm_essen.mean_
#mat[range(mat.shape[0]), range(mat.shape[0])] = 0
#mat[mat<-0.5]=-0.5
#mat[mat>0.5]=0.5
#plotting.plot_matrix(mat)
#plotting.show()


########################################################################################################################
# Train on Bochum data
########################################################################################################################

y_bochum=bochum_data["mean_QST_pain_sensitivity"].values.ravel()
#y_bochum=np.mean(bochum_data[["mean_QST_pain_sensitivityd2", "mean_QST_pain_sensitivity"]], axis=1)
# exclude NANs
X_bochum_ = X_bochum[~np.isnan(y_bochum), :]
y_bochum = y_bochum[~np.isnan(y_bochum)]


mymodel, p_grid = models.pipe_scale_fsel_model(scaler=preprocessing.MaxAbsScaler())
#p_grid = {'fsel__k': [10, 20, 30], 'model__alpha': [.000001, .00005], 'model__l1_ratio': [.999999999]}
#p_grid = {'fsel__k': [10, 20, 30, 50, 100, 150, 200], 'model__alpha': [.00001, .0001, .001, 0.005, .01], 'model__l1_ratio': [ .1, .3, .5, .7, .9]}
p_grid = {'fsel__k': [30], 'model__alpha': [.02], 'model__l1_ratio': [.999999999]}
p_grid = {'fsel__k': [30], 'model__alpha': [.02], 'model__l1_ratio': [.999999999]}
#p_grid = {'fsel__k': [50], 'model__alpha': [.01], 'model__l1_ratio': [.999999999]}

m, avg_model, all_models = train.train(X_bochum_, y_bochum, mymodel, p_grid, "Bochum", nested=True)

RES, mat, labels = train.get_full_coef(X_bochum_, m)
plotting.plot_matrix(vec_to_sym_matrix(avg_model, diagonal=np.repeat(0, len(labels)+1)), figure=(10, 10), labels=['GS']+labels['modules'].values.tolist(), title="", grid=True)  #['GS']+
plotting.show()

########################################################################################################################
# Try to predict the Essen sample with the trained model m
########################################################################################################################

y_essen = essen_data["painsensitivity.2.boch"].values.ravel() #liberal

# exclude NANs
essen_data_ = essen_data[~np.isnan(y_essen)]
X = X_essen[~np.isnan(y_essen), :]
y_essen = y_essen[~np.isnan(y_essen)]

print len(y_essen)

#pred_essen = m.predict(X)
pred_essen=train.bagged_predict(all_models, X)

#pred_essen=pred_essen-np.mean(pred_essen)#/np.std(pred_essen)
#y_essen=y_essen-np.mean(y_essen)#/np.std(y_essen)
#pred_essen=preprocessing.scale(pred_essen)
#y_essen=preprocessing.scale(y_essen)

print "LIBERAL Expl. Var.:" + str( 1- (-mean_squared_error(y_pred=pred_essen, y_true=y_essen))/-mean_squared_error(np.repeat(y_bochum.mean(), len(y_essen)), y_essen) )
print "Correlation: " + str(np.corrcoef(pred_essen, y_essen)[0,1])

from scipy import stats
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.scatter(y_essen, pred_essen, edgecolors=(0, 0, 0))
ax.plot([y_essen.min(), y_essen.max()],
                   [y_essen.min(), y_essen.max()],
                   'k--',
                   lw=2)

#regression part
slope, intercept, r_value, p_value, std_err = stats.linregress(y_essen,pred_essen)
print r_value
print p_value

#line = slope*y_essen+intercept
#plt.plot(y_essen, line, lw=2)

ax.set_xlabel('Pain Sensitivity (trained on Bochum sample, predicted on Essen sample)')
ax.set_ylabel('Predicted')

#for i, txt in enumerate(essen_data_["ID"]):
    #ax.annotate(str(txt), (y_essen[i], pred_essen[i]))
plt.title( "Expl. Var.:" + str( 1- (-mean_squared_error(y_pred=pred_essen, y_true=y_essen))/-mean_squared_error(np.repeat(y_bochum.mean(), len(y_essen)), y_essen) ) +
    "\nCorrelation: " + str(np.corrcoef(pred_essen, y_essen)[0,1]))
plt.show()
fig.savefig("prediction_1.pdf", bbox_inches='tight')

########################################################################################################################
# Try to predict the Essen sample with the trained model m
########################################################################################################################

y_essen = essen_data["painsensitivity.NA.boch"].values.ravel() #CPT conservative
# exclude NANs
essen_data_ = essen_data[~np.isnan(y_essen)]
X = X_essen[~np.isnan(y_essen), :]
y_essen = y_essen[~np.isnan(y_essen)]


#pred_essen = m.predict(X)
pred_essen=train.bagged_predict(all_models, X)

#pred_essen=pred_essen-np.mean(pred_essen)#/np.std(pred_essen)
#y_essen=y_essen-np.mean(y_essen)#/np.std(y_essen)
#pred_essen=preprocessing.scale(pred_essen)
#y_essen=preprocessing.scale(y_essen)

print "CONSERVATIVE Expl. Var.:" + str( 1- (-mean_squared_error(y_pred=pred_essen, y_true=y_essen))/-mean_squared_error(np.repeat(y_bochum.mean(), len(y_essen)), y_essen) )
print "MSE: " + str(-mean_squared_error(y_pred=pred_essen, y_true=y_essen))
print "Correlation: " + str(np.corrcoef(pred_essen, y_essen)[0,1])


import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.scatter(y_essen, pred_essen, edgecolors=(0, 0, 0))

#regression part
slope, intercept, r_value, p_value, std_err = stats.linregress(y_essen,pred_essen)
print r_value
print p_value

#line = slope*y_essen+intercept
#ax.plot(y_essen, line,'k--')

ax.plot([y_essen.min(), y_essen.max()],
                   [y_essen.min(), y_essen.max()],
                   'k--',
                   lw=2)
ax.set_xlabel('Pain Sensitivity (trained on Bochum sample, predicted on Essen sample)')
ax.set_ylabel('Predicted')

#for i, txt in enumerate(essen_data_["ID"]):
    #ax.annotate(str(txt), (y_essen[i], pred_essen[i]))
plt.title( "Expl. Var.:" + str( 1- (-mean_squared_error(y_pred=pred_essen, y_true=y_essen))/-mean_squared_error(y_pred=np.repeat(y_bochum.mean(), len(y_essen)), y_true=y_essen) ) +
    "\nCorrelation: " + str(np.corrcoef(pred_essen, y_essen)[0,1]))
plt.show()
fig.savefig("prediction_2.pdf", bbox_inches='tight')

########################################################################################################################
# Test on other variables (negative validators)
########################################################################################################################

VAR = "gender"
sex = essen_data[VAR].values.ravel() #CPT conservative
classnames, indices = np.unique(sex, return_inverse=True)
#print y
sex=indices
#print y
#print y
#y = np.log(y)
#print y
y = essen_data["painsensitivity.NA.boch"].values.ravel()
y = y[sex == 1]
X = X_essen
X = X[sex == 1, :]
# exclude NANs
X = X[~np.isnan(y), :]
y = y[~np.isnan(y)]



pred = m.predict(X)
pred=train.bagged_predict(all_models, X)

#pred_essen=pred_essen-np.mean(pred_essen)#/np.std(pred_essen)
#y_essen=y_essen-np.mean(y_essen)#/np.std(y_essen)
pred=preprocessing.scale(pred)
y=preprocessing.scale(y)

print "Sex Expl. Var.:" + str( 1- (-mean_squared_error(y_pred=pred, y_true=y))/-mean_squared_error(np.repeat(y.mean(), len(y)), y) )
print "Correlation: " + str(np.corrcoef(pred, y)[0,1])


import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.scatter(y, pred, edgecolors=(0, 0, 0))
#ax.plot([y.min(), y.max()],
#                   [y.min(), y.max()],
#                   'k--',
#                   lw=2)
ax.set_xlabel(VAR)
ax.set_ylabel('Predicted')

plt.title( "Expl. Var.:" + str( 1- (-mean_squared_error(y_pred=pred, y_true=y))/-mean_squared_error(np.repeat(y.mean(), len(y)), y) ) +
    "\nCorrelation: " + str(np.corrcoef(pred, y)[0,1]))
plt.show()

sums = np.sum(mat, axis=0)
labels = pd.read_csv("/Users/tspisak/res/PAINTeR/bochum/atlas_relabeled.tsv", sep="\t")
print np.argwhere(sums!=0).flatten()

print labels["labels"][np.argwhere(sums!=0).flatten()-1]
print pd.DataFrame( {'w': sums[sums!=0], 'label': labels["labels"][np.argwhere(sums!=0).flatten()-1]})

idx=np.transpose(np.nonzero(mat))
lab=np.array(['GlobSig'] + labels["labels"].values.tolist())
mod=np.array(['GlobSig'] + labels["modules"].values.tolist())

res=pd.DataFrame( {'idx_A': idx[:,0],
                   'reg_A': lab[np.array(idx[:,0])],
                   'mod_A': mod[np.array(idx[:,0])],
                   'idx_B': idx[:,1],
                   'reg_B': lab[np.array(idx[:,1])],
                   'mod_B': mod[np.array(idx[:,1])],
                   'weight': mat[np.nonzero(mat)].flatten()})

print res

res.to_csv("predictive_connections.csv")