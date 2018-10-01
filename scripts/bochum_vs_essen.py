#!/usr/bin/env python
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
import numpy as np


import PAINTeR.load_project_data as load
import PAINTeR.train as train
import PAINTeR.models as models
from sklearn.metrics import r2_score, explained_variance_score, mean_squared_error
from sklearn import preprocessing
import nilearn.plotting as plotting
from sklearn.feature_selection import SelectKBest, mutual_info_regression

###################################################################################################
# file lists
###################################################################################################

bochum_fmri=[
"/Volumes/Bingel_Mac/PAINTeR/std4D/pumi_dspk_cc_bochum_reg1antscpac/func2mni/_applywarp0func2mni_4_nuis_medang_bptf.nii.gz",
"/Volumes/Bingel_Mac/PAINTeR/std4D/pumi_dspk_cc_bochum_reg1antscpac/func2mni/_applywarp1func2mni_4_nuis_medang_bptf.nii.gz",
"/Volumes/Bingel_Mac/PAINTeR/std4D/pumi_dspk_cc_bochum_reg1antscpac/func2mni/_applywarp2func2mni_4_nuis_medang_bptf.nii.gz",
"/Volumes/Bingel_Mac/PAINTeR/std4D/pumi_dspk_cc_bochum_reg1antscpac/func2mni/_applywarp3func2mni_4_nuis_medang_bptf.nii.gz",
"/Volumes/Bingel_Mac/PAINTeR/std4D/pumi_dspk_cc_bochum_reg1antscpac/func2mni/_applywarp4func2mni_4_nuis_medang_bptf.nii.gz",
"/Volumes/Bingel_Mac/PAINTeR/std4D/pumi_dspk_cc_bochum_reg1antscpac/func2mni/_applywarp5func2mni_4_nuis_medang_bptf.nii.gz",
"/Volumes/Bingel_Mac/PAINTeR/std4D/pumi_dspk_cc_bochum_reg1antscpac/func2mni/_applywarp6func2mni_4_nuis_medang_bptf.nii.gz",
"/Volumes/Bingel_Mac/PAINTeR/std4D/pumi_dspk_cc_bochum_reg1antscpac/func2mni/_applywarp7func2mni_4_nuis_medang_bptf.nii.gz",
"/Volumes/Bingel_Mac/PAINTeR/std4D/pumi_dspk_cc_bochum_reg1antscpac/func2mni/_applywarp8func2mni_4_nuis_medang_bptf.nii.gz",
"/Volumes/Bingel_Mac/PAINTeR/std4D/pumi_dspk_cc_bochum_reg1antscpac/func2mni/_applywarp9func2mni_4_nuis_medang_bptf.nii.gz",
"/Volumes/Bingel_Mac/PAINTeR/std4D/pumi_dspk_cc_bochum_reg1antscpac/func2mni/_applywarp10func2mni_4_nuis_medang_bptf.nii.gz",
"/Volumes/Bingel_Mac/PAINTeR/std4D/pumi_dspk_cc_bochum_reg1antscpac/func2mni/_applywarp11func2mni_4_nuis_medang_bptf.nii.gz",
"/Volumes/Bingel_Mac/PAINTeR/std4D/pumi_dspk_cc_bochum_reg1antscpac/func2mni/_applywarp12func2mni_4_nuis_medang_bptf.nii.gz",
"/Volumes/Bingel_Mac/PAINTeR/std4D/pumi_dspk_cc_bochum_reg1antscpac/func2mni/_applywarp13func2mni_4_nuis_medang_bptf.nii.gz",
"/Volumes/Bingel_Mac/PAINTeR/std4D/pumi_dspk_cc_bochum_reg1antscpac/func2mni/_applywarp14func2mni_4_nuis_medang_bptf.nii.gz",
"/Volumes/Bingel_Mac/PAINTeR/std4D/pumi_dspk_cc_bochum_reg1antscpac/func2mni/_applywarp15func2mni_4_nuis_medang_bptf.nii.gz",
"/Volumes/Bingel_Mac/PAINTeR/std4D/pumi_dspk_cc_bochum_reg1antscpac/func2mni/_applywarp16func2mni_4_nuis_medang_bptf.nii.gz",
"/Volumes/Bingel_Mac/PAINTeR/std4D/pumi_dspk_cc_bochum_reg1antscpac/func2mni/_applywarp17func2mni_4_nuis_medang_bptf.nii.gz",
"/Volumes/Bingel_Mac/PAINTeR/std4D/pumi_dspk_cc_bochum_reg1antscpac/func2mni/_applywarp18func2mni_4_nuis_medang_bptf.nii.gz",
"/Volumes/Bingel_Mac/PAINTeR/std4D/pumi_dspk_cc_bochum_reg1antscpac/func2mni/_applywarp19func2mni_4_nuis_medang_bptf.nii.gz",
"/Volumes/Bingel_Mac/PAINTeR/std4D/pumi_dspk_cc_bochum_reg1antscpac/func2mni/_applywarp20func2mni_4_nuis_medang_bptf.nii.gz",
"/Volumes/Bingel_Mac/PAINTeR/std4D/pumi_dspk_cc_bochum_reg1antscpac/func2mni/_applywarp21func2mni_4_nuis_medang_bptf.nii.gz",
"/Volumes/Bingel_Mac/PAINTeR/std4D/pumi_dspk_cc_bochum_reg1antscpac/func2mni/_applywarp22func2mni_4_nuis_medang_bptf.nii.gz",
"/Volumes/Bingel_Mac/PAINTeR/std4D/pumi_dspk_cc_bochum_reg1antscpac/func2mni/_applywarp23func2mni_4_nuis_medang_bptf.nii.gz",
"/Volumes/Bingel_Mac/PAINTeR/std4D/pumi_dspk_cc_bochum_reg1antscpac/func2mni/_applywarp24func2mni_4_nuis_medang_bptf.nii.gz",
"/Volumes/Bingel_Mac/PAINTeR/std4D/pumi_dspk_cc_bochum_reg1antscpac/func2mni/_applywarp25func2mni_4_nuis_medang_bptf.nii.gz",
"/Volumes/Bingel_Mac/PAINTeR/std4D/pumi_dspk_cc_bochum_reg1antscpac/func2mni/_applywarp26func2mni_4_nuis_medang_bptf.nii.gz",
"/Volumes/Bingel_Mac/PAINTeR/std4D/pumi_dspk_cc_bochum_reg1antscpac/func2mni/_applywarp27func2mni_4_nuis_medang_bptf.nii.gz",
"/Volumes/Bingel_Mac/PAINTeR/std4D/pumi_dspk_cc_bochum_reg1antscpac/func2mni/_applywarp28func2mni_4_nuis_medang_bptf.nii.gz",
"/Volumes/Bingel_Mac/PAINTeR/std4D/pumi_dspk_cc_bochum_reg1antscpac/func2mni/_applywarp29func2mni_4_nuis_medang_bptf.nii.gz",
"/Volumes/Bingel_Mac/PAINTeR/std4D/pumi_dspk_cc_bochum_reg1antscpac/func2mni/_applywarp30func2mni_4_nuis_medang_bptf.nii.gz",
"/Volumes/Bingel_Mac/PAINTeR/std4D/pumi_dspk_cc_bochum_reg1antscpac/func2mni/_applywarp31func2mni_4_nuis_medang_bptf.nii.gz",
"/Volumes/Bingel_Mac/PAINTeR/std4D/pumi_dspk_cc_bochum_reg1antscpac/func2mni/_applywarp32func2mni_4_nuis_medang_bptf.nii.gz",
"/Volumes/Bingel_Mac/PAINTeR/std4D/pumi_dspk_cc_bochum_reg1antscpac/func2mni/_applywarp33func2mni_4_nuis_medang_bptf.nii.gz",
"/Volumes/Bingel_Mac/PAINTeR/std4D/pumi_dspk_cc_bochum_reg1antscpac/func2mni/_applywarp34func2mni_4_nuis_medang_bptf.nii.gz",
"/Volumes/Bingel_Mac/PAINTeR/std4D/pumi_dspk_cc_bochum_reg1antscpac/func2mni/_applywarp35func2mni_4_nuis_medang_bptf.nii.gz",
"/Volumes/Bingel_Mac/PAINTeR/std4D/pumi_dspk_cc_bochum_reg1antscpac/func2mni/_applywarp36func2mni_4_nuis_medang_bptf.nii.gz",
"/Volumes/Bingel_Mac/PAINTeR/std4D/pumi_dspk_cc_bochum_reg1antscpac/func2mni/_applywarp37func2mni_4_nuis_medang_bptf.nii.gz",
"/Volumes/Bingel_Mac/PAINTeR/std4D/pumi_dspk_cc_bochum_reg1antscpac/func2mni/_applywarp38func2mni_4_nuis_medang_bptf.nii.gz",
"/Volumes/Bingel_Mac/PAINTeR/std4D/pumi_dspk_cc_bochum_reg1antscpac/func2mni/_applywarp39func2mni_4_nuis_medang_bptf.nii.gz",
"/Volumes/Bingel_Mac/PAINTeR/std4D/pumi_dspk_cc_bochum_reg1antscpac/func2mni/_applywarp40func2mni_4_nuis_medang_bptf.nii.gz"
] # with cpacreg
bochum_fmri=[
"/Users/tspisak/res/PAINTeR/bochum/func_preproc/func2mni/_applywarp0func2mni_4_nuis_medang_bptf.nii.gz",
"/Users/tspisak/res/PAINTeR/bochum/func_preproc/func2mni/_applywarp1func2mni_4_nuis_medang_bptf.nii.gz",
"/Users/tspisak/res/PAINTeR/bochum/func_preproc/func2mni/_applywarp2func2mni_4_nuis_medang_bptf.nii.gz",
"/Users/tspisak/res/PAINTeR/bochum/func_preproc/func2mni/_applywarp3func2mni_4_nuis_medang_bptf.nii.gz",
"/Users/tspisak/res/PAINTeR/bochum/func_preproc/func2mni/_applywarp4func2mni_4_nuis_medang_bptf.nii.gz",
"/Users/tspisak/res/PAINTeR/bochum/func_preproc/func2mni/_applywarp5func2mni_4_nuis_medang_bptf.nii.gz",
"/Users/tspisak/res/PAINTeR/bochum/func_preproc/func2mni/_applywarp6func2mni_4_nuis_medang_bptf.nii.gz",
"/Users/tspisak/res/PAINTeR/bochum/func_preproc/func2mni/_applywarp7func2mni_4_nuis_medang_bptf.nii.gz",
"/Users/tspisak/res/PAINTeR/bochum/func_preproc/func2mni/_applywarp8func2mni_4_nuis_medang_bptf.nii.gz",
"/Users/tspisak/res/PAINTeR/bochum/func_preproc/func2mni/_applywarp9func2mni_4_nuis_medang_bptf.nii.gz",
"/Users/tspisak/res/PAINTeR/bochum/func_preproc/func2mni/_applywarp10func2mni_4_nuis_medang_bptf.nii.gz",
"/Users/tspisak/res/PAINTeR/bochum/func_preproc/func2mni/_applywarp11func2mni_4_nuis_medang_bptf.nii.gz",
"/Users/tspisak/res/PAINTeR/bochum/func_preproc/func2mni/_applywarp12func2mni_4_nuis_medang_bptf.nii.gz",
"/Users/tspisak/res/PAINTeR/bochum/func_preproc/func2mni/_applywarp13func2mni_4_nuis_medang_bptf.nii.gz",
"/Users/tspisak/res/PAINTeR/bochum/func_preproc/func2mni/_applywarp14func2mni_4_nuis_medang_bptf.nii.gz",
"/Users/tspisak/res/PAINTeR/bochum/func_preproc/func2mni/_applywarp15func2mni_4_nuis_medang_bptf.nii.gz",
"/Users/tspisak/res/PAINTeR/bochum/func_preproc/func2mni/_applywarp16func2mni_4_nuis_medang_bptf.nii.gz",
"/Users/tspisak/res/PAINTeR/bochum/func_preproc/func2mni/_applywarp17func2mni_4_nuis_medang_bptf.nii.gz",
"/Users/tspisak/res/PAINTeR/bochum/func_preproc/func2mni/_applywarp18func2mni_4_nuis_medang_bptf.nii.gz",
"/Users/tspisak/res/PAINTeR/bochum/func_preproc/func2mni/_applywarp19func2mni_4_nuis_medang_bptf.nii.gz",
"/Users/tspisak/res/PAINTeR/bochum/func_preproc/func2mni/_applywarp20func2mni_4_nuis_medang_bptf.nii.gz",
"/Users/tspisak/res/PAINTeR/bochum/func_preproc/func2mni/_applywarp21func2mni_4_nuis_medang_bptf.nii.gz",
"/Users/tspisak/res/PAINTeR/bochum/func_preproc/func2mni/_applywarp22func2mni_4_nuis_medang_bptf.nii.gz",
"/Users/tspisak/res/PAINTeR/bochum/func_preproc/func2mni/_applywarp23func2mni_4_nuis_medang_bptf.nii.gz",
"/Users/tspisak/res/PAINTeR/bochum/func_preproc/func2mni/_applywarp24func2mni_4_nuis_medang_bptf.nii.gz",
"/Users/tspisak/res/PAINTeR/bochum/func_preproc/func2mni/_applywarp25func2mni_4_nuis_medang_bptf.nii.gz",
"/Users/tspisak/res/PAINTeR/bochum/func_preproc/func2mni/_applywarp26func2mni_4_nuis_medang_bptf.nii.gz",
"/Users/tspisak/res/PAINTeR/bochum/func_preproc/func2mni/_applywarp27func2mni_4_nuis_medang_bptf.nii.gz",
"/Users/tspisak/res/PAINTeR/bochum/func_preproc/func2mni/_applywarp28func2mni_4_nuis_medang_bptf.nii.gz",
"/Users/tspisak/res/PAINTeR/bochum/func_preproc/func2mni/_applywarp29func2mni_4_nuis_medang_bptf.nii.gz",
"/Users/tspisak/res/PAINTeR/bochum/func_preproc/func2mni/_applywarp30func2mni_4_nuis_medang_bptf.nii.gz",
"/Users/tspisak/res/PAINTeR/bochum/func_preproc/func2mni/_applywarp31func2mni_4_nuis_medang_bptf.nii.gz",
"/Users/tspisak/res/PAINTeR/bochum/func_preproc/func2mni/_applywarp32func2mni_4_nuis_medang_bptf.nii.gz",
"/Users/tspisak/res/PAINTeR/bochum/func_preproc/func2mni/_applywarp33func2mni_4_nuis_medang_bptf.nii.gz",
"/Users/tspisak/res/PAINTeR/bochum/func_preproc/func2mni/_applywarp34func2mni_4_nuis_medang_bptf.nii.gz",
"/Users/tspisak/res/PAINTeR/bochum/func_preproc/func2mni/_applywarp35func2mni_4_nuis_medang_bptf.nii.gz",
"/Users/tspisak/res/PAINTeR/bochum/func_preproc/func2mni/_applywarp36func2mni_4_nuis_medang_bptf.nii.gz",
"/Users/tspisak/res/PAINTeR/bochum/func_preproc/func2mni/_applywarp37func2mni_4_nuis_medang_bptf.nii.gz",
"/Users/tspisak/res/PAINTeR/bochum/func_preproc/func2mni/_applywarp38func2mni_4_nuis_medang_bptf.nii.gz",
"/Users/tspisak/res/PAINTeR/bochum/func_preproc/func2mni/_applywarp39func2mni_4_nuis_medang_bptf.nii.gz",
"/Users/tspisak/res/PAINTeR/bochum/func_preproc/func2mni/_applywarp40func2mni_4_nuis_medang_bptf.nii.gz"
] # with own antsreg and own1 pipeline

essen_fmri=[
"/Volumes/Bingel_Mac/PAINTeR/std4D/pumi_dspk_cc_essen_reg1ants/func2mni/_applywarp0func2mni_4_nuis_medang_bptf.nii.gz",
"/Volumes/Bingel_Mac/PAINTeR/std4D/pumi_dspk_cc_essen_reg1ants/func2mni/_applywarp1func2mni_4_nuis_medang_bptf.nii.gz",
"/Volumes/Bingel_Mac/PAINTeR/std4D/pumi_dspk_cc_essen_reg1ants/func2mni/_applywarp2func2mni_4_nuis_medang_bptf.nii.gz",
"/Volumes/Bingel_Mac/PAINTeR/std4D/pumi_dspk_cc_essen_reg1ants/func2mni/_applywarp3func2mni_4_nuis_medang_bptf.nii.gz",
"/Volumes/Bingel_Mac/PAINTeR/std4D/pumi_dspk_cc_essen_reg1ants/func2mni/_applywarp4func2mni_4_nuis_medang_bptf.nii.gz",
"/Volumes/Bingel_Mac/PAINTeR/std4D/pumi_dspk_cc_essen_reg1ants/func2mni/_applywarp5func2mni_4_nuis_medang_bptf.nii.gz",
"/Volumes/Bingel_Mac/PAINTeR/std4D/pumi_dspk_cc_essen_reg1ants/func2mni/_applywarp6func2mni_4_nuis_medang_bptf.nii.gz",
"/Volumes/Bingel_Mac/PAINTeR/std4D/pumi_dspk_cc_essen_reg1ants/func2mni/_applywarp7func2mni_4_nuis_medang_bptf.nii.gz",
"/Volumes/Bingel_Mac/PAINTeR/std4D/pumi_dspk_cc_essen_reg1ants/func2mni/_applywarp8func2mni_4_nuis_medang_bptf.nii.gz",
"/Volumes/Bingel_Mac/PAINTeR/std4D/pumi_dspk_cc_essen_reg1ants/func2mni/_applywarp9func2mni_4_nuis_medang_bptf.nii.gz",
"/Volumes/Bingel_Mac/PAINTeR/std4D/pumi_dspk_cc_essen_reg1ants/func2mni/_applywarp10func2mni_4_nuis_medang_bptf.nii.gz",
"/Volumes/Bingel_Mac/PAINTeR/std4D/pumi_dspk_cc_essen_reg1ants/func2mni/_applywarp11func2mni_4_nuis_medang_bptf.nii.gz",
"/Volumes/Bingel_Mac/PAINTeR/std4D/pumi_dspk_cc_essen_reg1ants/func2mni/_applywarp12func2mni_4_nuis_medang_bptf.nii.gz",
"/Volumes/Bingel_Mac/PAINTeR/std4D/pumi_dspk_cc_essen_reg1ants/func2mni/_applywarp13func2mni_4_nuis_medang_bptf.nii.gz",
"/Volumes/Bingel_Mac/PAINTeR/std4D/pumi_dspk_cc_essen_reg1ants/func2mni/_applywarp14func2mni_4_nuis_medang_bptf.nii.gz",
"/Volumes/Bingel_Mac/PAINTeR/std4D/pumi_dspk_cc_essen_reg1ants/func2mni/_applywarp15func2mni_4_nuis_medang_bptf.nii.gz",
"/Volumes/Bingel_Mac/PAINTeR/std4D/pumi_dspk_cc_essen_reg1ants/func2mni/_applywarp16func2mni_4_nuis_medang_bptf.nii.gz",
"/Volumes/Bingel_Mac/PAINTeR/std4D/pumi_dspk_cc_essen_reg1ants/func2mni/_applywarp17func2mni_4_nuis_medang_bptf.nii.gz",
"/Volumes/Bingel_Mac/PAINTeR/std4D/pumi_dspk_cc_essen_reg1ants/func2mni/_applywarp18func2mni_4_nuis_medang_bptf.nii.gz",
"/Volumes/Bingel_Mac/PAINTeR/std4D/pumi_dspk_cc_essen_reg1ants/func2mni/_applywarp19func2mni_4_nuis_medang_bptf.nii.gz",
"/Volumes/Bingel_Mac/PAINTeR/std4D/pumi_dspk_cc_essen_reg1ants/func2mni/_applywarp20func2mni_4_nuis_medang_bptf.nii.gz",
"/Volumes/Bingel_Mac/PAINTeR/std4D/pumi_dspk_cc_essen_reg1ants/func2mni/_applywarp21func2mni_4_nuis_medang_bptf.nii.gz",
"/Volumes/Bingel_Mac/PAINTeR/std4D/pumi_dspk_cc_essen_reg1ants/func2mni/_applywarp22func2mni_4_nuis_medang_bptf.nii.gz",
"/Volumes/Bingel_Mac/PAINTeR/std4D/pumi_dspk_cc_essen_reg1ants/func2mni/_applywarp23func2mni_4_nuis_medang_bptf.nii.gz",
"/Volumes/Bingel_Mac/PAINTeR/std4D/pumi_dspk_cc_essen_reg1ants/func2mni/_applywarp24func2mni_4_nuis_medang_bptf.nii.gz",
"/Volumes/Bingel_Mac/PAINTeR/std4D/pumi_dspk_cc_essen_reg1ants/func2mni/_applywarp25func2mni_4_nuis_medang_bptf.nii.gz",
"/Volumes/Bingel_Mac/PAINTeR/std4D/pumi_dspk_cc_essen_reg1ants/func2mni/_applywarp26func2mni_4_nuis_medang_bptf.nii.gz",
"/Volumes/Bingel_Mac/PAINTeR/std4D/pumi_dspk_cc_essen_reg1ants/func2mni/_applywarp27func2mni_4_nuis_medang_bptf.nii.gz",
"/Volumes/Bingel_Mac/PAINTeR/std4D/pumi_dspk_cc_essen_reg1ants/func2mni/_applywarp28func2mni_4_nuis_medang_bptf.nii.gz",
"/Volumes/Bingel_Mac/PAINTeR/std4D/pumi_dspk_cc_essen_reg1ants/func2mni/_applywarp29func2mni_4_nuis_medang_bptf.nii.gz",
"/Volumes/Bingel_Mac/PAINTeR/std4D/pumi_dspk_cc_essen_reg1ants/func2mni/_applywarp30func2mni_4_nuis_medang_bptf.nii.gz",
"/Volumes/Bingel_Mac/PAINTeR/std4D/pumi_dspk_cc_essen_reg1ants/func2mni/_applywarp31func2mni_4_nuis_medang_bptf.nii.gz",
"/Volumes/Bingel_Mac/PAINTeR/std4D/pumi_dspk_cc_essen_reg1ants/func2mni/_applywarp32func2mni_4_nuis_medang_bptf.nii.gz",
"/Volumes/Bingel_Mac/PAINTeR/std4D/pumi_dspk_cc_essen_reg1ants/func2mni/_applywarp33func2mni_4_nuis_medang_bptf.nii.gz",
"/Volumes/Bingel_Mac/PAINTeR/std4D/pumi_dspk_cc_essen_reg1ants/func2mni/_applywarp34func2mni_4_nuis_medang_bptf.nii.gz",
"/Volumes/Bingel_Mac/PAINTeR/std4D/pumi_dspk_cc_essen_reg1ants/func2mni/_applywarp35func2mni_4_nuis_medang_bptf.nii.gz",
"/Volumes/Bingel_Mac/PAINTeR/std4D/pumi_dspk_cc_essen_reg1ants/func2mni/_applywarp36func2mni_4_nuis_medang_bptf.nii.gz",
"/Volumes/Bingel_Mac/PAINTeR/std4D/pumi_dspk_cc_essen_reg1ants/func2mni/_applywarp37func2mni_4_nuis_medang_bptf.nii.gz",
"/Volumes/Bingel_Mac/PAINTeR/std4D/pumi_dspk_cc_essen_reg1ants/func2mni/_applywarp38func2mni_4_nuis_medang_bptf.nii.gz",
"/Volumes/Bingel_Mac/PAINTeR/std4D/pumi_dspk_cc_essen_reg1ants/func2mni/_applywarp39func2mni_4_nuis_medang_bptf.nii.gz",
"/Volumes/Bingel_Mac/PAINTeR/std4D/pumi_dspk_cc_essen_reg1ants/func2mni/_applywarp40func2mni_4_nuis_medang_bptf.nii.gz",
"/Volumes/Bingel_Mac/PAINTeR/std4D/pumi_dspk_cc_essen_reg1ants/func2mni/_applywarp41func2mni_4_nuis_medang_bptf.nii.gz",
"/Volumes/Bingel_Mac/PAINTeR/std4D/pumi_dspk_cc_essen_reg1ants/func2mni/_applywarp42func2mni_4_nuis_medang_bptf.nii.gz",
"/Volumes/Bingel_Mac/PAINTeR/std4D/pumi_dspk_cc_essen_reg1ants/func2mni/_applywarp43func2mni_4_nuis_medang_bptf.nii.gz",
"/Volumes/Bingel_Mac/PAINTeR/std4D/pumi_dspk_cc_essen_reg1ants/func2mni/_applywarp44func2mni_4_nuis_medang_bptf.nii.gz"
] # with cpacreg
essen_fmri=[
"/Volumes/Bingel_Mac/PAINTeR/std4D/pumi_own1_essen_reg1antsown/func2mni/_applywarp0func2mni_4_nuis_medang_bptf.nii.gz",
"/Volumes/Bingel_Mac/PAINTeR/std4D/pumi_own1_essen_reg1antsown/func2mni/_applywarp1func2mni_4_nuis_medang_bptf.nii.gz",
"/Volumes/Bingel_Mac/PAINTeR/std4D/pumi_own1_essen_reg1antsown/func2mni/_applywarp2func2mni_4_nuis_medang_bptf.nii.gz",
"/Volumes/Bingel_Mac/PAINTeR/std4D/pumi_own1_essen_reg1antsown/func2mni/_applywarp3func2mni_4_nuis_medang_bptf.nii.gz",
"/Volumes/Bingel_Mac/PAINTeR/std4D/pumi_own1_essen_reg1antsown/func2mni/_applywarp4func2mni_4_nuis_medang_bptf.nii.gz",
"/Volumes/Bingel_Mac/PAINTeR/std4D/pumi_own1_essen_reg1antsown/func2mni/_applywarp5func2mni_4_nuis_medang_bptf.nii.gz",
"/Volumes/Bingel_Mac/PAINTeR/std4D/pumi_own1_essen_reg1antsown/func2mni/_applywarp6func2mni_4_nuis_medang_bptf.nii.gz",
"/Volumes/Bingel_Mac/PAINTeR/std4D/pumi_own1_essen_reg1antsown/func2mni/_applywarp7func2mni_4_nuis_medang_bptf.nii.gz",
"/Volumes/Bingel_Mac/PAINTeR/std4D/pumi_own1_essen_reg1antsown/func2mni/_applywarp8func2mni_4_nuis_medang_bptf.nii.gz",
"/Volumes/Bingel_Mac/PAINTeR/std4D/pumi_own1_essen_reg1antsown/func2mni/_applywarp9func2mni_4_nuis_medang_bptf.nii.gz",
"/Volumes/Bingel_Mac/PAINTeR/std4D/pumi_own1_essen_reg1antsown/func2mni/_applywarp10func2mni_4_nuis_medang_bptf.nii.gz",
"/Volumes/Bingel_Mac/PAINTeR/std4D/pumi_own1_essen_reg1antsown/func2mni/_applywarp11func2mni_4_nuis_medang_bptf.nii.gz",
"/Volumes/Bingel_Mac/PAINTeR/std4D/pumi_own1_essen_reg1antsown/func2mni/_applywarp12func2mni_4_nuis_medang_bptf.nii.gz",
"/Volumes/Bingel_Mac/PAINTeR/std4D/pumi_own1_essen_reg1antsown/func2mni/_applywarp13func2mni_4_nuis_medang_bptf.nii.gz",
"/Volumes/Bingel_Mac/PAINTeR/std4D/pumi_own1_essen_reg1antsown/func2mni/_applywarp14func2mni_4_nuis_medang_bptf.nii.gz",
"/Volumes/Bingel_Mac/PAINTeR/std4D/pumi_own1_essen_reg1antsown/func2mni/_applywarp15func2mni_4_nuis_medang_bptf.nii.gz",
"/Volumes/Bingel_Mac/PAINTeR/std4D/pumi_own1_essen_reg1antsown/func2mni/_applywarp16func2mni_4_nuis_medang_bptf.nii.gz",
"/Volumes/Bingel_Mac/PAINTeR/std4D/pumi_own1_essen_reg1antsown/func2mni/_applywarp17func2mni_4_nuis_medang_bptf.nii.gz",
"/Volumes/Bingel_Mac/PAINTeR/std4D/pumi_own1_essen_reg1antsown/func2mni/_applywarp18func2mni_4_nuis_medang_bptf.nii.gz",
"/Volumes/Bingel_Mac/PAINTeR/std4D/pumi_own1_essen_reg1antsown/func2mni/_applywarp19func2mni_4_nuis_medang_bptf.nii.gz",
"/Volumes/Bingel_Mac/PAINTeR/std4D/pumi_own1_essen_reg1antsown/func2mni/_applywarp20func2mni_4_nuis_medang_bptf.nii.gz",
"/Volumes/Bingel_Mac/PAINTeR/std4D/pumi_own1_essen_reg1antsown/func2mni/_applywarp21func2mni_4_nuis_medang_bptf.nii.gz",
"/Volumes/Bingel_Mac/PAINTeR/std4D/pumi_own1_essen_reg1antsown/func2mni/_applywarp22func2mni_4_nuis_medang_bptf.nii.gz",
"/Volumes/Bingel_Mac/PAINTeR/std4D/pumi_own1_essen_reg1antsown/func2mni/_applywarp23func2mni_4_nuis_medang_bptf.nii.gz",
"/Volumes/Bingel_Mac/PAINTeR/std4D/pumi_own1_essen_reg1antsown/func2mni/_applywarp24func2mni_4_nuis_medang_bptf.nii.gz",
"/Volumes/Bingel_Mac/PAINTeR/std4D/pumi_own1_essen_reg1antsown/func2mni/_applywarp25func2mni_4_nuis_medang_bptf.nii.gz",
"/Volumes/Bingel_Mac/PAINTeR/std4D/pumi_own1_essen_reg1antsown/func2mni/_applywarp26func2mni_4_nuis_medang_bptf.nii.gz",
"/Volumes/Bingel_Mac/PAINTeR/std4D/pumi_own1_essen_reg1antsown/func2mni/_applywarp27func2mni_4_nuis_medang_bptf.nii.gz",
"/Volumes/Bingel_Mac/PAINTeR/std4D/pumi_own1_essen_reg1antsown/func2mni/_applywarp28func2mni_4_nuis_medang_bptf.nii.gz",
"/Volumes/Bingel_Mac/PAINTeR/std4D/pumi_own1_essen_reg1antsown/func2mni/_applywarp29func2mni_4_nuis_medang_bptf.nii.gz",
"/Volumes/Bingel_Mac/PAINTeR/std4D/pumi_own1_essen_reg1antsown/func2mni/_applywarp30func2mni_4_nuis_medang_bptf.nii.gz",
"/Volumes/Bingel_Mac/PAINTeR/std4D/pumi_own1_essen_reg1antsown/func2mni/_applywarp31func2mni_4_nuis_medang_bptf.nii.gz",
"/Volumes/Bingel_Mac/PAINTeR/std4D/pumi_own1_essen_reg1antsown/func2mni/_applywarp32func2mni_4_nuis_medang_bptf.nii.gz",
"/Volumes/Bingel_Mac/PAINTeR/std4D/pumi_own1_essen_reg1antsown/func2mni/_applywarp33func2mni_4_nuis_medang_bptf.nii.gz",
"/Volumes/Bingel_Mac/PAINTeR/std4D/pumi_own1_essen_reg1antsown/func2mni/_applywarp34func2mni_4_nuis_medang_bptf.nii.gz",
"/Volumes/Bingel_Mac/PAINTeR/std4D/pumi_own1_essen_reg1antsown/func2mni/_applywarp35func2mni_4_nuis_medang_bptf.nii.gz",
"/Volumes/Bingel_Mac/PAINTeR/std4D/pumi_own1_essen_reg1antsown/func2mni/_applywarp36func2mni_4_nuis_medang_bptf.nii.gz",
"/Volumes/Bingel_Mac/PAINTeR/std4D/pumi_own1_essen_reg1antsown/func2mni/_applywarp37func2mni_4_nuis_medang_bptf.nii.gz",
"/Volumes/Bingel_Mac/PAINTeR/std4D/pumi_own1_essen_reg1antsown/func2mni/_applywarp38func2mni_4_nuis_medang_bptf.nii.gz",
"/Volumes/Bingel_Mac/PAINTeR/std4D/pumi_own1_essen_reg1antsown/func2mni/_applywarp39func2mni_4_nuis_medang_bptf.nii.gz",
"/Volumes/Bingel_Mac/PAINTeR/std4D/pumi_own1_essen_reg1antsown/func2mni/_applywarp40func2mni_4_nuis_medang_bptf.nii.gz",
"/Volumes/Bingel_Mac/PAINTeR/std4D/pumi_own1_essen_reg1antsown/func2mni/_applywarp41func2mni_4_nuis_medang_bptf.nii.gz",
"/Volumes/Bingel_Mac/PAINTeR/std4D/pumi_own1_essen_reg1antsown/func2mni/_applywarp42func2mni_4_nuis_medang_bptf.nii.gz",
"/Volumes/Bingel_Mac/PAINTeR/std4D/pumi_own1_essen_reg1antsown/func2mni/_applywarp43func2mni_4_nuis_medang_bptf.nii.gz",
"/Volumes/Bingel_Mac/PAINTeR/std4D/pumi_own1_essen_reg1antsown/func2mni/_applywarp44func2mni_4_nuis_medang_bptf.nii.gz"
] # with own antsreg and own1 pipeline

########################################################################
# LOAD ATLAS
########################################################################

mist_64_atlas_filename = '/Users/tspisak/data/atlases/MIST/Parcellations/MIST_122.nii.gz'
mist_64_labels_filename = '~/data/atlases/MIST/Parcel_Information/MIST_122.csv'

import pandas as pd

mist_hierarchy_filename = '/Users/tspisak/data/atlases/MIST/Hierarchy/MIST_PARCEL_ORDER_ROI.csv'
mist_hierarchy = pd.read_csv(mist_hierarchy_filename, sep=",")

mist_64_labels = pd.read_csv(mist_64_labels_filename, sep=";")
mist_64_hierarchy = mist_hierarchy['s122']
mist_64_hierarchy = mist_64_hierarchy.drop_duplicates()

###################################################################################################
# generate timeseries
###################################################################################################
generate_timeseries = False
standardise = True
if generate_timeseries:
    from nilearn.input_data import NiftiLabelsMasker

    masker = NiftiLabelsMasker(labels_img=mist_64_atlas_filename, standardize=True,
                               memory='nilearn_cache', background_label=0)

    mist_64_labels = mist_64_labels.reindex(mist_64_hierarchy - 1)


    print "Bochum..."
    pooled_subjects = []
    for subj in bochum_fmri:
        time_series = masker.fit_transform(subj)
        if standardise:
            from sklearn.preprocessing import StandardScaler
            time_series = StandardScaler().fit_transform(time_series)
        time_series = time_series[:, mist_64_hierarchy - 1]
        pooled_subjects.append(time_series)

    from sklearn.externals import joblib

    joblib.dump(pooled_subjects, "timeseries_122_bochum_pumi_own1_std.sav")

    print "Essen..."
    pooled_subjects = []
    for subj in essen_fmri:
        time_series = masker.fit_transform(subj)
        if standardise:
            from sklearn.preprocessing import StandardScaler
            time_series = StandardScaler().fit_transform(time_series)
        time_series = time_series[:, mist_64_hierarchy - 1]
        pooled_subjects.append(time_series)

    from sklearn.externals import joblib

    joblib.dump(pooled_subjects, "timeseries_122_essen_pumi_own1_std.sav")


ts_bochum = load.load_timeseries_sav("timeseries_122_bochum_pumi_own1_std.sav")
ts_essen = load.load_timeseries_sav("timeseries_122_essen_pumi_own1_std.sav")

###################################################################################################
# create matrix
###################################################################################################
X_bochum, cm_bochum = load.compute_connectivity(ts_bochum, kind="tangent")
X_essen, cm_essen = load.compute_connectivity(ts_essen, kind="tangent")

### X union #####################
#print len(ts_bochum)
#print len(ts_essen)
#
######
X_all, cm_all = load.compute_connectivity(ts_bochum + ts_essen, kind="tangent")

mat = cm_bochum.mean_
mat[range(mat.shape[0]), range(mat.shape[0])] = 0
#mat[mat<-0.5]=-0.5
#mat[mat>0.5]=0.5
plotting.plot_matrix(mat)
plotting.show()

#
X_bochum = X_all[:len(ts_bochum)]
#print len(X_bochum)
X_essen = X_all[len(ts_bochum):] #len(ts_bochum)#X_all[-len(ts_essen):]
#print len(X_essen)
######


### end X union #####################

# square
#X_bochum = np.hstack((X_bochum, np.square(X_bochum, X_bochum ** 2)))
#X_essen = np.hstack((X_essen, np.square(X_essen, X_essen ** 2)))

# dyn
#Xd_bochum = load.compute_dynconn(ts_bochum)
#Xd_essen = load.compute_dynconn(ts_essen)

#X_bochum = np.array(Xd_bochum) #np.hstack((X_bochum, Xd_bochum))
#X_essen = np.array(Xd_essen) #np.hstack((X_essen, Xd_essen))

print X_bochum.shape
print X_essen.shape

###################################################################################################
# load QST data
###################################################################################################
y_bochum_old = load.load_QST_data(data_filename="/Volumes/Bingel_Mac/PAINTeR/std4D/bochum_painthr.csv",
                             target_var= "mean_QST_pain_sensitivity",
                              exclude_ids=[1,2,3, 21, 22]) #40, 8???

y_essen_old = load.load_QST_data(data_filename="/Volumes/Bingel_Mac/PAINTeR/std4D/essen_painthr_fixed.csv",
                              idvar="subject.ID",
                              target_var= "compositepainsensitivity",
                           exclude_ids=["PAINTeR_46", "PAINTeR_47", "PAINTeR_48", "PAINTeR_49"])

y_all = load.load_QST_data_all(data_filename="/Volumes/Bingel_Mac/PAINTeR/std4D/bochum+essen_painthr.csv",
                              exclude_ids=["1","2","3", "21", "22", "PAINTeR_46", "PAINTeR_47", "PAINTeR_48", "PAINTeR_49"]) #40, 8???

y_bochum = y_all[y_all["sample"] == "BOCHUM"]
y_bochum = y_bochum["pain_sensitivity.lib"].values.ravel()

y_essen = y_all[y_all["sample"] == "ESSEN"]
y_essen = y_essen["pain_sensitivity.lib"].values.ravel()

#import matplotlib.pyplot as plt
#plt.plot(y_essen, y_essen_old)
#plt.show()

X_bochum = X_bochum[~np.isnan(y_bochum), :]
y_bochum = y_bochum[~np.isnan(y_bochum)]
#y_bochum_old = y_bochum_old[~np.isnan(y_bochum_old)]

X_essen = X_essen[~np.isnan(y_essen), :]
y_essen = y_essen[~np.isnan(y_essen)]
#y_essen_old = y_essen_old[~np.isnan(y_essen_old)]

#####################################################################################################
# conn masking
####################################################################################################
X = np.vstack((X_bochum, X_essen))
y = np.append(y_bochum, y_essen)

connmask = False
if connmask:
    import matplotlib.pyplot as plt
    from nilearn.connectome import vec_to_sym_matrix, sym_matrix_to_vec
    import nilearn.plotting as plt


    # bochum conn mask
    x = np.array(sym_matrix_to_vec(cm_bochum.mean_, discard_diagonal=True))
    N = int(np.floor(len(x)*0.5) ) # biggest 10%
    idx = np.argsort(abs(x))[-N:]
    X_bochum = X_bochum[:, idx]

    x[idx]=1
    plt.plot_matrix(vec_to_sym_matrix(x, diagonal=np.repeat(0, cm_bochum.mean_.shape[0])))
    plt.show()

    # essen conn mask
    x = np.array(sym_matrix_to_vec(cm_essen.mean_, discard_diagonal=True))
    N = int(np.floor(len(x)*0.5) ) # biggest 10%
    idx = np.argsort(abs(x))[-N:]
    X_essen = X_essen[:, idx]

    x[idx]=1
    plt.plot_matrix(vec_to_sym_matrix(x, diagonal=np.repeat(0, cm_essen.mean_.shape[0])))
    plt.show()

    # bochum+essen conn mask
    x = np.array([sym_matrix_to_vec(cm_essen.mean_, discard_diagonal=True),
                  sym_matrix_to_vec(cm_bochum.mean_, discard_diagonal=True)])
    print "*********"
    print x.shape
    x = np.average(x, axis=0)
    N = int(np.floor(len(x)*0.5) ) # biggest 10%
    idx = np.argsort(abs(x))[-N:]
    X = X[:, idx]

    x[idx]=1
    plt.plot_matrix(vec_to_sym_matrix(x, diagonal=np.repeat(0, cm_essen.mean_.shape[0])))
    plt.show()


#print ts_bochum.shape
#ts_all = np.append(ts_bochum,ts_essen, axis=0)
#print ts_all.shape
#X_all, cm_all = load.compute_connectivity(ts_all, kind="tangent")

#####################################################
# scale response?
#y_essen = preprocessing.scale(y_essen)
#y_bochum = preprocessing.scale(y_bochum)
#import matplotlib.pyplot as plt
#plt.hist(y_essen)
#plt.show()

# archtanh features
#X_essen = np.arctanh(X_essen)
#X_bochum = np.arctanh(X_bochum)
#X = np.arctanh(X)


#######################################

mymodel, p_grid = models.pipe_scale_fsel_model(scaler=preprocessing.RobustScaler())
p_grid = {'fsel__k': [500], 'model__alpha': [.001, .01, .05, .1], 'model__l1_ratio': [.4]}
#p_grid = {'fsel__k': [500], 'model__alpha': [.001, .01, .05, .1], 'model__l1_ratio': [.95, .99, .999]}
#p_grid = {'fsel__k': [450, 475, 500, 525, 550], 'model__alpha': [0.005, .01], 'model__l1_ratio': [.1, .2, .3, .4, .5, .8, .9, .95, .99, .999]}

m, avg_model = train.train(X, y, mymodel, p_grid, "Bochum + Essen", nested=False)
train.get_full_coef(X, m)

m, avg_model = train.train(X_bochum, y_bochum, mymodel, p_grid, "Bochum", nested=False)
#print "** Score BOCHUM sclaled dataset:\t" + str(-mean_squared_error(y_pred=m.predict(X_bochum), y_true=y_bochum_old))
print "** Score ESSEN dataset:\t" + str(-mean_squared_error(y_pred=m.predict(X_essen), y_true=y_essen))
train.get_full_coef(X, m)

m, avg_model = train.train(X_essen, y_essen, mymodel, p_grid, "Essen", nested=False)
#print "** Score ESSEN dataset:\t" + str(-mean_squared_error(y_pred=m.predict(X_essen), y_true=y_essen_old))
print "** Score BOCHUM dataset:\t" + str(-mean_squared_error(y_pred=m.predict(X_bochum), y_true=y_bochum))



