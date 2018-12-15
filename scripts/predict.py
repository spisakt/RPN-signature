#!/usr/bin/env python
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
import pandas as pd
import numpy as np
from sklearn.externals import joblib
from PAINTeR import global_vars
import PAINTeR.connectivity as conn
from PAINTeR import plot
from PAINTeR import model

# do robust regression to de-weight outliers
robust = False

# load trained model
mod = joblib.load(global_vars._RES_PRED_MOD_)

#######################################
# cross-val predict BOCHUM sample     #
#######################################
# load pain sensitivity data (excluded)
df = pd.read_csv(global_vars._RES_BOCHUM_TABLE_EXCL_)
y = df['mean_QST_pain_sensitivity']
bochum_y_mean = np.mean(y)
# load features (excluded)
X = joblib.load(global_vars._FEATURE_BOCHUM_)

pred = model.evaluate_crossval_prediction(mod, X, y, outfile=global_vars._PLOT_BOCHUM_PREDICTION_, robust=robust)
df['prediction'] = pred
df.to_csv(global_vars._RES_BOCHUM_TABLE_EXCL_)

############################
# predict ESSEN sample     #
############################
# load pain sensitivity data (excluded)
df = pd.read_csv(global_vars._RES_ESSEN_TABLE_EXCL_)
y = df['mean_QST_pain_sensitivity']

# load features (excluded)
X = joblib.load(global_vars._FEATURE_ESSEN_)

pred = model.evaluate_prediction(mod, X, y, orig_mean=bochum_y_mean, outfile=global_vars._PLOT_ESSEN_PREDICTION_,
                          robust=robust)
df['prediction'] = pred
df.to_csv(global_vars._RES_ESSEN_TABLE_EXCL_)

############################
# predict SZEGED sample    #
############################
# load pain sensitivity data (excluded)
df = pd.read_csv(global_vars._RES_SZEGED_TABLE_EXCL_)
y = df['mean_QST_pain_sensitivity']
y = y - np.mean(y)

# load features (excluded)
X = joblib.load(global_vars._FEATURE_SZEGED_)

pred = model.evaluate_prediction(mod, X, y, orig_mean=bochum_y_mean, outfile=global_vars._PLOT_SZEGED_PREDICTION_,
                          robust=robust)
df['prediction'] = pred
df.to_csv(global_vars._RES_SZEGED_TABLE_EXCL_)

############################
# predict SZEGED+ESSEN     #
############################

# load pain sensitivity data (excluded)
y_essen = pd.read_csv(global_vars._RES_SZEGED_TABLE_EXCL_)['mean_QST_pain_sensitivity'].tolist()
y_szeged = pd.read_csv(global_vars._RES_ESSEN_TABLE_EXCL_)['mean_QST_pain_sensitivity'].tolist()

y =np.array(y_essen + y_szeged)

# load features (excluded)
X = np.vstack( (joblib.load(global_vars._FEATURE_SZEGED_),
              joblib.load(global_vars._FEATURE_ESSEN_) ) )

model.evaluate_prediction(mod, X, y, orig_mean=bochum_y_mean, outfile=global_vars._PLOT_ESSEN_SZEGED_PREDICTION_,
                          robust=robust,
                          covar=np.concatenate( (np.repeat(0, len(y_essen)), np.repeat(1, len(y_szeged)) ), axis=0 ).tolist()
                          )