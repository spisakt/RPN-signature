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

# load pain sensitivity data (excluded)
y = pd.read_csv(global_vars._RES_BOCHUM_TABLE_EXCL_)['mean_QST_pain_sensitivity']

# load features (excluded)
X = joblib.load(global_vars._FEATURE_BOCHUM_)

# define model
mymodel, p_grid = model.pipe_scale_fsel_elnet()

# train model with cross-validation
m, avg_model, all_models = model.train(X, y, mymodel, p_grid, nested=False)

# serialise model
joblib.dump(m, global_vars._RES_PRED_MOD_)