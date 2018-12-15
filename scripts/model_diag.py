#!/usr/bin/env python
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.externals import joblib
from PAINTeR import global_vars
from PAINTeR import model
import seaborn as sns


# load trained model
mod = joblib.load(global_vars._RES_PRED_MOD_)

#######################################
# learning curve BOCHUM sample     #
#######################################
# load pain sensitivity data (excluded)
y = pd.read_csv(global_vars._RES_BOCHUM_TABLE_EXCL_)['mean_QST_pain_sensitivity']
X = joblib.load(global_vars._FEATURE_BOCHUM_)

Ns = [20, 22, 24, 26, 28, 30, 32, 34, 35]
# do calculation
#train, test = model.learning_curve(mod, X, y.tolist(), Ns=Ns)
#data = [train, test]
#joblib.dump(data, "learningcurve.sav")

data = joblib.load("learningcurve.sav")

#print len(range(20, len(y)+1))
#print len(np.sqrt(np.abs(data[1]))[29:])

plt.figure(figsize=(6,2))
sns.lineplot(Ns, np.abs(data[0]))
grid = sns.lineplot(Ns, np.abs(data[1]))
grid.set(yscale="log")

figure = plt.gcf()
figure.savefig(global_vars._PLOT_LEARNING_CURVE_, bbox_inches='tight')
plt.close(figure)

plt.show()

##############################################################
# specificitiy calculations plots                           #
##############################################################

table_bochum = pd.read_csv(global_vars._RES_BOCHUM_TABLE_EXCL_)


