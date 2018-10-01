#!/usr/bin/env python
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot

from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import r2_score, explained_variance_score, mean_squared_error

import PAINTeR.load_project_data as load
import PAINTeR.models as models
import PAINTeR.model_selection as modsel
from sklearn.model_selection import LeaveOneOut

from sklearn.feature_selection import f_regression, mutual_info_regression, SelectKBest

import matplotlib.pyplot as plt

ts_cpac = load.load_timeseries_sav("/Users/tspisak/tmp/test/timeseries_64_friston_cpac_s1.sav")
ts_pumi = load.load_timeseries_sav("/Users/tspisak/tmp/test/timeseries_64_friston_pumi_s1.sav")

#plt.plot(ts_cpac[37][:,50])
#plt.plot(ts_pumi[37][:,50])

#plt.show()

#print np.corrcoef(ts_cpac[37][:,50], ts_pumi[37][:,50])

X_cpac, cm_cpac = load.compute_connectivity(ts_cpac, kind="tangent")
X_pumi, cm_pumi = load.compute_connectivity(ts_pumi, kind="tangent")

y = load.load_QST_data(exclude_ids=[1,2,3, 8, 21, 22, 40])

X_cpac = X_cpac[~np.isnan(y)]
X_pumi = X_pumi[~np.isnan(y)]
y = y[~np.isnan(y)]

F_cpac = f_regression(X_cpac, y)
F_pumi = f_regression(X_pumi, y)

N=1000

_x = F_cpac[0].argsort()[-N:][::-1]
_y = F_pumi[0].argsort()[-N:][::-1]

print F_cpac[0][1883]
plt.plot(X_cpac[:,36])
plt.plot(X_pumi[:,36])
plt.show()
plt.plot(X_cpac[:,1883])
plt.plot(X_pumi[:,1883])
plt.show()
plt.plot(X_cpac[:,1636])
plt.plot(X_pumi[:,1636])
plt.show()

_z = _x
_z = np.append(_z,_y)
print N*2-len(np.unique(_z))


mymodel, p_grid = models.pipe_scale_fsel_model()

################
#X = X_cpac
################

p_grid = {'fsel__k': [100, 200, 500, 1000, 2000], 'model__alpha': [.01, .1, 1.5, .5, .8, 1, 2], 'model__l1_ratio': [.1, .3, .5, .8, .9]}

inner_cv = LeaveOneOut()
outer_cv = LeaveOneOut()

#mymodel.fit(X, y)
#print(cross_val_score(mymodel.predict(X), y))


print "** Number of subjects: " + str(len(y))

print "*************************************************************************************************************"
print "*** PUMI ***"
clf = GridSearchCV(estimator=mymodel, param_grid=p_grid, cv=inner_cv, scoring="neg_mean_squared_error", verbose=False, return_train_score=False, n_jobs=8)
clf.fit(X_pumi, y)

print "*** Score on mean as model: " + str(-mean_squared_error(np.repeat(y.mean(), len(y)), y))
print "**** Non-nested analysis ****"
print "** Best hyperparameters: " + str(clf.best_params_)

print "** Score on full data as training set:\t" + str(-mean_squared_error(y_pred=clf.best_estimator_.predict(X_pumi), y_true=y))
print "** Best Non-nested cross-validated score on test:\t" + str(clf.best_score_)


print "****** Score on CPAC data as test set:\t" + str(-mean_squared_error(y_pred=clf.best_estimator_.predict(X_cpac), y_true=y))


print "*** CPAC ***"
clf = GridSearchCV(estimator=mymodel, param_grid=p_grid, cv=inner_cv, scoring="neg_mean_squared_error", verbose=False, return_train_score=False, n_jobs=8)
clf.fit(X_cpac, y)

print "*** Score on mean as model: " + str(-mean_squared_error(np.repeat(y.mean(), len(y)), y))
print "**** Non-nested analysis ****"
print "** Best hyperparameters: " + str(clf.best_params_)

print "** Score on full data as training set:\t" + str(-mean_squared_error(y_pred=clf.best_estimator_.predict(X_cpac), y_true=y))
print "** Best Non-nested cross-validated score on test:\t" + str(clf.best_score_)


print "****** Score on PUMI data as test set:\t" + str(-mean_squared_error(y_pred=clf.best_estimator_.predict(X_pumi), y_true=y))




