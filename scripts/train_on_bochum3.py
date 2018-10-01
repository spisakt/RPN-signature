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

y = load.load_QST_data()  # default load Bochum "QST_pain_sesnitiviy" from QST data
ts = load.load_timeseries_sav()
X = load.compute_connectivity(ts, kind="tangent")

# eliminate NA-s
X = X[~np.isnan(y)]
y = y[~np.isnan(y)]

print X[1]

mymodel, p_grid = models.pipe_scale_fsel_model()
#mymodel, p_grid = models.pipe_scale_fsel_ridge()


p_grid = {'fsel__k': [20, 30, 35, 40, 50], 'model__alpha': [.01, .05, .1], 'model__l1_ratio': [.1, .3, .5, .8, .9]}
#p_grid = {'fsel__k': [20, 30, 35, 40, 50], 'model__alpha': [.01, .05, .1]} # for pure ridge
############ hyperparamter tuning with gridserach and model evaluation ####################


k = 15  # number of folds

#inner_cv = modsel.RepeatedKFold(n_splits=k, n_repeats=50)
#outer_cv = modsel.KFold(n_splits=k)
#inner_cv = modsel.RepeatedSortedStratifiedKFold(n_splits=k, n_repeats=15)
#outer_cv = modsel.RepeatedSortedStratifiedKFold(n_splits=k, n_repeats=2)
inner_cv = LeaveOneOut()
outer_cv = LeaveOneOut()

#mymodel.fit(X, y)
#print(cross_val_score(mymodel.predict(X), y))

clf = GridSearchCV(estimator=mymodel, param_grid=p_grid, cv=inner_cv, scoring="neg_mean_squared_error", verbose=False, return_train_score=False, n_jobs=8)

print "** Number of subjects: " + str(len(y))

clf.fit(X, y)


print "*** Score on mean as model: " + str(-mean_squared_error(np.repeat(y.mean(), len(y)), y))
print "**** Non-nested analysis ****"
print "** Best hyperparameters: " + str(clf.best_params_)

print "** Score on full data as training set:\t" + str(-mean_squared_error(y_pred=clf.best_estimator_.predict(X), y_true=y))
print "** Best Non-nested cross-validated score on test:\t" + str(clf.best_score_)


print "**** Nested analysis ****"


#nested_scores = cross_val_score(clf, X, y, cv=outer_cv, scoring="explained_variance")
#print "** Nested Score on test:\t" + str(nested_scores.mean())
# this above has the same output as this below:

best_params = []
predicted = np.zeros(len(y))
actual = np.zeros(len(y))
nested_scores_train = np.zeros(outer_cv.get_n_splits(X))
nested_scores_test = np.zeros(outer_cv.get_n_splits(X))
nested_scores_test2 = np.zeros(outer_cv.get_n_splits(X))
i = 0
# doing the crossval itewrations manually
print "model\tinner_cv mean score\touter vc score"
for train, test in outer_cv.split(X, y):
    clf.fit(X[train], y[train])
    # plot histograms to check distributions
    #bins = np.linspace(-1.5, 1.5, 6)
    #pyplot.hist(y[train], bins, alpha=0.5, label='train')
    #pyplot.hist(y[test], bins, alpha=0.5, label='test')
    #pyplot.legend(loc='upper right')
    #pyplot.show()

    print str(clf.best_params_) + " " + str(clf.best_score_) + " " + str(clf.score(X[test], y[test]))
    predicted[i] = clf.predict(X[test])
    actual[i] = y[test]

    best_params.append(clf.best_params_)
    nested_scores_train[i] = clf.best_score_
    nested_scores_test[i] = clf.score(X[test], y[test])
    # clf.score is the same as calculating the score to the prediced values of the test dataset:
    #nested_scores_test2[i] = explained_variance_score(y_pred=clf.predict(X[test]), y_true=y[test])
    i = i+1

print "*** Score on mean as model:\t" + str(-mean_squared_error(np.repeat(y.mean(), len(y)), y))
print "** Mean score in the inner crossvaludation (inner_cv):\t" + str(nested_scores_train.mean())
print "** Mean Nested Crossvalidation Score (outer_cv):\t" + str(nested_scores_test.mean())

print "Explained Variance: " +  str( 1- nested_scores_test.mean()/-mean_squared_error(np.repeat(y.mean(), len(y)), y) )

#plot the prediction of the outer cv
fig, ax = plt.subplots()
ax.scatter(actual, predicted, edgecolors=(0, 0, 0))
ax.plot([y.min(), y.max()],
           [y.min(), y.max()],
           'k--',
           lw=2)
ax.set_xlabel('Pain Sensitivity')
ax.set_ylabel('Predicted (Nested LOO)')
plt.title("Expl. Var.:" +  str( 1- nested_scores_test.mean()/-mean_squared_error(np.repeat(y.mean(), len(y)), y) ) )
plt.show()


