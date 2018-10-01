# load the packages
import os
import pandas as pd
import numpy as np
from nilearn.connectome import ConnectivityMeasure
import PAINTeR.models as models
from sklearn.model_selection import LeaveOneOut, GridSearchCV
from sklearn.metrics import r2_score, explained_variance_score, mean_squared_error

##########################################################
##########################################################
##########################################################

#   Inputs are the timesseries derived from different ROI

##########################################################
##########################################################
##########################################################
# The calculation mode of model estimation
correlationtype='tangent'
conn_measure = ConnectivityMeasure(kind=correlationtype, vectorize=True, discard_diagonal=True)
###########################################################
###      Here, we load the Essen_PAINTeR dataset.       ###
###########################################################
# The response variable:
data_essen = pd.read_csv("/home/analyser/Documents/PAINTER/essen_painthr.csv", sep=",")
y_essen = data_essen["compositepainsensitivity"].values.ravel()
# The independent variables
path="/home/analyser/Documents/PAINTER/regional_timeseries_essen/"
files=os.listdir(path)
files.sort()
for i in range(len(files)):
    files[i]=path + files[i]
pooled_subjects_essen=[]
for subj in files:
    ts = pd.read_csv(subj, sep="\t")
    pooled_subjects_essen.append(ts.values)


correlation_matrix_essen = conn_measure.fit_transform(pooled_subjects_essen)
X_essen=correlation_matrix_essen
###############################################################
###             Load the Bochum_PAINTeR dataset             ###
###############################################################
# Response variable:
data_bochum=pd.read_csv("/home/analyser/Documents/PAINTER/data.csv",sep=",")
exclude_ids=[1,2,3, 8, 21, 22] #these are the IDs of the


for id in exclude_ids:
    data_bochum = data_bochum[data_bochum["ID"] != id]
includedindicesoffiles=data_bochum[~np.isnan(data_bochum["mean_QST_pain_sensitivity"].values)]["ID"].values
y_bochum=data_bochum["mean_QST_pain_sensitivity"].values.ravel()
y_bochum=y_bochum[~np.isnan(y_bochum)]
subjectID_bochum=pd.read_csv("/home/analyser/projects/PAINTER/res/bochum/subjectsIDs.txt",sep="\t",header=None)
subjID_cpacID={int(subjectID_bochum.values[i, 1].split("/rest")[0][-2:]):subjectID_bochum.values[i, 0] for i in range(len(subjectID_bochum)) }
# for id in exclude_ids:

#exclude_idx_frombochumnumber=[4,28] # here the 4 means idx8,and 28 means idx34. The rest is exluded by default.
# The independent variables
path="/home/analyser/Documents/PAINTER/regional_timeseries_5compcor/"
files=os.listdir(path)
files.sort()
for i in range(len(files)):
    files[i]=path + files[i]
subjID_cpacID_excl={k: v for k, v in subjID_cpacID.items() for i in includedindicesoffiles if k == i}
files_excl=[]
for na in files:
    if int(na.split('extract')[1][-2:]) in subjID_cpacID_excl.values():
        files_excl.append(na)



pooled_subjects_bochum_MAmask=[]
for subj in files_excl:
    ts = pd.read_csv(subj, sep="\t")
    pooled_subjects_bochum_MAmask.append(ts.values)
# Normalize timesseries
pooled_subjects_bochum_MAmask_norm=np.empty_like(pooled_subjects_bochum_MAmask)
for i in range(len(pooled_subjects_bochum_MAmask)):
    for j in range(pooled_subjects_bochum_MAmask[i].shape[1]):
        if pooled_subjects_bochum_MAmask[i][:, j].std() != 0:
            pooled_subjects_bochum_MAmask_norm[i][:, j] = pooled_subjects_bochum_MAmask[i][:, j] / \
                                                          pooled_subjects_bochum_MAmask[i][:, j].std()
        else:
            pooled_subjects_bochum_MAmask_norm[i][:, j] = 0


correlation_matrix_bochum_MAmask = conn_measure.fit_transform(pooled_subjects_bochum_MAmask)
X_bochum_compcor=correlation_matrix_bochum_MAmask
correlation_matrix_cpac = conn_measure.fit_transform(pooled_subjects_cpac)
X_cpac=correlation_matrix_cpac
# The response and predictive values are preprocessed previously, Here the cleaned data are used.
# Independent variable X contains the connectivity strength between every brain area in one row per subject.
# Dependent/response variable Y contains the composite pain sensitivity scores


mymodel, p_grid = pipe_scale_fsel_model()
p_grid = {'fsel__k': [20, 30, 35, 40, 50], 'model__alpha': [.01, .05, .1], 'model__l1_ratio': [.1, .3, .5, .8, .9]}
inner_cv = LeaveOneOut()
outer_cv = LeaveOneOut()


clf = GridSearchCV(estimator=mymodel, param_grid=p_grid, cv=inner_cv, scoring="neg_mean_squared_error", verbose=False, return_train_score=False, n_jobs=8)
y=y_bochum
# X=X_bochum
# X=X_bochum_MAmask_norm
# X=X_bochum_strictMC
X=X_bochum_compcor
# X=X_cpac
# y=y_essen
# X=X_essen
# y=y_bochum_cpac


clf.fit(X, y)


print "*** Score on mean as model: " + str(-mean_squared_error(np.repeat(y.mean(), len(y)), y))
print "**** Non-nested analysis ****"
print "** Best hyperparameters: " + str(clf.best_params_)

print "** Score on full data as training set:\t" + str(-mean_squared_error(y_pred=clf.best_estimator_.predict(X), y_true=y))
print "** Best Non-nested cross-validated score on test:\t" + str(clf.best_score_)


print "**** Nested analysis ****"

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



