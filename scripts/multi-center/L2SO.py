import sys
sys.path.append('../..')
from PAINTeR import connectivity # in-house lib used for the RPN-signature
from PAINTeR import plot # in-house lib used for the RPN-signature
from PAINTeR import model # in-house lib used for the RPN-signature
import numpy as np # hi old friend
import pandas as pd

from sklearn.preprocessing import StandardScaler
from nilearn.connectome import ConnectivityMeasure

from matplotlib.colors import ListedColormap
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("white")

from sklearn.linear_model import ElasticNet, Ridge
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.model_selection import LeaveOneOut, KFold, GroupKFold, LeavePGroupsOut
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error, r2_score, explained_variance_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_validate

import joblib

#%%%%%%%%%%%%%%%%%%%%%%%%%%%
thres_mean_FD = 0.15 # mm
scrub_threshold = 0.15 # mm
thres_perc_scrub = 30 # % scubbed out

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# load bochum data
df_bochum = pd.read_csv("../../res/bochum_sample_excl.csv")
df_essen = pd.read_csv("../../res/essen_sample_excl.csv")
df_szeged = pd.read_csv("../../res/szeged_sample_excl.csv")
df_bochum['study']='bochum'
df_essen['study']='essen'
df_szeged['study']='szeged'
df=pd.concat((df_bochum, df_essen, df_szeged), sort=False)
df=df.reset_index()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%

timeseries = []
perc_scrubbed = []
for i, f in enumerate(df['ts_file']):
    f = '../..' + f.split('/..')[1]
    f_scrub = f.split('.tsv')[0] + '-scrubbed.tsv'

    ts = pd.read_csv(f_scrub).iloc[:, 1:]  # here we can omit global signal...

    fd_file = df["fd_file"].values[i]
    fd_file = '../..' + fd_file.split('/..')[1]
    fd = pd.read_csv(fd_file).values.ravel().tolist()
    fd = [0] + fd

    perc_scrubbed.append(100 - 100 * len(ts.shape) / len(fd))
    timeseries.append(ts.values)

# region names
labels = ts.columns.values
l = pd.read_csv('../../data/atlas_relabeled.tsv', sep="\t")
modules = np.insert(l['modules'].values, 0, "GlobSig")

correlation_measure = ConnectivityMeasure(kind='partial correlation', vectorize=True, discard_diagonal=True)
X = correlation_measure.fit_transform(timeseries) # these are the features
mat=correlation_measure.mean_
#mat=mat[1:, 1:] #fisrt row and column is global signal
mat[range(mat.shape[0]), range(mat.shape[0])] = 0 # zero diag

y = df.mean_QST_pain_sensitivity

# create groups to balance-out cross-validations
n_szeged = np.sum(df.study == 'szeged')  # size of the smallest study
n_essen = np.sum(df.study == 'essen')
n_bochum = np.sum(df.study == 'bochum')
print(n_bochum, n_essen, n_szeged)

groups = np.zeros(len(df), dtype=int)

g = 0
i = 0
while i < n_bochum:
    groups[i] = g
    # groups[i+1] = g
    i += 1
    g += 1

g = 0
i = n_bochum
while i < n_bochum + n_essen:
    groups[i] = g
    # groups[i+1] = g
    i += 1
    g += 1
g = 0
i = n_bochum + n_essen
while i < len(df):
    groups[i] = g
    i += 1
    g += 1

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def pipe_scale_fsel_elnet(scaler=preprocessing.RobustScaler(),
                          fsel=SelectKBest(f_regression),
                          model=ElasticNet(max_iter=1000000),
                          # p_grid = {'fsel__k': [10, 50, 100, 200, 500, 700, 1000, 2000, 3000, 4000, 5000, 'all'], 'model__alpha': [.001, .01, .1, 1, 10], 'model__l1_ratio': [0.001, .1, .3, .5, .7, .9, .999]
                          # p_grid = {'fsel__k': [25, 100, 1000], 'model__alpha': [.001, .01, .1, 1], 'model__l1_ratio': [.001, .5, .999]
                          p_grid={'fsel__k': [25, 50, 100, 1000, 5000, 'all'],
                              'model__alpha': [0.001,  0.1, 1, 10, 100],
                                  'model__l1_ratio': [0.0001, .5, 0.9999]
                                  }):
    mymodel = Pipeline(
        [('scaler', scaler),
         ('fsel', fsel),
         ('model', model)])
    return mymodel, p_grid


model, p_grid = pipe_scale_fsel_elnet()

outer_cv = LeavePGroupsOut(2)  # Leave-two-sudy-out
inner_cv = LeaveOneOut()  # do 30-fold quasi-balanced splits within the other two studies for hyperparam optimization.
clf = GridSearchCV(estimator=model, param_grid=p_grid, cv=inner_cv,
                   scoring="neg_mean_squared_error", verbose=True, return_train_score=False,
                   n_jobs=-1)

all_models = []
best_params = []
predicted = np.zeros(len(y))
nested_scores_train = np.zeros(outer_cv.get_n_splits(X, groups=df.study))
nested_scores_test = np.zeros(outer_cv.get_n_splits(X, groups=df.study))

print("model\tinner_cv mean score\touter vc score")
i = 0
for train, test in outer_cv.split(X, y, groups=df.study):
    print(test)
    group_train = groups[train]
    clf.fit(X[train], y[train], groups=group_train)

    print(str(clf.best_params_) + " " + str(clf.best_score_) + " " + str(clf.score(X[test], y[test])))

    all_models.append(clf.best_estimator_)
    best_params.append(clf.best_params_)

    predicted[test] += clf.predict(X[test])  # added, to later construct average

    nested_scores_train[i] = clf.best_score_
    nested_scores_test[i] = clf.score(X[test], y[test])
    i = i + 1

predicted /= 2  # to crete average of the two predictions we had

print("*** Score on mean as model:\t" + str(-mean_squared_error(np.repeat(y.mean(), len(y)), y)))
print("** Mean score in the inner crossvaludation (inner_cv):\t" + str(nested_scores_train.mean()))
print("** Mean Nested Crossvalidation Score (outer_cv):\t" + str(nested_scores_test.mean()))
print("Explained Variance: " + str(1 - nested_scores_test.mean() / -mean_squared_error(np.repeat(y.mean(), len(y)), y)))
print("Correlation: " + str(np.corrcoef(y, predicted)[0, 1]))

plot.plot_prediction(y, predicted, sd=True, outfile='L1SO_all.pdf')

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%

study='bochum'
plot.plot_prediction(y[df.study==study], predicted[df.study==study], sd=True, outfile='L1SO_bochum.pdf')
print("*** Score on mean as model:\t" + str(-mean_squared_error(np.repeat(y[df.study==study].mean(), len(y[df.study==study])), y[df.study==study])))
print("** Mean score in the inner crossvaludation (inner_cv):\t" + str(nested_scores_train[0].mean()))
print("** Mean Nested Crossvalidation Score (outer_cv):\t" + str(nested_scores_test[0].mean()))
print("Explained Variance: " +  str( 1- nested_scores_test[0].mean()/-mean_squared_error(np.repeat(y[df.study==study].mean(), len(y[df.study==study])), y[df.study==study]) ))
print("Correlation: " + str(np.corrcoef(y[df.study==study], predicted[df.study==study])[0,1]))

study='essen'
plot.plot_prediction(y[df.study==study], predicted[df.study==study], sd=True, outfile='L1SO_essen.pdf')
print("*** Score on mean as model:\t" + str(-mean_squared_error(np.repeat(y[df.study==study].mean(), len(y[df.study==study])), y[df.study==study])))
print("** Mean score in the inner crossvaludation (inner_cv):\t" + str(nested_scores_train[1].mean()))
print("** Mean Nested Crossvalidation Score (outer_cv):\t" + str(nested_scores_test[1].mean()))
print("Explained Variance: " +  str( 1- nested_scores_test[1].mean()/-mean_squared_error(np.repeat(y[df.study==study].mean(), len(y[df.study==study])), y[df.study==study]) ))
print("Correlation: " + str(np.corrcoef(y[df.study==study], predicted[df.study==study])[0,1]))

study='szeged'
plot.plot_prediction(y[df.study==study], predicted[df.study==study], sd=True, outfile='L1SO_szeged.pdf')
print("*** Score on mean as model:\t" + str(-mean_squared_error(np.repeat(y[df.study==study].mean(), len(y[df.study==study])), y[df.study==study])))
print("** Mean score in the inner crossvaludation (inner_cv):\t" + str(nested_scores_train[2].mean()))
print("** Mean Nested Crossvalidation Score (outer_cv):\t" + str(nested_scores_test[2].mean()))
print("Explained Variance: " +  str( 1- nested_scores_test[2].mean()/-mean_squared_error(np.repeat(y[df.study==study].mean(), len(y[df.study==study])), y[df.study==study]) ))
print("Correlation: " + str(np.corrcoef(y[df.study==study], predicted[df.study==study])[0,1]))

# save nested cv-predictions
np.savetxt("L1SO_nested_cv_pred.csv", predicted, delimiter=",")

# essen+szeged -> bochum
joblib.dump(all_models[0], 'model_trained_on_szeged.joblib')

# bochum+szeged -> essen
joblib.dump(all_models[1], 'model_trained_on_essen.joblib')

# bochum+essen -> szeged
joblib.dump(all_models[2], 'model_trained_on_bochum.joblib')

