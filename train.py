from sklearn.feature_selection import f_regression, mutual_info_regression, SelectKBest
import matplotlib.pyplot as plt
from matplotlib import pyplot
import PAINTeR.model_selection as modsel
from sklearn.model_selection import LeaveOneOut
import numpy as np

from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import r2_score, explained_variance_score, mean_squared_error

def get_full_coef(X, m, labels="/Users/tspisak/res/PAINTeR/bochum/atlas_relabeled.tsv", plot=True):
    from nilearn.connectome import vec_to_sym_matrix, sym_matrix_to_vec
    import nilearn.plotting as plotting

    featuremask = m.named_steps['fsel'].get_support()

    import pandas as pd
    labels = pd.read_csv(labels, sep="\t")

    RES = np.zeros(X[1].shape)
    RES[featuremask] = m.named_steps['model'].coef_
    mat = vec_to_sym_matrix(RES, diagonal=np.repeat(0, len(labels))) #+1

    # partial correlation, RobustScaler, NoDiagonal, ts_scaling, scaled only on bochum: 23.6%
    # partial correlation, MaxAbsScaler, NoDiagonal, ts_scaling, scaled_together_with_essen: 25.4%
    # partial correlation, MaxAbsScaler, NoDiagonal, ts_scaling, scaled_together_with_essen, noglobsig: 21.5%
    # partial correlation, StandardScaler, NoDiagonal, ts_scaling, scaled_together_with_essen, noglobsig: 19%

    if plot:
        plotting.plot_matrix(mat, figure=(12, 12), labels=labels['modules'].values.tolist(), title="", grid=True)  #['GS']+
        plotting.show()

    return RES, mat, labels

def train(X, y, model, p_grid, name="sample_name", nested=False, model_averaging=True, ):

    inner_cv = LeaveOneOut()
    outer_cv = LeaveOneOut()

    print "*** " + name + " ***"
    print X.shape
    print "** Number of subjects: " + str(len(y))
    clf = GridSearchCV(estimator=model, param_grid=p_grid, cv=inner_cv, scoring="neg_mean_squared_error", verbose=False, return_train_score=False, n_jobs=8)
    clf.fit(X, y)

    print "**** Non-nested analysis ****"
    print "** Best hyperparameters: " + str(clf.best_params_)

    print "** Score on full data as training set:\t" + str(-mean_squared_error(y_pred=clf.best_estimator_.predict(X), y_true=y))
    print "** Score on mean as model: " + str(-mean_squared_error(np.repeat(y.mean(), len(y)), y))
    print "** Best Non-nested cross-validated score on test:\t" + str(clf.best_score_)

    model=clf.best_estimator_

    print "XXXXX Explained Variance: " + str(
        1 - clf.best_score_ / -mean_squared_error(np.repeat(y.mean(), len(y)), y))

    avg_model = None
    if nested:
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
        avg = []
        # doing the crossval itewrations manually
        print "model\tinner_cv mean score\touter vc score"
        for train, test in outer_cv.split(X, y):
            clf.fit(X[train], y[train])

            # model avaraging
            RES, mat, labels = get_full_coef(X[train], clf.best_estimator_, plot=False)
            avg.append(RES)
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

        avg_model = np.mean(np.array(avg), axis=0)

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

    return model, avg_model