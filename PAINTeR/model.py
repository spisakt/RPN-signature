import numpy as np
from scipy import stats
import statsmodels.api as sm
from sklearn.linear_model import ElasticNet
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_predict

import PAINTeR.plot as plot

def pipe_scale_fsel_elnet(scaler=preprocessing.MaxAbsScaler(),
                          fsel=SelectKBest(f_regression),
                          model=ElasticNet(max_iter=100000),
                          p_grid = {'fsel__k': [20, 30, 40, 50, 60, 70, 80], 'model__alpha': [.0005, .001, .005, .01, .05, .1], 'model__l1_ratio': [.000000001, .1, .2, .3, .4, .5, .6, .7, .8, .9, .999999999]}
                          ):
    mymodel = Pipeline(
        [('scaler', scaler), ('fsel', fsel),
         ('model', model)])
    return mymodel, p_grid

def train(X, y, model, p_grid, nested=False, model_averaging=True, ):

    inner_cv = LeaveOneOut()
    outer_cv = LeaveOneOut()

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
    all_models = []
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
            all_models.append(clf.best_estimator_)
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
        print "Correlation: " + str(np.corrcoef(actual, predicted)[0,1])

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
        plt.title("Expl. Var.:" +  str( 1- nested_scores_test.mean()/-mean_squared_error(np.repeat(y.mean(), len(y)), y) ) +
        "\nCorrelation: " + str(np.corrcoef(actual, predicted)[0, 1]) )
        plt.show()
    else:
        all_models = [model]

    model.fit(X, y) # fot to whole data

    return model, avg_model, all_models


def pred_stat(observed, predicted, robust=False):

    if robust:
        res = sm.RLM(observed, sm.add_constant(predicted)).fit()
        p_value = res.pvalues[1]
        regline = res.fittedvalues
        residual = res.sresid

        # this is a pseudo r_squared, see: https://stackoverflow.com/questions/31655196/how-to-get-r-squared-for-robust-regression-rlm-in-statsmodels
        r_2 = sm.WLS(observed, sm.add_constant(predicted), weights=res.weights).fit().rsquared

    else:
        slope, intercept, r_value, p_value, std_err = stats.linregress(observed, predicted)
        regline = slope*observed+intercept
        r_2 = r_value**2
        residual = observed - regline

    return p_value, r_2, residual, regline


def evaluate_prediction(model, X, y, outfile="", robust=False):
    predicted = model.predict(X)

    p_value, r_2, residual, regline = pred_stat(y, predicted, robust=robust)

    plot.plot_prediction(y, predicted, outfile, robust=robust, sd=True,
                         text="$R^2$ = " + "{:.3f}".format(r_2) + "  p = " + "{:.3}".format(p_value) )



def evaluate_crossval_prediction(model, X, y, outfile="", cv=LeaveOneOut(), robust=False):
    predicted = cross_val_predict(model, X, y, cv=cv)
    plot.plot_prediction(y, predicted, outfile,  robust=robust)
