targets1=[
            'PCA_pain_sensitivity', 'mean_QST_pain_sensitivity', 'mean_QST_pain_sensitivityd2',
            'CPT', 'HPT', 'MPT_log_geomean', 'CDT_log_mean', 'WDT_log_mean', 'WUR_log_mean',

            'Glx_mean', 'Glx_overall', 'tCR_Glx_mean',
            'GABA_mean', 'GABA_overall', 'tCR_GABA_mean',

            'tCr_GABA_rACC', 'tCr_GABA_dACC', 'tCr_GABA_ins', 'tCr_Glx_rACC', 'tCr_Glx_dACC', 'tCr_Glx_ins',
            'tCr_Glx_DLPFC', 'tCr_Glx_thal',
            'ads_k', 'psq', 'pcs_catastrophizing', 'pcs_rumination', 'anx_state', 'anx_trait',

          'height', 'weight', 'edu', 'alk_per_w', 'alk_per_occ', 'age', 'food', 'bmi',

          'MPS_heavy_mean', 'MPS_light_mean',  'CPTd2', 'HPTd2', 'MPT_log_geomeand2', 'CDT_log_meand2',
          'WDT_log_meand2', 'MPS_heavy_meand2', 'MPS_light_meand2', 'WUR_log_meand2',
          'GABAvsCr_rACC', 'GABAvsCr_dACC', 'GABAvsCr_ins', 'GLXvsCr_rACC', 'GLXvsCr_dACC', 'GLXvsCr_DLPFC',
          'GLXvsCr_ins', 'GLXvsCr_thal'
    ]

for target in targets1:

    ####################################################################################################
    # load data
    ####################################################################################################
    from sklearn.externals import joblib
    import numpy as np
    import pandas as pd

    connectivity_biomarkers = joblib.load("connectivity_biomarkers_s122_friston.sav")
    kind = "tangent"
    X_orig = connectivity_biomarkers[kind]

    X_orig=np.arctanh(X_orig)

    data_filename = '/Users/tspisak/projects/Mercure_rest/data/data.csv'
    data = pd.read_csv(data_filename, sep=",")  # .values.ravel()

    print list(data)
    print target
    # excludes:
    data = data[3:]  # exclude pilot data
    data = data[data["ID"] != 8]  # exclude subj 8
    data = data[data["ID"] != 21]  # exclude subj 21
    data = data[data["ID"] != 22]  # exclude subj 22
    data = data[data["ID"] != 40]  # exclude subj 22
    #print len(data)



    #target = 'tCr_GABA_rACC'  # something with kcv=10
    #target = 'anx_state'  # quite OK, better withZ Z-score

    outcome = data[target].values.ravel()

    #
    #outcome = data['GABAvsCr_rACC'].values.ravel() # works with elastic net

    Y_orig = outcome[~np.isnan(outcome)]
    if Y_orig.std() < 0.0001:
        print "upscale"
        Y_orig = Y_orig*1000
    #print Y_orig
    # scale data
    from sklearn import preprocessing

    X_orig = X_orig[~np.isnan(outcome)]

    ##########################################################################
    #trainx = np.random.choice(range(len(Y_orig)), 30, replace=False)
    #X_orig = X_orig[trainx]
    #Y_orig = Y_orig[trainx]
    ##########################################################################

    #X = preprocessing.minmax_scale(X_orig)
    #X = preprocessing.scale(X_orig)
    #X = preprocessing.minmax_scale(X_orig, feature_range=(-1, 1))
    #print(len(X))
    #print X.shape
    #interc=np.ones((len(X),1))
    #print interc.shape
    #X=np.hstack((X, interc) ) # intercept term

    #add confounder var!!
    #conf = data['anx_state'] # has a very good prediction
    #conf = np.array(conf[~np.isnan(outcome)])
    #conf.shape = (35,1)
    #print X_orig.shape
    #print conf
    #X_orig = np.hstack((X_orig, conf))
    # remove confounder nan
    #conf = data['anx_state'] # has a very good prediction
    #conf = np.array(conf[~np.isnan(outcome)])
    #X_orig = X_orig[~np.isnan(conf)]
    #Y_orig = Y_orig[~np.isnan(conf)]

    #no sclaing here
    X = X_orig
    Y = Y_orig
    #Y = preprocessing.minmax_scale(Y_orig)
    #Y = preprocessing.scale(Y_orig)
    #Y = preprocessing.minmax_scale(Y_orig, feature_range=(-1, 1))

    import matplotlib.pyplot as plt
    #for i in range(len(X)):
    #    plt.hist(X[i])
    #    plt.show()

    def model(k):
        ####################################################################################################
        # SVR
        ####################################################################################################
        from sklearn.feature_selection import SelectKBest, f_regression
        from sklearn.svm import SVR, LinearSVR
        from sklearn.linear_model import LassoCV, ElasticNetCV
        #loo = LeaveOneOut(len(outcome_pain_sens))
        # Define the dimension reduction to be used.
        # Here we use a classical univariate feature selection based on F-test,
        # We set the number of features to be selected to k
        feature_selection = SelectKBest(f_regression, k=k)
        from sklearn.linear_model import LogisticRegression, Ridge
        from sklearn.grid_search import GridSearchCV
        #svr = LinearSVR(loss='hinge', intercept_scaling=1000)
        svreg = LinearSVR(fit_intercept=True, C=1)
        svreg = SVR()
        svreg = GridSearchCV(SVR(),
                          param_grid={'C': [.1, .5, 1., 5., 10., 50., 100.]},
                          scoring='neg_mean_squared_error', n_jobs=1)
        svreg = ElasticNetCV(cv=20, max_iter=10000, fit_intercept=True)
        #build pipeline
        from sklearn.pipeline import Pipeline
        scaler = preprocessing.StandardScaler()
        svr = Pipeline([('scaler', scaler), ('fsel', feature_selection), ('svr', svreg)])
        #svr = Pipeline([('fsel', feature_selection), ('svr', svreg)])


        #svr.fit(X_scaled, Y)
        return svr

    ####################################################################################################
    # fit and evaluate
    ####################################################################################################
    from sklearn.model_selection import cross_val_predict
    from sklearn.cross_validation import KFold


    def RepCrossVal(model, X, y, y_orig_std, kcv=10, rep=100, learning_curve_n=len(X)):
        from sklearn.cross_validation import cross_val_score, LeaveOneOut
        #kcv = 10 #k-fold corssval
        #rep = 10 #repeated crossval
        # repeated crossvalidation

        all_mse = []
        for i in range(rep):
            train = np.random.choice(range(len(Y)), learning_curve_n, replace=False)
            X_ = X[train]
            y_ = y[train]
            #X_ = preprocessing.scale(X_)
            #Y_ = preprocessing.scale(X_)
            cv = KFold(n=len(y_), n_folds=min(kcv, len(y_)), shuffle=True)
            mse = cross_val_score(model, X_, y_, scoring='neg_mean_squared_error', cv=cv, n_jobs=1)
            all_mse.append(mse.mean()*y_orig_std)

        #print all_mse
        #plt.hist(all_mse)
        #plt.title("Histogram of cross-validated MSE")
        #plt.show()
        #print "Mean CV MSE"
        #print np.mean(all_mse)
        return [np.median(all_mse), np.percentile(all_mse, 75) - np.percentile(all_mse, 25)]


    #################################################################################################
    #################################################################################################
    #################################################################################################
    #################################################################################################
    #################################################################################################
    from multiprocessing import Pool
    #########################################################################################
    #######################################################################################
    #learning curve
    #######################################################################################

    def workhorse2(n):
        svr = model(k=10)
        #print n
        #test = np.random.choice(range(len(Y)), n, replace=False)
        return RepCrossVal(svr, X, Y, 1, kcv=n, rep=3, learning_curve_n=n)


    print "Run this!"

    #num_samples = range(10, len(Y)+1, 2)
    #pool2 = Pool()
    #scores = pool2.map(workhorse2, num_samples)
    #pool2.close()
    #pool2.join()
    #
    #scores = np.array(scores)
    #
    #fig, ax = plt.subplots()
    #ax.scatter(num_samples, scores[:,0]*100, edgecolors=(0, 0, 0))
    #plt.errorbar(num_samples, scores[:,0]*100, scores[:,1]*100, linestyle='None', marker='^')
    #ax.set_xlabel('N')
    #ax.set_ylabel(
    #    'neg_mean_squared_error')
    #plt.show()

    ####################

    kcv = 20
    rep = 2
    scores = []
    ks = [6, 8, 10, 13, 15, 18, 20, 25, 30, 50, 70, 100]#range(10, X.shape[1], X.shape[1]/8)

    def workhorse(k):
        #print k
        svr = model(k=k)
        return RepCrossVal(svr, X, Y, 1, kcv=kcv, rep=rep)

    #pool = Pool(8)
    #scores = pool.map(workhorse, ks)
    #pool.close()

    #pool = Pool(processes=8)
    #scores = pool.map(workhorse, ks)
    #pool.close()


    for k in ks:
        print k
        svr = model(k=k)
        scores.append(RepCrossVal(svr, X, Y, 1, kcv=kcv, rep=rep))

    #fig, ax = plt.subplots()
    #ax.scatter(ks, scores, edgecolors=(0, 0, 0))
    #ax.set_xlabel('k (feature number)')
    #ax.set_ylabel('median neg_mean_squared_error')
    #plt.show()

    #from sklearn.cross_validation import cross_val_score

    #scores = []
    #ks = range(2, 39)
    #ks=[10, 20]
    #for k in ks:
    #    #print k
    #    sc = cross_val_score(svr, X, Y, scoring='neg_mean_squared_error', cv=k)
    #    #print sc.mean()
    #    scores.append(sc.mean())
    scores = np.array(scores)
    #print scores

    #fig, ax = plt.subplots()
    #ax.scatter(ks, scores[:,0]*100, edgecolors=(0, 0, 0))
    #plt.errorbar(ks, scores[:,0]*100, scores[:,1]*100, linestyle='None', marker='^')
    #ax.set_xlabel('k')
    #ax.set_ylabel(
    #    'neg_mean_squared_error')
    #plt.show()

    #print "**********"

    #from sklearn.model_selection import RepeatedKFold
    #print scores[:,0].argmax()
    opt_k = ks[ scores[:,0].argmax() ]

    #print ks
    #print "Optimal feature number:"
    #print opt_k
    opt_k = opt_k#input("Selected optimal k:")

    print "Optimal number of features"
    print opt_k
    print "Optimal Sqrt Mean CV MSE"
    print np.sqrt( -scores[:,0].max() )
    print "Data sd:"
    print Y.std()


    # do crossval predict myself!!
    #predicted = np.zeros(len(Y))

    #print "++++++++++++++++++++++++++++++++++++++++++"
    #cv = KFold(n=len(Y), n_folds=kcv, shuffle=True)
    #for train, test in cv:
    #    mod = model(k=opt_k)
    #    mod.fit(X[train], Y[train])
    #    predicted[test]=mod.predict(X[test])
    #
    #print predicted
    #print Y
    print "++++++++++++++++++++++++++++++++++++++++++"

    predicted = cross_val_predict(model(k=opt_k), X, Y, cv=kcv)
    fig, ax = plt.subplots()
    ax.scatter(Y, predicted, edgecolors=(0, 0, 0))
    ax.plot([Y.min(), Y.max()],
                [Y.min(), Y.max()],
                'k--',
                lw=2)
    ax.set_xlabel('Y')
    ax.set_ylabel('Pred')
    plt.title("%s %s %d %s %f" % (kind, target, opt_k, "MSE/Std: ", -scores[:,0].max()/(Y.std()*Y.std()) ) )
    plt.show()

