########################################################################
# DO SVR
########################################################################

#Load training data
#loaded_model = pickle.load(open(filename, 'rb'))
from sklearn.externals import joblib
connectivity_biomarkers = joblib.load("connectivity_biomarkers.sav")


# Load variable to predict
# in this case:  pain sensitivity

import pandas as pd

outcome_pain_sens_filename = '/Users/tspisak/projects/Mercure_rest/res/reho/pain_sens/pain_sens_included_subj.txt'
outcome_pain_sens = pd.read_csv(outcome_pain_sens_filename, sep=",", header=None).values.ravel()

print "Running SVR!!!"
from sklearn.svm import LinearSVR, SVR

svr = LinearSVR(fit_intercept=False)
#svr = SVR(kernel='linear', C=1e-2)

from sklearn import preprocessing
#X_train_scaled = preprocessing.scale(connectivity_biomarkers['tangent'])
#x = connectivity_biomarkers['tangent'].std(axis=0)

outcome_pain_sens_orig = outcome_pain_sens
outcome_pain_sens = preprocessing.scale(outcome_pain_sens)
#outcome_pain_sens = preprocessing.minmax_scale(outcome_pain_sens)
#outcome_pain_sens = preprocessing.normalize(outcome_pain_sens)
#print(outcome_pain_sens)
print(outcome_pain_sens_orig.std())

#svr.fit(X_train, outcome_pain_sens)
#print(svr)

##############################################################################################################
from sklearn.cross_validation import cross_val_score, LeaveOneOut

mean_scores = []
loo = LeaveOneOut(len(outcome_pain_sens))

#############################
X_train = preprocessing.scale(connectivity_biomarkers['tangent'])
svr.fit(X_train[:-3], outcome_pain_sens[:-3])
prediction = svr.predict(X_train[-3:])
print("************")
print(prediction)
print(outcome_pain_sens[-3:])

from sklearn.model_selection import cross_val_predict
predicted = cross_val_predict(svr, X_train, outcome_pain_sens, cv=loo)
print(predicted[-3:])

#quit()
############################
mist_64_labels_filename = '~/data/atlases/MIST/Parcel_Information/MIST_12.csv'
mist_64_labels = pd.read_csv(mist_64_labels_filename, sep=";")

mist_hierarchy_filename = '/Users/tspisak/data/atlases/MIST/Hierarchy/MIST_PARCEL_ORDER_ROI.csv'
mist_hierarchy = pd.read_csv(mist_hierarchy_filename, sep=",")
mist_64_hierarchy = mist_hierarchy['s12']
mist_64_hierarchy = mist_64_hierarchy.drop_duplicates()
mist_64_labels = mist_64_labels.reindex(mist_64_hierarchy - 1)
mist_64_labels=mist_64_labels.reset_index()

#############################

measures = ["correlation", "partial correlation", "tangent"]
for kind in measures:
    X_train_scaled = preprocessing.scale(connectivity_biomarkers[kind])
    #X_train_scaled = preprocessing.minmax_scale(connectivity_biomarkers[kind])
    #X_train_scaled = connectivity_biomarkers[kind]
    cv_scores = cross_val_score(svr, X_train_scaled,
                                y=outcome_pain_sens, cv=loo, scoring='neg_mean_squared_error')
    mean_scores.append(cv_scores.mean()*outcome_pain_sens_orig.std())
    from sklearn.model_selection import cross_val_predict

    predicted = cross_val_predict(svr, X_train_scaled, outcome_pain_sens, cv=loo)

    predicted_orig=predicted*outcome_pain_sens_orig.std()+outcome_pain_sens_orig.mean()

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.scatter(outcome_pain_sens, predicted, edgecolors=(0, 0, 0))
    ax.plot([outcome_pain_sens.min(), outcome_pain_sens.max()],
            [outcome_pain_sens.min(), outcome_pain_sens.max()],
            'k--',
            lw=4)
    ax.set_xlabel('Measured')
    ax.set_ylabel(
        'Predicted by ' + kind)
    plt.show()

    from nilearn import plotting
    #######################################################################
    svr.fit(X_train_scaled, outcome_pain_sens)

    import numpy as np
    from nilearn.connectome import vec_to_sym_matrix


    m = vec_to_sym_matrix(svr.coef_, diagonal=np.repeat(0, len(mist_64_labels)))
    mist_64_labels[kind] = abs(m).max(axis=1)
    #mist_64_labels[kind] = m[55]
    #m = (abs(m) > (0.9) ) * m
    print("************")

    plotting.plot_connectome(m, mist_64_labels[['x', 'y', 'z']], display_mode='lzry',
                             title="%s %s" % ("mean connectivity", kind), colorbar=True)
    plotting.show()

    plotting.plot_matrix(m, figure=(15, 15),   labels=mist_64_labels['label'], title=kind, grid=True)
    plotting.show()

    #m=vec_to_sym_matrix(svr.coef_, diagonal=np.repeat(0, 64))
    #print mist_64_labels['label']

    ######################################################################

import numpy as np
plt.figure(figsize=(6, 4))
positions = np.arange(len(measures)) * .1 + .1
plt.barh(positions, mean_scores, align='center', height=.05)
yticks = [kind.replace(' ', '\n') for kind in measures]
plt.yticks(positions, yticks)
plt.xlabel('Classification accuracy')
plt.grid(True)
plt.tight_layout()

plt.show()

# cross validated parameter
# svr_cv = GridSearchCV(LinearSVR(),
#                      param_grid={'C': [.1, .5, 1., 5., 10., 50., 100.]},
#                      scoring='f1', n_jobs=8)

# The ridge classifier has a specific 'CV' object that can set it's
# parameters faster than using a GridSearchCV
# ridge = RidgeClassifier()
# ridge_cv = RidgeClassifierCV()

# Run
# Make a data splitting object for cross validation
