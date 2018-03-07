from sklearn.externals import joblib
import numpy as np
import pandas as pd
from nilearn.connectome import vec_to_sym_matrix
from nilearn import plotting



############################
# Load atlas
############################
mist_64_labels_filename = '~/data/atlases/MIST/Parcel_Information/MIST_122.csv'
mist_64_labels = pd.read_csv(mist_64_labels_filename, sep=";")


mist_hierarchy_filename = '/Users/tspisak/data/atlases/MIST/Hierarchy/MIST_PARCEL_ORDER_ROI.csv'
mist_hierarchy = pd.read_csv(mist_hierarchy_filename, sep=",")
mist_64_hierarchy = mist_hierarchy['s122']
mist_hierarchy = pd.read_csv(mist_hierarchy_filename, sep=",")
mist_7_hierarchy = mist_hierarchy['s7']
mist_64_hierarchy = mist_64_hierarchy.drop_duplicates()
mist_7_hierarchy = mist_7_hierarchy[mist_64_hierarchy.index]



mist_64_labels = mist_64_labels.reindex(mist_64_hierarchy - 1)
mist_64_labels=mist_64_labels.reset_index()

############################
# Load data
############################
target = ['mean_QST_pain_sensitivity']
opt_k = 35

connectivity_biomarkers = joblib.load("connectivity_biomarkers_s122_friston.sav")
kind = "tangent"
#kind = "correlation"
X_orig = connectivity_biomarkers[kind]

X_orig=np.arctanh(X_orig)

data_filename = '/Users/tspisak/projects/Mercure_rest/data/data.csv'
data = pd.read_csv(data_filename, sep=",")  # .values.ravel()

print list(data)
print target

data = data[3:]  # exclude pilot data
data = data[data["ID"] != 8]  # exclude subj 8
data = data[data["ID"] != 21]  # exclude subj 21
data = data[data["ID"] != 22]  # exclude subj 22
data = data[data["ID"] != 40]  # exclude subj 22

outcome = data[target].values.ravel()
print outcome.shape

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
X = X_orig
Y = Y_orig
import matplotlib.pyplot as plt


def model(k):
    ####################################################################################################
    # SVR
    ####################################################################################################
    from sklearn.feature_selection import SelectKBest, f_regression
    from sklearn.svm import SVR, LinearSVR
    from sklearn.linear_model import LassoCV, ElasticNetCV
    # loo = LeaveOneOut(len(outcome_pain_sens))
    # Define the dimension reduction to be used.
    # Here we use a classical univariate feature selection based on F-test,
    # We set the number of features to be selected to k
    feature_selection = SelectKBest(f_regression, k=k)
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.grid_search import GridSearchCV
    # svr = LinearSVR(loss='hinge', intercept_scaling=1000)
    svreg = LinearSVR(fit_intercept=True, C=1)
    svreg = SVR()
    svreg = GridSearchCV(SVR(),
                         param_grid={'C': [.1, .5, 1., 5., 10., 50., 100.]},
                         scoring='neg_mean_squared_error', n_jobs=1)
    svreg = ElasticNetCV(cv=20, max_iter=10000, fit_intercept=True)
    # build pipeline
    from sklearn.pipeline import Pipeline
    scaler = preprocessing.StandardScaler()
    svr = Pipeline([('scaler', scaler), ('fsel', feature_selection), ('elastic', svreg)])
    # svr = Pipeline([('fsel', feature_selection), ('svr', svreg)])

    # svr.fit(X_scaled, Y)
    return svr

mod = model(k=opt_k)
fit = mod.fit(X, Y)
print fit.predict(X)




print mod.named_steps['elastic'].coef_
featuremask = mod.named_steps['fsel'].get_support()
RES = np.zeros(X[1].shape)
RES_bin = np.zeros(X[1].shape)
RES[featuremask] = mod.named_steps['elastic'].coef_
RES_bin[featuremask] = np.ones(len(mod.named_steps['elastic'].coef_))
m = vec_to_sym_matrix(RES, diagonal=np.repeat(0, len(mist_64_labels)))
m_sign = m/abs(m)
m_bin = vec_to_sym_matrix(RES_bin, diagonal=np.repeat(0, len(mist_64_labels)))
print "itt"
plotting.plot_matrix(m_sign, figure=(15, 15),   labels=mist_64_labels['label'], title="", grid=True)
plotting.show()
print "itt"
from matplotlib import colors as mcolors
plotting.plot_connectome(m, mist_64_labels[['x', 'y', 'z']], display_mode='lzry',
                             colorbar=True)
print "itt"
plotting.show()
print "itt"

nodes = mist_64_labels[['x', 'y', 'z']]  # 'label']]
print type(mist_7_hierarchy)
nodes['col'] = mist_7_hierarchy.values
nodes['strength_bin'] = abs(m_bin).sum(axis=1)
nodes['strength'] = abs(m).sum(axis=1)
#nodes = nodes.append(z)
nodes['label'] = mist_64_labels['label']

nodes.to_csv("MIST_122.nodes", sep=" ", header=False, index=False)
np.savetxt("mean_pain_sensitivity_tangent_elastic_net.edge", m, delimiter=" ")

nodes = nodes[nodes['strength']>0]
nodes.to_csv("nodes.txt", sep=" ", header=False, index=False)
nod=nodes
#nodes = nodes.sort_values(by='strength', ascending=False)

y_pos = np.arange(len(nodes['label']))

# Create bars
plt.bar(y_pos, nodes['strength'])
# Create names on the x-axis
plt.xticks(y_pos, nodes['label'], rotation=90)
plt.subplots_adjust(bottom=0.2, top=0.99)
# Show graphic
plt.show()

#print np.sort(mod.named_steps['fsel'].scores_)[-35:]
#plt.hist(mod.named_steps['fsel'].scores_)
#plt.title("Histogram of F-scores")
#plt.show()

KBestIndex = np.argsort(mod.named_steps['fsel'].scores_)[-35:]

print "table"

# Get the integer indices of the rows that sum up to 3
# and the columns that sum up to 3.
good_rows = np.nonzero(m.sum(axis=1) != 0)
bad_rows = np.nonzero(m.sum(axis=1) == 0)
bad_cols = np.nonzero(m.sum(axis=0) == 0)

# Now use the numpy.delete() function to get the matrix
# with those rows and columns removed from the original matrix.
P = np.delete(m, bad_rows, axis=0)
P = np.delete(P, bad_cols, axis=1)

print P.shape

from matplotlib import colors as mcolors
plotting.plot_connectome(P, nod[['x', 'y', 'z']], display_mode='lzry',
                             colorbar=True)
print "itt"
plotting.show()

np.savetxt("test.txt", P, delimiter=" ")
print nodes['label'].values

#P=P/abs(P)
plotting.plot_matrix(P, figure=(15, 15),   labels=nod['label'], title="", grid=True, vmin=-0.18, vmax=0.18 )
plotting.show()
print "itt"


