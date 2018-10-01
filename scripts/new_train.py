#!/usr/bin/env python
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
import numpy as np


import PAINTeR.load_project_data as load
import PAINTeR.train as train
import PAINTeR.models as models
from sklearn.metrics import r2_score, explained_variance_score, mean_squared_error
from sklearn import preprocessing
import nilearn.plotting as plotting
from nilearn.connectome import vec_to_sym_matrix


###################################################################################################
# timeseries file lists
###################################################################################################
bochum = ["/Users/tspisak/res/PAINTeR/bochum/regional_timeseries/_extract_timeseries0timeseries.tsv",
    "/Users/tspisak/res/PAINTeR/bochum/regional_timeseries/_extract_timeseries1timeseries.tsv",
    "/Users/tspisak/res/PAINTeR/bochum/regional_timeseries/_extract_timeseries2timeseries.tsv",
    "/Users/tspisak/res/PAINTeR/bochum/regional_timeseries/_extract_timeseries3timeseries.tsv",
    "/Users/tspisak/res/PAINTeR/bochum/regional_timeseries/_extract_timeseries4timeseries.tsv",
    "/Users/tspisak/res/PAINTeR/bochum/regional_timeseries/_extract_timeseries5timeseries.tsv",
    "/Users/tspisak/res/PAINTeR/bochum/regional_timeseries/_extract_timeseries6timeseries.tsv",
    "/Users/tspisak/res/PAINTeR/bochum/regional_timeseries/_extract_timeseries7timeseries.tsv",
    "/Users/tspisak/res/PAINTeR/bochum/regional_timeseries/_extract_timeseries8timeseries.tsv",
    "/Users/tspisak/res/PAINTeR/bochum/regional_timeseries/_extract_timeseries9timeseries.tsv",
    "/Users/tspisak/res/PAINTeR/bochum/regional_timeseries/_extract_timeseries10timeseries.tsv",
    "/Users/tspisak/res/PAINTeR/bochum/regional_timeseries/_extract_timeseries11timeseries.tsv",
    "/Users/tspisak/res/PAINTeR/bochum/regional_timeseries/_extract_timeseries12timeseries.tsv",
    "/Users/tspisak/res/PAINTeR/bochum/regional_timeseries/_extract_timeseries13timeseries.tsv",
    "/Users/tspisak/res/PAINTeR/bochum/regional_timeseries/_extract_timeseries14timeseries.tsv",
    "/Users/tspisak/res/PAINTeR/bochum/regional_timeseries/_extract_timeseries15timeseries.tsv",
    "/Users/tspisak/res/PAINTeR/bochum/regional_timeseries/_extract_timeseries16timeseries.tsv",
    "/Users/tspisak/res/PAINTeR/bochum/regional_timeseries/_extract_timeseries17timeseries.tsv",
    "/Users/tspisak/res/PAINTeR/bochum/regional_timeseries/_extract_timeseries18timeseries.tsv",
    "/Users/tspisak/res/PAINTeR/bochum/regional_timeseries/_extract_timeseries19timeseries.tsv",
    "/Users/tspisak/res/PAINTeR/bochum/regional_timeseries/_extract_timeseries20timeseries.tsv",
    "/Users/tspisak/res/PAINTeR/bochum/regional_timeseries/_extract_timeseries21timeseries.tsv",
    "/Users/tspisak/res/PAINTeR/bochum/regional_timeseries/_extract_timeseries22timeseries.tsv",
    "/Users/tspisak/res/PAINTeR/bochum/regional_timeseries/_extract_timeseries23timeseries.tsv",
    "/Users/tspisak/res/PAINTeR/bochum/regional_timeseries/_extract_timeseries24timeseries.tsv",
    "/Users/tspisak/res/PAINTeR/bochum/regional_timeseries/_extract_timeseries25timeseries.tsv",
    "/Users/tspisak/res/PAINTeR/bochum/regional_timeseries/_extract_timeseries26timeseries.tsv",
    "/Users/tspisak/res/PAINTeR/bochum/regional_timeseries/_extract_timeseries27timeseries.tsv",
    "/Users/tspisak/res/PAINTeR/bochum/regional_timeseries/_extract_timeseries28timeseries.tsv",
    "/Users/tspisak/res/PAINTeR/bochum/regional_timeseries/_extract_timeseries29timeseries.tsv",
    "/Users/tspisak/res/PAINTeR/bochum/regional_timeseries/_extract_timeseries30timeseries.tsv",
    "/Users/tspisak/res/PAINTeR/bochum/regional_timeseries/_extract_timeseries31timeseries.tsv",
    "/Users/tspisak/res/PAINTeR/bochum/regional_timeseries/_extract_timeseries32timeseries.tsv",
    "/Users/tspisak/res/PAINTeR/bochum/regional_timeseries/_extract_timeseries33timeseries.tsv",
    "/Users/tspisak/res/PAINTeR/bochum/regional_timeseries/_extract_timeseries34timeseries.tsv",
    "/Users/tspisak/res/PAINTeR/bochum/regional_timeseries/_extract_timeseries35timeseries.tsv",
    "/Users/tspisak/res/PAINTeR/bochum/regional_timeseries/_extract_timeseries36timeseries.tsv",
    "/Users/tspisak/res/PAINTeR/bochum/regional_timeseries/_extract_timeseries37timeseries.tsv",
    "/Users/tspisak/res/PAINTeR/bochum/regional_timeseries/_extract_timeseries38timeseries.tsv",
    "/Users/tspisak/res/PAINTeR/bochum/regional_timeseries/_extract_timeseries39timeseries.tsv",
    "/Users/tspisak/res/PAINTeR/bochum/regional_timeseries/_extract_timeseries40timeseries.tsv"]

###################################################################################################

bochum_ts = load.load_timeseries_tsv(bochum, standardise=True)

#from sklearn.preprocessing import StandardScaler
#bochum_ts = StandardScaler().fit_transform(np.transpose(bochum_ts))


X_bochum, cm_bochum = load.compute_connectivity(bochum_ts, kind="partial correlation", discard_diagonal=True)

mat=cm_bochum.mean_
mat[range(mat.shape[0]), range(mat.shape[0])] = 0
#mat[mat<-0.5]=-0.5
#mat[mat>0.5]=0.5
plotting.plot_matrix(mat)
plotting.show()

y_bochum = load.load_QST_data(data_filename="/Volumes/Bingel_Mac/PAINTeR/std4D/bochum_painthr.csv",
                             target_var= "mean_QST_pain_sensitivity",
                              exclude_ids=[1,2,3, 21, 22]) #40, 8???

y_all = load.load_QST_data_all(data_filename="/Volumes/Bingel_Mac/PAINTeR/std4D/bochum+essen_painthr.csv",
                              exclude_ids=["1","2","3", "21", "22", "PAINTeR_46", "PAINTeR_47", "PAINTeR_48", "PAINTeR_49"]) #40, 8???

y_bochum = y_all[y_all["sample"] == "BOCHUM"]
y_bochum = y_bochum["pain_sensitivity.lib"].values.ravel()

X_bochum = X_bochum[~np.isnan(y_bochum), :]
y_bochum = y_bochum[~np.isnan(y_bochum)]

mymodel, p_grid = models.pipe_scale_fsel_model(scaler=preprocessing.StandardScaler())
p_grid = {'fsel__k': [40, 45, 50], 'model__alpha': [.001, .005, .01], 'model__l1_ratio': [ .99999]}
#p_grid = {'fsel__k': [35, 45, 60, 70, 80, 100], 'model__alpha': [.0001, .001, .005, .01, .02, .05], 'model__l1_ratio': [ .999999, .9999999]}
#p_grid = {'fsel__k': [350, 400, 450, 500], 'model__alpha': [.0001, .001, .005], 'model__l1_ratio': [.01, .1, .2]}
m, avg_model = train.train(X_bochum, y_bochum, mymodel, p_grid, "Bochum", nested=True)

RES, mat, labels = train.get_full_coef(X_bochum, m)

mat_avg = vec_to_sym_matrix(avg_model, diagonal=np.repeat(0, len(labels)+1))
plotting.plot_matrix(mat_avg, figure=(10, 10), labels=['GS']+labels['modules'].values.tolist(), title="", grid=True)  #
plotting.show()