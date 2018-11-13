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
bochum = [
    "/Users/tspisak/res/PAINTeR/bochum/regional_timeseries/_extract_timeseries0timeseries.tsv",  # 004
    #"/Users/tspisak/res/PAINTeR/bochum/regional_timeseries/_extract_timeseries1timeseries.tsv",  # 005     %FD=31   ???
    "/Users/tspisak/res/PAINTeR/bochum/regional_timeseries/_extract_timeseries2timeseries.tsv",  # 006
    "/Users/tspisak/res/PAINTeR/bochum/regional_timeseries/_extract_timeseries3timeseries.tsv",  # 007
    "/Users/tspisak/res/PAINTeR/bochum/regional_timeseries/_extract_timeseries4timeseries.tsv",  # 008
    "/Users/tspisak/res/PAINTeR/bochum/regional_timeseries/_extract_timeseries5timeseries.tsv",  # 009
    "/Users/tspisak/res/PAINTeR/bochum/regional_timeseries/_extract_timeseries6timeseries.tsv",  # 010
    "/Users/tspisak/res/PAINTeR/bochum/regional_timeseries/_extract_timeseries7timeseries.tsv",  # 011
    "/Users/tspisak/res/PAINTeR/bochum/regional_timeseries/_extract_timeseries8timeseries.tsv",  # 012
    "/Users/tspisak/res/PAINTeR/bochum/regional_timeseries/_extract_timeseries9timeseries.tsv",  # 013
    # "/Users/tspisak/res/PAINTeR/bochum/regional_timeseries/_extract_timeseries10timeseries.tsv",  # 014   mFD=0.21, %FD=42
    "/Users/tspisak/res/PAINTeR/bochum/regional_timeseries/_extract_timeseries11timeseries.tsv",  # 015
    "/Users/tspisak/res/PAINTeR/bochum/regional_timeseries/_extract_timeseries12timeseries.tsv",  # 016
    "/Users/tspisak/res/PAINTeR/bochum/regional_timeseries/_extract_timeseries13timeseries.tsv",  # 017
    "/Users/tspisak/res/PAINTeR/bochum/regional_timeseries/_extract_timeseries14timeseries.tsv",  # 018
    "/Users/tspisak/res/PAINTeR/bochum/regional_timeseries/_extract_timeseries15timeseries.tsv",  # 019
    "/Users/tspisak/res/PAINTeR/bochum/regional_timeseries/_extract_timeseries16timeseries.tsv",  # 020
    "/Users/tspisak/res/PAINTeR/bochum/regional_timeseries/_extract_timeseries17timeseries.tsv",  # 023
    "/Users/tspisak/res/PAINTeR/bochum/regional_timeseries/_extract_timeseries18timeseries.tsv",  # 024
    "/Users/tspisak/res/PAINTeR/bochum/regional_timeseries/_extract_timeseries19timeseries.tsv",  # 025
    "/Users/tspisak/res/PAINTeR/bochum/regional_timeseries/_extract_timeseries20timeseries.tsv",  # 026
    "/Users/tspisak/res/PAINTeR/bochum/regional_timeseries/_extract_timeseries21timeseries.tsv",  # 027
    "/Users/tspisak/res/PAINTeR/bochum/regional_timeseries/_extract_timeseries22timeseries.tsv",  # 028
    # "/Users/tspisak/res/PAINTeR/bochum/regional_timeseries/_extract_timeseries23timeseries.tsv",  # 029   mFD=0.2, %FD=32
    "/Users/tspisak/res/PAINTeR/bochum/regional_timeseries/_extract_timeseries24timeseries.tsv",  # 030
    "/Users/tspisak/res/PAINTeR/bochum/regional_timeseries/_extract_timeseries25timeseries.tsv",  # 031
    "/Users/tspisak/res/PAINTeR/bochum/regional_timeseries/_extract_timeseries26timeseries.tsv",  # 032
    "/Users/tspisak/res/PAINTeR/bochum/regional_timeseries/_extract_timeseries27timeseries.tsv",  # 033
    # "/Users/tspisak/res/PAINTeR/bochum/regional_timeseries/_extract_timeseries28timeseries.tsv",  # 034   mFD=0.25
    "/Users/tspisak/res/PAINTeR/bochum/regional_timeseries/_extract_timeseries29timeseries.tsv",  # 035
    "/Users/tspisak/res/PAINTeR/bochum/regional_timeseries/_extract_timeseries30timeseries.tsv",  # 036
    "/Users/tspisak/res/PAINTeR/bochum/regional_timeseries/_extract_timeseries31timeseries.tsv",  # 037
    "/Users/tspisak/res/PAINTeR/bochum/regional_timeseries/_extract_timeseries32timeseries.tsv",  # 038
    "/Users/tspisak/res/PAINTeR/bochum/regional_timeseries/_extract_timeseries33timeseries.tsv",  # 039
    # "/Users/tspisak/res/PAINTeR/bochum/regional_timeseries/_extract_timeseries34timeseries.tsv",  # 040   mFD=0.26, %FD=40
    "/Users/tspisak/res/PAINTeR/bochum/regional_timeseries/_extract_timeseries35timeseries.tsv",  # 041
    "/Users/tspisak/res/PAINTeR/bochum/regional_timeseries/_extract_timeseries36timeseries.tsv",  # 042
    "/Users/tspisak/res/PAINTeR/bochum/regional_timeseries/_extract_timeseries37timeseries.tsv",  # 043
    "/Users/tspisak/res/PAINTeR/bochum/regional_timeseries/_extract_timeseries38timeseries.tsv",  # 044 AFNI_OUTLIER ??
    "/Users/tspisak/res/PAINTeR/bochum/regional_timeseries/_extract_timeseries39timeseries.tsv",  # 045
    "/Users/tspisak/res/PAINTeR/bochum/regional_timeseries/_extract_timeseries40timeseries.tsv"   # 046
]

essen = [
"/Volumes/Elements/painter_res_essen_1mm_new1/regional_timeseries/_extract_timeseries0timeseries.tsv", #001
"/Volumes/Elements/painter_res_essen_1mm_new1/regional_timeseries/_extract_timeseries1timeseries.tsv", #002
"/Volumes/Elements/painter_res_essen_1mm_new1/regional_timeseries/_extract_timeseries2timeseries.tsv", #003
"/Volumes/Elements/painter_res_essen_1mm_new1/regional_timeseries/_extract_timeseries3timeseries.tsv", #004
"/Volumes/Elements/painter_res_essen_1mm_new1/regional_timeseries/_extract_timeseries4timeseries.tsv", #005
"/Volumes/Elements/painter_res_essen_1mm_new1/regional_timeseries/_extract_timeseries5timeseries.tsv", #006
"/Volumes/Elements/painter_res_essen_1mm_new1/regional_timeseries/_extract_timeseries6timeseries.tsv", #007
"/Volumes/Elements/painter_res_essen_1mm_new1/regional_timeseries/_extract_timeseries7timeseries.tsv", #008 # MAXFD HUGE!
"/Volumes/Elements/painter_res_essen_1mm_new1/regional_timeseries/_extract_timeseries8timeseries.tsv", #009 # MAXFD BIG!
"/Volumes/Elements/painter_res_essen_1mm_new1/regional_timeseries/_extract_timeseries9timeseries.tsv", #010
"/Volumes/Elements/painter_res_essen_1mm_new1/regional_timeseries/_extract_timeseries10timeseries.tsv", #011
"/Volumes/Elements/painter_res_essen_1mm_new1/regional_timeseries/_extract_timeseries11timeseries.tsv", #012
#"/Volumes/Elements/painter_res_essen_1mm_new1/regional_timeseries/_extract_timeseries12timeseries.tsv", #013   %FD=35
"/Volumes/Elements/painter_res_essen_1mm_new1/regional_timeseries/_extract_timeseries13timeseries.tsv", #014
"/Volumes/Elements/painter_res_essen_1mm_new1/regional_timeseries/_extract_timeseries14timeseries.tsv", #015
"/Volumes/Elements/painter_res_essen_1mm_new1/regional_timeseries/_extract_timeseries15timeseries.tsv", #016
"/Volumes/Elements/painter_res_essen_1mm_new1/regional_timeseries/_extract_timeseries16timeseries.tsv", #017
#"/Volumes/Elements/painter_res_essen_1mm_new1/regional_timeseries/_extract_timeseries17timeseries.tsv", #018   %FD=32
"/Volumes/Elements/painter_res_essen_1mm_new1/regional_timeseries/_extract_timeseries41timeseries.tsv", #019  # 41 instead of 18!!
"/Volumes/Elements/painter_res_essen_1mm_new1/regional_timeseries/_extract_timeseries19timeseries.tsv", #020
"/Volumes/Elements/painter_res_essen_1mm_new1/regional_timeseries/_extract_timeseries20timeseries.tsv", #021   GHOST!!
"/Volumes/Elements/painter_res_essen_1mm_new1/regional_timeseries/_extract_timeseries21timeseries.tsv", #022
"/Volumes/Elements/painter_res_essen_1mm_new1/regional_timeseries/_extract_timeseries22timeseries.tsv", #023
"/Volumes/Elements/painter_res_essen_1mm_new1/regional_timeseries/_extract_timeseries23timeseries.tsv", #024
#"/Volumes/Elements/painter_res_essen_1mm_new1/regional_timeseries/_extract_timeseries24timeseries.tsv", #025   mFD=0.21 %FD=36, maxFD
"/Volumes/Elements/painter_res_essen_1mm_new1/regional_timeseries/_extract_timeseries25timeseries.tsv", #026
"/Volumes/Elements/painter_res_essen_1mm_new1/regional_timeseries/_extract_timeseries26timeseries.tsv", #027   %FD=30% maxFD
"/Volumes/Elements/painter_res_essen_1mm_new1/regional_timeseries/_extract_timeseries27timeseries.tsv", #028
#"/Volumes/Elements/painter_res_essen_1mm_new1/regional_timeseries/_extract_timeseries28timeseries.tsv", #029   %FD=34%
"/Volumes/Elements/painter_res_essen_1mm_new1/regional_timeseries/_extract_timeseries29timeseries.tsv", #030
"/Volumes/Elements/painter_res_essen_1mm_new1/regional_timeseries/_extract_timeseries30timeseries.tsv", #031
"/Volumes/Elements/painter_res_essen_1mm_new1/regional_timeseries/_extract_timeseries31timeseries.tsv", #032
"/Volumes/Elements/painter_res_essen_1mm_new1/regional_timeseries/_extract_timeseries32timeseries.tsv", #033
"/Volumes/Elements/painter_res_essen_1mm_new1/regional_timeseries/_extract_timeseries33timeseries.tsv", #034
#"/Volumes/Elements/painter_res_essen_1mm_new1/regional_timeseries/_extract_timeseries34timeseries.tsv", #035   mFD=0.34 %FD=34, maxFD
"/Volumes/Elements/painter_res_essen_1mm_new1/regional_timeseries/_extract_timeseries35timeseries.tsv", #036
"/Volumes/Elements/painter_res_essen_1mm_new1/regional_timeseries/_extract_timeseries36timeseries.tsv", #037
"/Volumes/Elements/painter_res_essen_1mm_new1/regional_timeseries/_extract_timeseries37timeseries.tsv", #038
"/Volumes/Elements/painter_res_essen_1mm_new1/regional_timeseries/_extract_timeseries38timeseries.tsv", #039
"/Volumes/Elements/painter_res_essen_1mm_new1/regional_timeseries/_extract_timeseries39timeseries.tsv", #040
#"/Volumes/Elements/painter_res_essen_1mm_new1/regional_timeseries/_extract_timeseries40timeseries.tsv", #041 # ABDUL
"/Volumes/Elements/painter_res_essen_1mm_new1/regional_timeseries/_extract_timeseries41timeseries.tsv", #042
"/Volumes/Elements/painter_res_essen_1mm_new1/regional_timeseries/_extract_timeseries42timeseries.tsv", #043
"/Volumes/Elements/painter_res_essen_1mm_new1/regional_timeseries/_extract_timeseries43timeseries.tsv", #044
"/Volumes/Elements/painter_res_essen_1mm_new1/regional_timeseries/_extract_timeseries44timeseries.tsv",#045

"/Users/tspisak/projects/PAINTeR/res/essen/last5/regional_timeseries/_extract_timeseries0timeseries.tsv", #X45
"/Users/tspisak/projects/PAINTeR/res/essen/last5/regional_timeseries/_extract_timeseries1timeseries.tsv", #X46
#"/Users/tspisak/projects/PAINTeR/res/essen/last5/regional_timeseries/_extract_timeseries2timeseries.tsv", #X47 #mFD=0.31 %FD=47
#"/Users/tspisak/projects/PAINTeR/res/essen/last5/regional_timeseries/_extract_timeseries3timeseries.tsv", #X48
"/Users/tspisak/projects/PAINTeR/res/essen/last5/regional_timeseries/_extract_timeseries4timeseries.tsv" #X49

]

###################################################################################################

bochum_ts = load.load_timeseries_tsv(bochum, standardise=True)
essen_ts = load.load_timeseries_tsv(essen, standardise=True)

#from sklearn.preprocessing import StandardScaler
#bochum_ts = StandardScaler().fit_transform(np.transpose(bochum_ts))


X_bochum, cm_bochum = load.compute_connectivity(bochum_ts, kind="partial correlation", discard_diagonal=True)
X_essen, cm_essen = load.compute_connectivity(essen_ts, kind="partial correlation", discard_diagonal=True)

mat=cm_bochum.mean_
mat[range(mat.shape[0]), range(mat.shape[0])] = 0
#mat[mat<-0.5]=-0.5
#mat[mat>0.5]=0.5
plotting.plot_matrix(mat)
plotting.show()

mat=cm_essen.mean_
mat[range(mat.shape[0]), range(mat.shape[0])] = 0
#mat[mat<-0.5]=-0.5
#mat[mat>0.5]=0.5
plotting.plot_matrix(mat)
plotting.show()

y_bochum = load.load_QST_data_all(data_filename="/Users/tspisak/data/PAINTeR/bingel_drive/bochum_painthr.csv",
                              exclude_ids=[1,2,3, 21, 22,   5, 14, 29, 34, 40])["mean_QST_pain_sensitivity"].values.ravel()

y_essen = load.load_QST_data_all(data_filename="/Users/tspisak/data/PAINTeR/bingel_drive/essen_painthr_fixed.csv",
                              idvar="pumi.ID",
                           exclude_ids= ["X47", "X48"] +
                                          #+ ["P08", "P09"] +
                                          ["P13", "P18", "P25", "P29", "P35"] #motion excludes
                                            #+ ["PAINTeR_16"]
                                 )

y_essen_id = y_essen["pumi.ID"].values.ravel()
y_essen_CPT =  y_essen["coldthr"].values.ravel()
print y_essen_id
y_essen = y_essen["compositepainsensitivity"].values.ravel()

#y_all = load.load_QST_data_all(data_filename="/Volumes/Bingel_Mac/PAINTeR/std4D/bochum+essen_painthr.csv",
#                              exclude_ids=["1","2","3", "21", "22"] + ["14", "29", "34", "40"] +
#                                            ["5"] +
#                                          ["PAINTeR_46", "PAINTeR_47", "PAINTeR_48", "PAINTeR_49"] +
#                                           ["PAINTeR_08", "PAINTeR_09"] +
#                                          ["PAINTeR_13", "PAINTeR_18", "PAINTeR_21", "PAINTeR_25", "PAINTeR_29", "PAINTeR_35"]
#                               ) #40, 8???

#y_bochum = y_all[y_all["sample"] == "BOCHUM"]
#y_bochum = y_bochum["pain_sensitivity.lib"].values.ravel()

#y_essen = y_all[y_all["sample"] == "ESSEN"]
#y_essen_id = y_essen["ID"].values.ravel()
#y_essen = y_essen["pain_sensitivity.lib"].values.ravel()


X_bochum = X_bochum[~np.isnan(y_bochum), :]
y_bochum = y_bochum[~np.isnan(y_bochum)]

X_essen = X_essen[~np.isnan(y_essen), :]
y_essen_id = y_essen_id[~np.isnan(y_essen)]
y_essen_CPT = y_essen_CPT[~np.isnan(y_essen)]
y_essen = y_essen[~np.isnan(y_essen)]
# filter low-end
X_essen = X_essen[y_essen_CPT > 5, :]
y_essen_id = y_essen_id[y_essen_CPT > 5]
y_essen = y_essen[y_essen_CPT > 5]


mymodel, p_grid = models.pipe_scale_fsel_model(scaler=preprocessing.MinMaxScaler())
p_grid = {'fsel__k': [10, 20, 30], 'model__alpha': [.000001, .00005], 'model__l1_ratio': [.999999999]}
p_grid = {'fsel__k': [10, 20, 30], 'model__alpha': [.001], 'model__l1_ratio': [.999999999]}
#p_grid = {'fsel__k': [19, 20, 21], 'model__alpha': [ .01], 'model__l1_ratio': [.9999999]}
#p_grid = {'fsel__k': [60, 70, 80, 90, 100, 110, 120, 130, 140], 'model__alpha': [.0001, .001, .005, .01, .02, .05], 'model__l1_ratio': [0.1, 0.3, 0.5, 0.7, .9]}
#p_grid = {'fsel__k': [350, 400, 450, 500], 'model__alpha': [.0001, .001, .005], 'model__l1_ratio': [.01, .1, .2]}


print "*********************************************************************************************************"

m, avg_model = train.train(X_bochum, y_bochum, mymodel, p_grid, "Bochum", nested=False)

RES, mat, labels = train.get_full_coef(X_bochum, m)

#mat_avg = vec_to_sym_matrix(avg_model, diagonal=np.repeat(0, len(labels)+1))
#plotting.plot_matrix(mat_avg, figure=(10, 10), labels=['GS']+labels['modules'].values.tolist(), title="", grid=True)  #
#plotting.show()

pred_essen = m.predict(X_essen)
#pred_essen=pred_essen-np.mean(pred_essen)#/np.std(pred_essen)
#y_essen=y_essen-np.mean(y_essen)#/np.std(y_essen)
#pred_essen=preprocessing.scale(pred_essen)
#y_essen=preprocessing.scale(y_essen)

print "Expl. Var.:" + str( 1- (-mean_squared_error(y_pred=pred_essen, y_true=y_essen))/-mean_squared_error(np.repeat(y_bochum.mean(), len(y_essen)), y_essen) )
print "Correlation: " + str(np.corrcoef(pred_essen, y_essen)[0,1])


import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.scatter(y_essen, pred_essen, edgecolors=(0, 0, 0))
ax.plot([y_essen.min(), y_essen.max()],
                   [y_essen.min(), y_essen.max()],
                   'k--',
                   lw=2)
ax.set_xlabel('Pain Sensitivity (trained on Bochum sample, predicted on Essen sample)')
ax.set_ylabel('Predicted')

for i, txt in enumerate(y_essen_id):
    ax.annotate(txt, (y_essen[i], pred_essen[i]))
plt.title( "Expl. Var.:" + str( 1- (-mean_squared_error(y_pred=pred_essen, y_true=y_essen))/-mean_squared_error(np.repeat(y_bochum.mean(), len(y_essen)), y_essen) ) +
    "\nCorrelation: " + str(np.corrcoef(pred_essen, y_essen)[0,1]))
plt.show()


print "*********************************************************************************************************"

p_grid = {'fsel__k': [20, 30, 40, 50, 60], 'model__alpha': [.005, .01, .02], 'model__l1_ratio': [.9, .9999999]}
#p_grid = {'fsel__k': [60, 70, 80, 90, 100, 110, 120, 130, 140], 'model__alpha': [.0001, .001, .005, .01, .02, .05], 'model__l1_ratio': [0.1, 0.3, 0.5, 0.7, .9]}
#p_grid = {'fsel__k': [350, 400, 450, 500], 'model__alpha': [.0001, .001, .005], 'model__l1_ratio': [.01, .1, .2]}
m, avg_model = train.train(X_essen, y_essen, mymodel, p_grid, "Essen", nested=True)

RES, mat, labels = train.get_full_coef(X_essen, m)

#mat_avg = vec_to_sym_matrix(avg_model, diagonal=np.repeat(0, len(labels)+1))
#plotting.plot_matrix(mat_avg, figure=(10, 10), labels=['GS']+labels['modules'].values.tolist(), title="", grid=True)  #
#plotting.show()


