#!/usr/bin/env python
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.externals import joblib
from PAINTeR import global_vars
from PAINTeR import model
import seaborn as sns


# load trained model
mod = joblib.load(global_vars._RES_PRED_MOD_)

#######################################
# learning curve BOCHUM sample     #
#######################################
# load pain sensitivity data (excluded)
y = pd.read_csv(global_vars._RES_BOCHUM_TABLE_EXCL_)['mean_QST_pain_sensitivity']
X = joblib.load(global_vars._FEATURE_BOCHUM_)

Ns = [20, 22, 24, 26, 28, 30, 32, 34, 35]
# do calculation
#train, test = model.learning_curve(mod, X, y.tolist(), Ns=Ns)
#data = [train, test]
#joblib.dump(data, "learningcurve.sav")

data = joblib.load("learningcurve.sav")

#print len(range(20, len(y)+1))
#print len(np.sqrt(np.abs(data[1]))[29:])

plt.figure(figsize=(6,2))
sns.lineplot(Ns, np.abs(data[0]))
grid = sns.lineplot(Ns, np.abs(data[1]))
grid.set(yscale="log")

figure = plt.gcf()
figure.savefig(global_vars._PLOT_LEARNING_CURVE_, bbox_inches='tight')
plt.close(figure)

plt.show()

##############################################################
# specificitiy calculations plots                           #
##############################################################

table_bochum = pd.read_csv(global_vars._RES_BOCHUM_TABLE_EXCL_)
table_essen = pd.read_csv(global_vars._RES_ESSEN_TABLE_EXCL_)
table_szeged = pd.read_csv(global_vars._RES_SZEGED_TABLE_EXCL_)

var = "meanFD"
print var
p_value, r_2, residual, regline =  model.pred_stat(table_bochum[var], table_bochum['prediction'])
print "{:.3f}".format(r_2) + "\t" + "{:.3f}".format(p_value)
p_value, r_2, residual, regline =  model.pred_stat(table_essen[var], table_essen['prediction'])
print "{:.3f}".format(r_2) + "\t" + "{:.3f}".format(p_value)
p_value, r_2, residual, regline =  model.pred_stat(table_szeged[var], table_szeged['prediction'])
print "{:.3f}".format(r_2) + "\t" + "{:.3f}".format(p_value)
var = "medianFD"
print var
p_value, r_2, residual, regline =  model.pred_stat(table_bochum[var], table_bochum['prediction'])
print "{:.3f}".format(r_2) + "\t" + "{:.3f}".format(p_value)
p_value, r_2, residual, regline =  model.pred_stat(table_essen[var], table_essen['prediction'])
print "{:.3f}".format(r_2) + "\t" + "{:.3f}".format(p_value)
p_value, r_2, residual, regline =  model.pred_stat(table_szeged[var], table_szeged['prediction'])
print "{:.3f}".format(r_2) + "\t" + "{:.3f}".format(p_value)
var = "maxFD"
print var
p_value, r_2, residual, regline =  model.pred_stat(table_bochum[var], table_bochum['prediction'])
print "{:.3f}".format(r_2) + "\t" + "{:.3f}".format(p_value)
p_value, r_2, residual, regline =  model.pred_stat(table_essen[var], table_essen['prediction'])
print "{:.3f}".format(r_2) + "\t" + "{:.3f}".format(p_value)
p_value, r_2, residual, regline =  model.pred_stat(table_szeged[var], table_szeged['prediction'])
print "{:.3f}".format(r_2) + "\t" + "{:.3f}".format(p_value)
var = "perc_scrubbed"
print var
p_value, r_2, residual, regline =  model.pred_stat(table_bochum[var], table_bochum['prediction'])
print "{:.3f}".format(r_2) + "\t" + "{:.3f}".format(p_value)
p_value, r_2, residual, regline =  model.pred_stat(table_essen[var], table_essen['prediction'])
print "{:.3f}".format(r_2) + "\t" + "{:.3f}".format(p_value)
p_value, r_2, residual, regline =  model.pred_stat(table_szeged[var], table_szeged['prediction'])
print "{:.3f}".format(r_2) + "\t" + "{:.3f}".format(p_value)
var = "Age"
print var
p_value, r_2, residual, regline =  model.pred_stat(table_bochum[var], table_bochum['prediction'])
print "{:.3f}".format(r_2) + "\t" + "{:.3f}".format(p_value)
p_value, r_2, residual, regline =  model.pred_stat(table_essen[var], table_essen['prediction'])
print "{:.3f}".format(r_2) + "\t" + "{:.3f}".format(p_value)
p_value, r_2, residual, regline =  model.pred_stat(table_szeged[var], table_szeged['prediction'])
print "{:.3f}".format(r_2) + "\t" + "{:.3f}".format(p_value)
var = "Male"
print var
p_value, r_2, residual, regline =  model.pred_stat(table_bochum[var], table_bochum['prediction'])
print "{:.3f}".format(r_2) + "\t" + "{:.3f}".format(p_value)
p_value, r_2, residual, regline =  model.pred_stat(table_essen[var], table_essen['prediction'])
print "{:.3f}".format(r_2) + "\t" + "{:.3f}".format(p_value)
p_value, r_2, residual, regline =  model.pred_stat(table_szeged[var], table_szeged['prediction'])
print "{:.3f}".format(r_2) + "\t" + "{:.3f}".format(p_value)
var = "alk_per_w"
print var
p_value, r_2, residual, regline =  model.pred_stat(table_bochum[var], table_bochum['prediction'])
print "{:.3f}".format(r_2) + "\t" + "{:.3f}".format(p_value)
p_value, r_2, residual, regline =  model.pred_stat(table_essen[var], table_essen['prediction'])
print "{:.3f}".format(r_2) + "\t" + "{:.3f}".format(p_value)
#p_value, r_2, residual, regline =  model.pred_stat(table_szeged[var], table_szeged['prediction'])
#print "{:.3f}".format(r_2) + "\t" + "{:.3f}".format(p_value)
print "NA\tNA"
var = "day_menses"
print var
p_value, r_2, residual, regline =  model.pred_stat(table_bochum[var], table_bochum['prediction'])
print "{:.3f}".format(r_2) + "\t" + "{:.3f}".format(p_value)
p_value, r_2, residual, regline =  model.pred_stat(table_essen[var], table_essen['prediction'])
print "{:.3f}".format(r_2) + "\t" + "{:.3f}".format(p_value)
p_value, r_2, residual, regline =  model.pred_stat(table_szeged[var], table_szeged['prediction'])
print "{:.3f}".format(r_2) + "\t" + "{:.3f}".format(p_value)
var = "edu"
print var
p_value, r_2, residual, regline =  model.pred_stat(table_bochum[var], table_bochum['prediction'])
print "{:.3f}".format(r_2) + "\t" + "{:.3f}".format(p_value)
p_value, r_2, residual, regline =  model.pred_stat(table_essen[var], table_essen['prediction'])
print "{:.3f}".format(r_2) + "\t" + "{:.3f}".format(p_value)
#p_value, r_2, residual, regline =  model.pred_stat(table_szeged[var], table_szeged['prediction'])
#print "{:.3f}".format(r_2) + "\t" + "{:.3f}".format(p_value)
print "NA\tNA"
var = "BMI"
print var
p_value, r_2, residual, regline =  model.pred_stat(table_bochum[var], table_bochum['prediction'])
print "{:.3f}".format(r_2) + "\t" + "{:.3f}".format(p_value)
p_value, r_2, residual, regline =  model.pred_stat(table_essen[var], table_essen['prediction'])
print "{:.3f}".format(r_2) + "\t" + "{:.3f}".format(p_value)
p_value, r_2, residual, regline =  model.pred_stat(table_szeged[var], table_szeged['prediction'])
print "{:.3f}".format(r_2) + "\t" + "{:.3f}".format(p_value)

var = "BP_MRI_sys"
print var
#p_value, r_2, residual, regline =  model.pred_stat(table_bochum[var], table_bochum['prediction'])
#print "{:.3f}".format(r_2) + "\t" + "{:.3f}".format(p_value)
print "NA\tNA"
p_value, r_2, residual, regline =  model.pred_stat(table_essen[var], table_essen['prediction'])
print "{:.3f}".format(r_2) + "\t" + "{:.3f}".format(p_value)
p_value, r_2, residual, regline =  model.pred_stat(table_szeged[var], table_szeged['prediction'])
print "{:.3f}".format(r_2) + "\t" + "{:.3f}".format(p_value)

var = "BP_MRI_dias"
print var
#p_value, r_2, residual, regline =  model.pred_stat(table_bochum[var], table_bochum['prediction'])
#print "{:.3f}".format(r_2) + "\t" + "{:.3f}".format(p_value)
print "NA\tNA"
p_value, r_2, residual, regline =  model.pred_stat(table_essen[var], table_essen['prediction'])
print "{:.3f}".format(r_2) + "\t" + "{:.3f}".format(p_value)
p_value, r_2, residual, regline =  model.pred_stat(table_szeged[var], table_szeged['prediction'])
print "{:.3f}".format(r_2) + "\t" + "{:.3f}".format(p_value)

var = "BP_QST_sys"
print var
#p_value, r_2, residual, regline =  model.pred_stat(table_bochum[var], table_bochum['prediction'])
#print "{:.3f}".format(r_2) + "\t" + "{:.3f}".format(p_value)
print "NA\tNA"
p_value, r_2, residual, regline =  model.pred_stat(table_essen[var], table_essen['prediction'])
print "{:.3f}".format(r_2) + "\t" + "{:.3f}".format(p_value)
p_value, r_2, residual, regline =  model.pred_stat(table_szeged[var], table_szeged['prediction'])
print "{:.3f}".format(r_2) + "\t" + "{:.3f}".format(p_value)

var = "BP_QST_dias"
print var
#p_value, r_2, residual, regline =  model.pred_stat(table_bochum[var], table_bochum['prediction'])
#print "{:.3f}".format(r_2) + "\t" + "{:.3f}".format(p_value)
print "NA\tNA"
p_value, r_2, residual, regline =  model.pred_stat(table_essen[var], table_essen['prediction'])
print "{:.3f}".format(r_2) + "\t" + "{:.3f}".format(p_value)
p_value, r_2, residual, regline =  model.pred_stat(table_szeged[var], table_szeged['prediction'])
print "{:.3f}".format(r_2) + "\t" + "{:.3f}".format(p_value)

print "*****************"
var = "anx_state"
print var
p_value, r_2, residual, regline =  model.pred_stat(table_bochum[var], table_bochum['prediction'])
print "{:.3f}".format(r_2) + "\t" + "{:.3f}".format(p_value)

var = "anx_trait"
print var
p_value, r_2, residual, regline =  model.pred_stat(table_bochum[var], table_bochum['prediction'])
print "{:.3f}".format(r_2) + "\t" + "{:.3f}".format(p_value)

var = "ads_k"
print var
p_value, r_2, residual, regline =  model.pred_stat(table_bochum[var], table_bochum['prediction'])
print "{:.3f}".format(r_2) + "\t" + "{:.3f}".format(p_value)

var = "psq"
print var
p_value, r_2, residual, regline =  model.pred_stat(table_bochum[var], table_bochum['prediction'])
print "{:.3f}".format(r_2) + "\t" + "{:.3f}".format(p_value)

var = "pcs_catastrophizing"
print var
p_value, r_2, residual, regline =  model.pred_stat(table_bochum[var], table_bochum['prediction'])
print "{:.3f}".format(r_2) + "\t" + "{:.3f}".format(p_value)

var = "pcs_rumination"
print var
p_value, r_2, residual, regline =  model.pred_stat(table_bochum[var], table_bochum['prediction'])
print "{:.3f}".format(r_2) + "\t" + "{:.3f}".format(p_value)

var = "Glx_mean"
print var
p_value, r_2, residual, regline =  model.pred_stat(table_bochum[var], table_bochum['prediction'])
print "{:.3f}".format(r_2) + "\t" + "{:.3f}".format(p_value)

var = "GABA_mean"
print var
p_value, r_2, residual, regline =  model.pred_stat(table_bochum[var], table_bochum['prediction'])
print "{:.3f}".format(r_2) + "\t" + "{:.3f}".format(p_value)


print "*****************"
var = "MRI_QST_dif"
print var
p_value, r_2, residual, regline =  model.pred_stat(table_bochum[var], table_bochum['prediction'])
print "{:.3f}".format(r_2) + "\t" + "{:.3f}".format(p_value)
p_value, r_2, residual, regline =  model.pred_stat(table_essen[var], table_essen['prediction'])
print "{:.3f}".format(r_2) + "\t" + "{:.3f}".format(p_value)
p_value, r_2, residual, regline =  model.pred_stat(table_szeged[var], table_szeged['prediction'])
print "{:.3f}".format(r_2) + "\t" + "{:.3f}".format(p_value)

var = "t50"
print var
p_value, r_2, residual, regline =  model.pred_stat(table_bochum[var], table_bochum['prediction'])
print "{:.3f}".format(r_2) + "\t" + "{:.3f}".format(p_value)

var = "CDT_log_mean"
print var
p_value, r_2, residual, regline =  model.pred_stat(table_bochum[var], table_bochum['prediction'])
print "{:.3f}".format(r_2) + "\t" + "{:.3f}".format(p_value)
p_value, r_2, residual, regline =  model.pred_stat(table_essen[var], table_essen['prediction'])
print "{:.3f}".format(r_2) + "\t" + "{:.3f}".format(p_value)
p_value, r_2, residual, regline =  model.pred_stat(table_szeged[var], table_szeged['prediction'])
print "{:.3f}".format(r_2) + "\t" + "{:.3f}".format(p_value)

var = "WDT_log_mean"
print var
p_value, r_2, residual, regline =  model.pred_stat(table_bochum[var], table_bochum['prediction'])
print "{:.3f}".format(r_2) + "\t" + "{:.3f}".format(p_value)
p_value, r_2, residual, regline =  model.pred_stat(table_essen[var], table_essen['prediction'])
print "{:.3f}".format(r_2) + "\t" + "{:.3f}".format(p_value)
p_value, r_2, residual, regline =  model.pred_stat(table_szeged[var], table_szeged['prediction'])
print "{:.3f}".format(r_2) + "\t" + "{:.3f}".format(p_value)

var = "MDT_log_geom"
print var
#p_value, r_2, residual, regline =  model.pred_stat(table_bochum[var], table_bochum['prediction'])
#print "{:.3f}".format(r_2) + "\t" + "{:.3f}".format(p_value)
print "NA\tNA"
p_value, r_2, residual, regline =  model.pred_stat(table_essen[var], table_essen['prediction'])
print "{:.3f}".format(r_2) + "\t" + "{:.3f}".format(p_value)
p_value, r_2, residual, regline =  model.pred_stat(table_szeged[var], table_szeged['prediction'])
print "{:.3f}".format(r_2) + "\t" + "{:.3f}".format(p_value)

