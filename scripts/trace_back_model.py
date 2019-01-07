#!/usr/bin/env python
#!/usr/bin/env python
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.externals import joblib
from PAINTeR import global_vars
from PAINTeR import plot
from PAINTeR import model
import seaborn as sns
from nilearn.connectome import vec_to_sym_matrix, sym_matrix_to_vec

# load trained model
M = joblib.load(global_vars._RES_PRED_MOD_)
# load atlas labels
labels = pd.read_csv(global_vars._ATLAS_LABELS_, sep="\t")[["index", "labels", "modules"]]

labels.loc[-1] = [0, "aMEAN_GM", "aMEAN_GM"]  # adding a row
labels.index = labels.index + 1  # shifting index
labels = labels.sort_index()  # sorting by index

RES = np.zeros(len(labels)*(len(labels)-1)/2)

featuremask = M.named_steps['fsel'].get_support()
RES[featuremask] = M.named_steps['model'].coef_

RES_MAT = vec_to_sym_matrix(RES, diagonal=np.repeat(0, len(labels)))
plot.plot_matrix(RES_MAT, labels['labels'].values, labels['modules'].values, outfile=global_vars._PLOT_PRED_MATRIX_)

df = pd.DataFrame(RES_MAT, columns=labels['labels'].values, index=labels['labels'].values)

idx = np.transpose(np.nonzero(np.triu(RES_MAT, k=1)))
print "Number of predictive connections:" + str(len(idx))

lab = labels["labels"].values
mod = labels["modules"].values
# hack for visulaization order
mod = np.array(['2_MEAN_GM',
'1_CER', '1_CER', '1_CER', '1_CER', '1_CER', '1_CER', '1_CER', '1_CER', '1_CER', '1_CER',
 '1_CER', '1_CER', '1_CER', '1_CER', '1_CER', '1_CER', '7_DMnet', '7_DMnet', '7_DMnet', '7_DMnet',
 '7_DMnet', '7_DMnet', '7_DMnet', '7_DMnet', '7_DMnet', '7_DMnet', '7_DMnet', '7_DMnet', '7_DMnet',
 '7_DMnet', '7_DMnet', '7_DMnet', '7_DMnet', '7_DMnet', '7_DMnet', '7_DMnet', '7_DMnet', '7_DMnet',
 '5_FPnet_VISDN', '5_FPnet_VISDN', '5_FPnet_VISDN', '5_FPnet_VISDN', '5_FPnet_VISDN',
 '5_FPnet_VISDN', '5_FPnet_VISDN', '5_FPnet_VISDN', '5_FPnet_VISDN', '5_FPnet_VISDN',
 '5_FPnet_VISDN', '5_FPnet_VISDN', '5_FPnet_VISDN', '5_FPnet_VISDN', '5_FPnet_VISDN',
 '5_FPnet_VISDN', '5_FPnet_VISDN', '5_FPnet_VISDN', '5_FPnet_VISDN', '5_FPnet_VISDN',
 '5_FPnet_VISDN', '5_FPnet_VISDN', '5_FPnet_VISDN', '5_FPnet_VISDN', '5_FPnet_VISDN',
 '5_FPnet_VISDN', '5_FPnet_VISDN', '5_FPnet_VISDN', '4_LIMnet', '4_LIMnet', '4_LIMnet',
 '4_LIMnet', '4_LIMnet', '4_LIMnet', '4_LIMnet', '4_LIMnet', '4_LIMnet', '4_LIMnet', '4_LIMnet',
 '4_LIMnet', '6_MOTnet', '6_MOTnet', '6_MOTnet', '6_MOTnet', '6_MOTnet', '6_MOTnet', '6_MOTnet',
 '3_VATTnet_SALnet_BG_THAL', '3_VATTnet_SALnet_BG_THAL',
 '3_VATTnet_SALnet_BG_THAL', '3_VATTnet_SALnet_BG_THAL',
 '3_VATTnet_SALnet_BG_THAL', '3_VATTnet_SALnet_BG_THAL',
 '3_VATTnet_SALnet_BG_THAL', '3_VATTnet_SALnet_BG_THAL',
 '3_VATTnet_SALnet_BG_THAL', '3_VATTnet_SALnet_BG_THAL',
 '3_VATTnet_SALnet_BG_THAL', '3_VATTnet_SALnet_BG_THAL',
 '3_VATTnet_SALnet_BG_THAL', '3_VATTnet_SALnet_BG_THAL',
 '3_VATTnet_SALnet_BG_THAL', '3_VATTnet_SALnet_BG_THAL',
 '3_VATTnet_SALnet_BG_THAL', '3_VATTnet_SALnet_BG_THAL',
 '3_VATTnet_SALnet_BG_THAL', '3_VATTnet_SALnet_BG_THAL',
 '3_VATTnet_SALnet_BG_THAL', '3_VATTnet_SALnet_BG_THAL',
 '3_VATTnet_SALnet_BG_THAL', '3_VATTnet_SALnet_BG_THAL', '8_VISnet', '8_VISnet',
 '8_VISnet', '8_VISnet', '8_VISnet', '8_VISnet', '8_VISnet', '8_VISnet', '8_VISnet', '8_VISnet',
 '8_VISnet', '8_VISnet', '8_VISnet'])

#hack: rename for visualization order

table = pd.DataFrame( {'idx_A': idx[:,0],
                   'reg_A': lab[np.array(idx[:,0])],
                   'mod_A': mod[np.array(idx[:,0])],
                   'idx_B': idx[:,1],
                   'reg_B': lab[np.array(idx[:,1])],
                   'mod_B': mod[np.array(idx[:,1])],
                   'weight': RES_MAT[np.nonzero(np.triu(RES_MAT, k=1))].flatten()})

print table
table.to_csv(global_vars._RES_PRED_CONN_)

print "Region indices:"
indices_pred = np.unique(np.sort(idx.flatten()))
print len(indices_pred)
print indices_pred

print "Weights:"
w_all = np.sum(RES_MAT, axis=1)
print len(w_all[indices_pred])
print w_all[indices_pred]

print pd.DataFrame({"reg": indices_pred,
                    "w": w_all[indices_pred]})