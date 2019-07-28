#!/usr/bin/env python
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
import pandas as pd
import numpy as np
from sklearn.externals import joblib
from PAINTeR import global_vars
import PAINTeR.connectivity as conn
from PAINTeR import plot
from PAINTeR import model
from sklearn import preprocessing
from sklearn.linear_model import ElasticNet
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
# scikit-learn bootstrap
from sklearn.utils import resample
from nilearn.connectome import vec_to_sym_matrix, sym_matrix_to_vec

# load pain sensitivity data (excluded)
y = pd.read_csv(global_vars._RES_BOCHUM_TABLE_EXCL_)['mean_QST_pain_sensitivity']

# load features (excluded)
X = joblib.load(global_vars._FEATURE_BOCHUM_)

# load up trained model
m_orig=joblib.load(global_vars._RES_PRED_MOD_)

# pre-selecxt features as in the original model
featuremask = m_orig.named_steps['fsel'].get_support()
original_coefs = m_orig.named_steps['model'].coef_
print(X.shape)
X=X[:,featuremask]
print(X.shape)

# create identical model to be bootstrapped
def pipe_scale_elnet(scaler=preprocessing.RobustScaler(),
                          model=ElasticNet(alpha=m_orig.named_steps['model'].alpha, l1_ratio=m_orig.named_steps['model'].l1_ratio, max_iter=100000)):
    mymodel = Pipeline(
        [('scaler', scaler),
         ('model', model)])
    return mymodel

m_boot=pipe_scale_elnet()

# load atlas labels
labels = pd.read_csv(global_vars._ATLAS_LABELS_, sep="\t")[["index", "labels", "modules"]]

labels.loc[-1] = [0, "aMEAN_GM", "aMEAN_GM"]  # adding a row
labels.index = labels.index + 1  # shifting index
labels = labels.sort_index()  # sorting by index

N_samples=10000
N_obs=len(y)#global_vars.N # same as the original data
indices = range(len(y))

RES = np.zeros([N_samples,len(labels) * (len(labels) - 1) / 2])

for boot_i in range(N_samples):
    # prepare bootstrap sample
    boot = resample(indices, replace=True, n_samples=N_obs)
    #print('Bootstrap Sample: %s' % boot)
    #print(X[boot])
    # out of bag observations
    oob = [x for x in indices if x not in boot]
    #print('OOB Sample: %s' % oob)

    bootfit=m_boot.fit(X[boot], y[boot])


    RES[boot_i,featuremask] = bootfit.named_steps['model'].coef_

    #print(RES[boot_i,featuremask])

    RES_MAT = vec_to_sym_matrix(RES[boot_i], diagonal=np.repeat(0, len(labels)))
    #plot.plot_matrix(RES_MAT, labels['labels'].values, labels['modules'].values, outfile=global_vars._PLOT_PRED_MATRIX_)

    idx = np.transpose(np.nonzero(np.triu(RES_MAT, k=1)))
    #print "Number of predictive connections:" + str(len(idx))



df = pd.DataFrame(RES_MAT, columns=labels['labels'].values, index=labels['labels'].values)

cils=np.zeros(len(original_coefs))
cihs=np.zeros(len(original_coefs))
occs=np.zeros(len(original_coefs))
ps=np.zeros(len(original_coefs))

for coef_i in range(len(original_coefs)):
    #print("orig_coef : " + str(original_coefs[coef_i]))
    #print(RES[:, featuremask][:,coef_i])

    data=RES[:, featuremask][:,coef_i]
    confidence=np.quantile(data[np.nonzero(data)] , [0.025, 0.975])
    cils[coef_i]=confidence[0]
    cihs[coef_i] = confidence[1]

    occs[coef_i]=1.0*sum(i != 0 for i in RES[:, featuremask][:, coef_i])/N_samples

    if original_coefs[coef_i]>0:
        p = 1-1.0 * sum(i > 0 for i in RES[:, featuremask][:, coef_i]) / sum(i != 0 for i in RES[:, featuremask][:, coef_i])#N_samples
    else:
        p = 1-1.0 * sum(i < 0 for i in RES[:, featuremask][:, coef_i]) / sum(i != 0 for i in RES[:, featuremask][:, coef_i])#N_samples
    ps[coef_i]=p


regs=["PO/pSTG	VAN+SN+BG+Thal	119	pPut	VAN+SN+BG+Thal",
"FP	FPN	75	5	CER",
"pCVI	CER	9	SMC	VAN+SN+BG+Thal",
"R aCrus2	CER	62	lPrCG	SMN",
"dPrCG	SMN	67	pmVN	VN",
"pdlVN	VN	43	mVN	VN",
"L IPL	DMN	114	mean GM	mean GM",
"vCaud	VAN+SN+BG+Thal	2	plVN	VN",
"Acc	MLN	78	pvmVN	VN",
"CF	MLN	79	vlPrCG	SMN",
"5	CER	48	pdlVN	VN",
"pThal/Hb	VAN+SN+BG+Thal	36	plVN	VN",
"dCVI	CER	44	lOTG	FPN",
"dCiX	CER	11	L vMFG	FPN",
"R IPS	FPN	20	plVN	VN",
"avIns	VAN+SN+BG+Thal	12	admVN	VN",
"R aMFG	FPN	58	lPoCG	VAN+SN+BG+Thal",
"CrusI	CER	84	dPoCG	VAN+SN+BG+Thal",
"pgACC	DMN	115	mSTG	VAN+SN+BG+Thal",
"Precun	DMN	103	LOG	MLN",
"vThal	VAN+SN+BG+Thal	36	FEF	VAN+SN+BG+Thal", " ", " ", " ", " "]


table = pd.DataFrame({'_orig_coef': original_coefs,
                    '_orig_coef_abs': np.abs(original_coefs),
                    '_occurance': occs*100,
                    'ci_low': cils,
                    'ci_high': cihs,
                    'p': ps,
                    })

table=table.sort_values('_orig_coef_abs', ascending=False)
table["_REG"]=regs
table=table.reset_index()
print(table)

for i in table['p']: print(i)
for i in range(len(original_coefs)): print("[" + str(table['ci_low'][i]) + ", " + str(table['ci_high'][i]) + "]")
for i in table['_occurance']: print(i)