import numpy as np
import pandas as pd

def load_QST_data(data_filename='/home/analyser/Documents/PAINTER/data.csv',
                  target_var="mean_QST_pain_sensitivity", sep=",", exclude_ids=[1,2,3, 8, 21, 22]):
    # load QST
    # 1,2,3: pilots
    # 40: bad TR data
    data = pd.read_csv(data_filename, sep=sep)
    # excludes:
    for id in exclude_ids:
        data = data[data["ID"] != id]

    y = data[target_var].values.ravel()
    if y.std() < 0.0001:
        raise Warning("Data has to be upsacled!")
        y = y* 1000

    print "*** Included subjects: " + str(data["ID"])

    return y

def load_timeseries_tsv(file_list):
    pooled_subjects = []
    for f in file_list:
        ts = pd.read_csv(f, sep="\t").values
        pooled_subjects.append(ts)
        #import matplotlib.pyplot as plt
        #plt.plot(ts[..., 0])
        #plt.show()

    return pooled_subjects



def load_timeseries_sav(timeseries_sav_file="/Users/tspisak/projects/Mercure_rest/src/conn/timeseries_122_friston_new.sav"):
    from sklearn.externals import joblib
    pooled_timeseries = joblib.load(timeseries_sav_file)
    pooled_timeseries = np.array(pooled_timeseries)
    return pooled_timeseries

def compute_connectivity(pooled_timeseries, kind="tangent"):
    from nilearn.connectome import ConnectivityMeasure
    correlation_measure = ConnectivityMeasure(kind=kind, vectorize=True, discard_diagonal=True)
    correlation_matrix = correlation_measure.fit_transform(pooled_timeseries)
    return correlation_matrix


