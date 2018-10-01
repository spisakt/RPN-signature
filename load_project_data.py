import numpy as np
import pandas as pd

def load_QST_data(data_filename='/Users/tspisak/data/Mercure_rest/data.csv',
                  target_var="mean_QST_pain_sensitivity", sep=",", idvar="ID", exclude_ids=[1,2,3, 8, 21, 22, 40]):
    # load QST
    # 1,2,3: pilots
    # 40: bad TR data
    data = pd.read_csv(data_filename, sep=sep)
    # excludes:
    for id in exclude_ids:
        data = data[data[idvar] != id]

    y = data[target_var].values.ravel()
    if y.std() < 0.0001:
        raise Warning("Data has to be upsacled!")
        y = y* 1000

    #print "*** Included subjects: " + str(data[idvar])

    return y

def load_QST_data_all(data_filename='/Users/tspisak/data/Mercure_rest/data.csv',
                  sep=",", idvar="ID", exclude_ids=[1,2,3, 8, 21, 22, 40]):
    # load QST
    # 1,2,3: pilots
    # 40: bad TR data
    data = pd.read_csv(data_filename, sep=sep)
    # excludes:
    for id in exclude_ids:
        data = data[data[idvar] != id]

    #print "*** Included subjects: " + str(data[idvar])

    return data


def load_timeseries_tsv(file_list, standardise=True):
    pooled_subjects = []
    for f in file_list:
        ts = pd.read_csv(f, sep="\t").values

        if standardise:
            from sklearn.preprocessing import StandardScaler
            ts = StandardScaler().fit_transform(ts)

        pooled_subjects.append(ts)
        #import matplotlib.pyplot as plt
        #plt.plot(ts[..., 0])
        #plt.show()

    return pooled_subjects



def load_timeseries_sav(timeseries_sav_file="/Users/tspisak/projects/Mercure_rest/src/conn/timeseries_122_friston_new.sav"):
    from sklearn.externals import joblib
    pooled_timeseries = joblib.load(timeseries_sav_file)
    #pooled_timeseries = np.array(pooled_timeseries)
    return pooled_timeseries

def compute_connectivity(pooled_timeseries, kind="tangent", discard_diagonal=True):
    from nilearn.connectome import ConnectivityMeasure
    correlation_measure = ConnectivityMeasure(kind=kind, vectorize=True, discard_diagonal=discard_diagonal)
    correlation_matrix = correlation_measure.fit_transform(pooled_timeseries)
    return correlation_matrix, correlation_measure

def compute_dynconn(pooled_timeseries, kind="tangent", timewindow=38,winstepsize=2):
    from nilearn.connectome import ConnectivityMeasure
    strtidx = np.arange(((pooled_timeseries[0].shape[0] - timewindow) / winstepsize) + 1) * winstepsize
    conn_measure = ConnectivityMeasure(kind=kind, vectorize=True, discard_diagonal=True)
    funccorstd = []
    for i in range(len(pooled_timeseries)):
        print i
        slidingwincormatrix = []
        tmpsubj = pooled_timeseries[i]
        subject = []
        for j in strtidx:
            subject.append(np.array(tmpsubj[j:(j + timewindow), :]))
        tmp = conn_measure.fit_transform(subject)
        tempstdmatricis = np.std(np.array(tmp), axis=0)
        funccorstd.append(tempstdmatricis)
    return funccorstd