import pandas as pd
import numpy as np
from PAINTeR import global_vars
from PAINTeR import model
from PAINTeR import plot
from sklearn.preprocessing import StandardScaler
from nilearn.connectome import ConnectivityMeasure
from sklearn.externals import joblib



def calculate_connectivity(table=global_vars._RES_BOCHUM_TABLE_,
                           fd_files=global_vars.bochum_fd_files,
                           ts_files=global_vars.bochum_ts_files,
                           atlas_file=global_vars._ATLAS_FILE_,
                           thres_mean_FD=0.15,
                           scrubbing=True,
                           scrub_threshold=0.15,
                           thres_perc_scrub=30,
                           plot_labelmap=False,
                           plot_connectome=False,
                           plotfile_mean_mtx=global_vars._PLOT_BOCHUM_MEAN_MATRIX_,
                           save_features=global_vars._FEATURE_BOCHUM_,
                           save_table=global_vars._RES_BOCHUM_TABLE_,
                           save_table_excl=global_vars._RES_BOCHUM_TABLE_EXCL_
                           ):
    # load bochum data
    df = pd.read_csv(table)

    # load FD data
    add_FD_data(fd_files, df)

    # load timeseries data
    ts, labels = load_timeseries(ts_files, df, scrubbing=scrubbing,
                                      scrub_threshold=scrub_threshold)

    # exclude data with extensive motion
    df_excl = exclude(df, thres_mean_FD=thres_mean_FD, thres_perc_scrub=thres_perc_scrub, QST=True )

    # compute connectivity
    excl = np.argwhere(df['Excluded'].values < 1).flatten()
    X, cm = connectivity_matrix(np.array(ts)[excl])
    # plot group-mean matrix
    l = pd.read_csv(global_vars._ATLAS_LABELS_, sep="\t")
    if plot_labelmap:
        plot.plot_labelmap(atlas_file)
    plot.plot_matrix(cm.mean_, labels, np.insert(l['modules'].values, 0, "GlobSig"),
                     outfile=plotfile_mean_mtx)
    if plot_connectome:
        plot.plot_connectome(cm.mean_, atlas_file, threshold=0.05)

    #X=np.arctanh(X)
    #X = X/np.percentile(X, 95)

    # serialise feature space
    if save_features:
        joblib.dump(X, save_features)

    # save data.frame
    print("Saving:" + save_table)
    df.to_csv(save_table)
    df_excl.to_csv(save_table_excl)

    return X

def add_FD_data(fd_files, data_frame):
    FD = []
    mean_FD = []
    median_FD = []
    max_FD = []
    for f in fd_files:
        fd = pd.read_csv(f, sep="\t").values.flatten()
        fd = np.insert(fd, 0, 0)
        FD.append(fd.ravel())
        mean_FD.append(fd.mean())
        median_FD.append(np.median(fd))
        max_FD.append(fd.max())

    print (len(mean_FD))
    print (data_frame.values.shape)

    data_frame['meanFD'] = mean_FD
    data_frame['medianFD'] = median_FD
    data_frame['maxFD'] = max_FD
    data_frame['fd_file'] = fd_files
    return data_frame

def scrub(ts, fd, scrub_threshold, frames_before=0, frames_after = 0):

    frames_out = np.argwhere(fd>scrub_threshold).flatten().tolist()
    extra_indices=[]
    for i in frames_out:
        # remove preceding frames
        if i > 0:
            count = 1
            while count <= frames_before:
                extra_indices.append(i - count)
                count += 1

        # remove following frames
        count = 1
        while count <= frames_after:
            if i+count < len(fd):  # do not censor unexistent data
                extra_indices.append(i + count)
            count += 1

    indices_out = list(set(frames_out) | set(extra_indices))
    indices_out.sort()

    return np.delete(ts, indices_out, axis=0)

def load_timeseries(ts_files, data_frame, scrubbing = True, scrub_threshold=0.15):
    timeseries = []
    perc_scrubbed = []
    for i, f in enumerate(ts_files):
        ts = pd.read_csv(f, sep="\t").values
        # ts = pd.read_csv(f, sep="\t").drop('GlobSig', 1).values

        fd = pd.read_csv(data_frame["fd_file"].values.ravel()[i]).values.ravel().tolist()
        fd = [0] + fd
        if scrubbing:
            ts = scrub(ts, np.array(fd), scrub_threshold)  # ts[np.array(fd) < scrub_threshold, ]

        perc_scrubbed.append(100 - 100*len(ts[:,1])/len(fd) )

        # standardise timeseries
        ts = StandardScaler().fit_transform(ts)
        timeseries.append(ts)

    data_frame['perc_scrubbed'] = perc_scrubbed
    data_frame['ts_file'] = ts_files

    labels = pd.read_csv(ts_files[0], sep="\t").columns

    return timeseries, labels

def exclude(data_frame,
            thres_mean_FD=None, thres_median_FD=None, thres_perc_scrub=None,
            QST=False,
            CPT_min=0, #ref value: m
            CPT_max=27.21, #ref value: f
            HPT_min=36.33, #ref value: f
            HPT_max=49.88, #ref value: m
            MPT_min=2.48, #ref value: m
            MPT_max=6.35 #ref value: m
            ):
    # Just in case:
    data_frame['exclusion_crit'] = data_frame['exclusion_crit'].astype(str)
    if thres_mean_FD:
        data_frame.loc[data_frame.meanFD > thres_mean_FD, 'Excluded'] = 1
        data_frame.loc[data_frame.meanFD > thres_mean_FD, 'exclusion_crit'] += '+meanFD'
    if thres_median_FD:
        data_frame.loc[data_frame.medianFD > thres_median_FD, 'Excluded'] = 1
        data_frame.loc[data_frame.medianFD > thres_median_FD, 'exclusion_crit'] += '+medianFD'
    if thres_perc_scrub:
        data_frame.loc[data_frame.perc_scrubbed > thres_perc_scrub, 'Excluded'] = 1
        data_frame.loc[data_frame.perc_scrubbed > thres_perc_scrub, 'exclusion_crit'] += '+perc_scrub'

    if QST:
        data_frame.loc[data_frame.CPT < CPT_min, 'Excluded'] += 0.5
        data_frame.loc[data_frame.CPT < CPT_min, 'exclusion_crit'] += '+CPT_low'

        data_frame.loc[data_frame.CPT > CPT_max, 'Excluded'] += 0.5
        data_frame.loc[data_frame.CPT > CPT_max, 'exclusion_crit'] += '+CPT_high'

        data_frame.loc[data_frame.HPT < HPT_min, 'Excluded'] += 0.5
        data_frame.loc[data_frame.HPT < HPT_min, 'exclusion_crit'] += '+HPT_low'

        data_frame.loc[data_frame.HPT > HPT_max, 'Excluded'] += 0.5
        data_frame.loc[data_frame.HPT > HPT_max, 'exclusion_crit'] += '+HPT_high'

        data_frame.loc[data_frame.MPT_log_geom < MPT_min, 'Excluded'] += 0.5
        data_frame.loc[data_frame.MPT_log_geom < MPT_min, 'exclusion_crit'] += '+MPT_low'

        data_frame.loc[data_frame.MPT_log_geom > MPT_max, 'Excluded'] += 0.5
        data_frame.loc[data_frame.MPT_log_geom > MPT_max, 'exclusion_crit'] += '+MPT_high'

    print "Before exclusion: " + str(data_frame.shape[0])
    data_frame_excl = data_frame[data_frame["Excluded"] < 1]
    print "After exclusion: " + str(data_frame_excl.shape[0])
    return data_frame_excl

def connectivity_matrix(timeseries, kind='partial correlation'):
    # timeseries: as output by load_timeseries
    correlation_measure = ConnectivityMeasure(kind=kind, vectorize=True, discard_diagonal=True)
    correlation_matrix = correlation_measure.fit_transform(timeseries)
    return correlation_matrix, correlation_measure