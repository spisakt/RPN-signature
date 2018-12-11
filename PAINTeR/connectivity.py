import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

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

    data_frame['meanFD'] = mean_FD
    data_frame['medianFD'] = median_FD
    data_frame['maxFD'] = max_FD
    data_frame['fd_file'] = fd_files

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

        perc_scrubbed.append( 100*len(ts[:,1])/len(fd) )
        print 100-100*len(ts[:,1])/len(fd)

        # standardise timeseries
        ts = StandardScaler().fit_transform(ts)
        timeseries.append(ts)

    data_frame['perc_scrubbed'] = perc_scrubbed
    data_frame['ts_file'] = ts_files

    return timeseries

def exclude():
    # ToDo: add timeseries file and percent scrubbed
    return 0
