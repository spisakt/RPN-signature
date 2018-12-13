#!/usr/bin/env python
import warnings
warnings.filterwarnings("always")
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
from PAINTeR import global_vars
import PAINTeR.connectivity as conn


#######################################
# processing parameters:
thres_mean_FD = 0.15 # mm
scrubbing=True
scrub_threshold = 0.15 # mm
thres_perc_scrub = 30 # % scubbed out
#######################################

#######################################
# BOCHUM sample                       #
#######################################
conn.calculate_connectivity(table=global_vars._RES_BOCHUM_TABLE_,
                           fd_files=global_vars.bochum_fd_files,
                           ts_files=global_vars.bochum_ts_files,
                           atlas_file=global_vars._ATLAS_FILE_,
                           thres_mean_FD=thres_mean_FD,
                           scrubbing=scrubbing,
                           scrub_threshold=scrub_threshold,
                           thres_perc_scrub=30,
                           plot_labelmap=False,
                           plot_connectome=False,
                           plotfile_mean_mtx=global_vars._PLOT_BOCHUM_MEAN_MATRIX_,
                           save_features=global_vars._FEATURE_BOCHUM_,
                           save_table=global_vars._RES_BOCHUM_TABLE_,
                           save_table_excl=global_vars._RES_BOCHUM_TABLE_EXCL_
                           )

#######################################
# ESSEN sample                        #
#######################################
conn.calculate_connectivity(table=global_vars._RES_ESSEN_TABLE_,
                           fd_files=global_vars.essen_fd_files,
                           ts_files=global_vars.essen_ts_files,
                           atlas_file=global_vars._ATLAS_FILE_,
                           thres_mean_FD=thres_mean_FD,
                           scrubbing=scrubbing,
                           scrub_threshold=scrub_threshold,
                           thres_perc_scrub=thres_perc_scrub,
                           plot_labelmap=False,
                           plot_connectome=False,
                           plotfile_mean_mtx=global_vars._PLOT_ESSEN_MEAN_MATRIX_,
                           save_features=global_vars._FEATURE_ESSEN_,
                           save_table=global_vars._RES_ESSEN_TABLE_,
                           save_table_excl=global_vars._RES_ESSEN_TABLE_EXCL_
                           )

#######################################
# SZEGED sample                       #
#######################################
conn.calculate_connectivity(table=global_vars._RES_SZEGED_TABLE_,
                           fd_files=global_vars.szeged_fd_files,
                           ts_files=global_vars.szeged_ts_files,
                           atlas_file=global_vars._ATLAS_FILE_,
                           thres_mean_FD=thres_mean_FD,
                           scrubbing=scrubbing,
                           scrub_threshold=scrub_threshold,
                           thres_perc_scrub=thres_perc_scrub,
                           plot_labelmap=False,
                           plot_connectome=False,
                           plotfile_mean_mtx=global_vars._PLOT_SZEGED_MEAN_MATRIX_,
                           save_features=global_vars._FEATURE_SZEGED_,
                           save_table=global_vars._RES_SZEGED_TABLE_,
                           save_table_excl=global_vars._RES_SZEGED_TABLE_EXCL_
                           )