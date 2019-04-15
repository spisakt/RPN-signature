#!/usr/bin/env python
# This is a pipeline doing resting-state fMRI preprocessing and calculates the RPN-siganture
# that is, makes a guess for pain sensitivity
#
# This is the BIDS version.
# See:
# https://spisakt.github.io/RPN-signature/
#
# The pipeline performs:
# - a standard anatomical image processing (AnatProc) with:
#   - brain extraction with FSL bet
#   - FSL FAST tissue segmentation
#   - ANTs-based standardisation
#   - etc.
# - BBR-based func2anat registration
# - preprocessing of functional images with (FuncProc_despike_afni):
#   - brain extraction
#   - FSL McFlirt motion correction
#   - despiking with AFNI
#   - temporal filtering (default band: 0.008-0.08)
#   - a "liberal" scrubbing (since despiking has done much of the work already...)
# - picks brain atlas, reorders labels by modules
# - transforms atlas into the native space, masks it with native GM mask and generates regional timeseries
# - computes partial correlation matrix based on the normalised regional timeseries
#
# And finally:
# - computing the pain sensitivity score


# start it like:
# /Users/tspisak/src/PUMI/scripts/pipeline_PAINTeR.py "/Users/tspisak/projects/PAINTeR/data/nii-szeged/data/PA*/highres.nii" "/Users/tspisak/projects/PAINTeR/data/nii-szeged/data/PA*/rest.nii" /Users/tspisak/projects/PAINTeR/res/szeged /Users/tspisak/data/atlases/MIST/


import sys
import os

import nipype
import nipype.pipeline as pe
import nipype.interfaces.io as io
import nipype.interfaces.utility as util
import nipype.interfaces.fsl as fsl

import PUMI.AnatProc as anatproc
import PUMI.FuncProc as funcproc
import PUMI.func_preproc.Compcor as cc
import PUMI.anat_preproc.Func2Anat as bbr
import PUMI.utils.utils_convert as utils_convert
import PUMI.utils.globals as globals
import PUMI.connectivity.TimeseriesExtractor as tsext


# parse command line arguments
if (len(sys.argv) <= 2):
    print("Please specify command line arguments!")
    print("Usage:")
    print(sys.argv[0] + " bids_dir result_directory [bet_fract_int_thr] [bet_vert_grad_thr] [atlas_directory]")
    print("Example:")
    print(sys.argv[0] + " /home/user/my_bids_dir /home/user/my_bids_dir/derivatives/RPN")
    quit()

if (len(sys.argv) > 3):
    _MISTDIR_=sys.argv[3]
else:
    _MISTDIR_ = os.path.abspath(os.path.join(os.path.dirname(__file__),"../data/atlas/MIST"))

if (len(sys.argv) > 4):
    bet_fract_int_thr = sys.argv[4]
else:
    bet_fract_int_thr = 0.4  # default

if (len(sys.argv) > 5):
    bet_fract_int_thr = sys.argv[5]
else:
    bet_vertical_gradient = -0.1   # default

##############################
globals._brainref="/data/standard/MNI152_T1_1mm_brain.nii.gz"
globals._headref="/data/standard/MNI152_T1_1mm.nii.gz"
globals._brainref_mask="/data/standard/MNI152_T1_1mm_brain_mask_dil.nii.gz"
##############################
_refvolplace_ = globals._RefVolPos_.first


# specify atlas for network construction:
# name of labelmap nii (or list of probmaps)
_ATLAS_FILE = _MISTDIR_ + '/Parcellations/MIST_122.nii.gz'
# a list of labels, where index+1 corresponds to the label in the labelmap
_ATLAS_LABELS = tsext.mist_labels(mist_directory=_MISTDIR_, resolution="122")
# a list of labels, where index i corresponds to the module of the i+1th region, this is optional
_ATLAS_MODULES = tsext.mist_modules(mist_directory=_MISTDIR_, resolution="122")
##############################
##############################
#_regtype_ = globals._RegType_.FSL
globals._regType_ = globals._RegType_.ANTS
##############################

totalWorkflow = nipype.Workflow('RPN')
totalWorkflow.base_dir = '.'

########################
# parse command line args
bids_dir = sys.argv[1]

# create BIDS data grabber
datagrab = pe.Node(io.BIDSDataGrabber(), name='data_grabber')
datagrab.inputs.base_dir = bids_dir
#datagrab.inputs.subject = '01' # ToDo: specify subjects

# sink: file - idx relationship!!
pop_id = pe.Node(interface=utils_convert.List2TxtFile,
                     name='pop_id')
pop_id.inputs.rownum = 0
pop_id.inputs.out_file = "subject_IDs.txt"
totalWorkflow.connect(datagrab, 'bold', pop_id, 'in_list')

ds_id = pe.Node(interface=io.DataSink(), name='ds_pop_id')
ds_id.inputs.regexp_substitutions = [("(\/)[^\/]*$", "IDs.txt")]
ds_id.inputs.base_directory = globals._SinkDir_
totalWorkflow.connect(pop_id, 'txt_file', ds_id, 'subjects')

# build the actual pipeline
reorient_struct = pe.MapNode(fsl.utils.Reorient2Std(output_type='NIFTI'),
                      iterfield=['in_file'],
                      name="reorient_struct")
totalWorkflow.connect(datagrab, 'T1w', reorient_struct, 'in_file')

reorient_func = pe.MapNode(fsl.utils.Reorient2Std(output_type='NIFTI'),
                      iterfield=['in_file'],
                      name="reorient_func")
totalWorkflow.connect(datagrab, 'bold', reorient_func, 'in_file')

myanatproc = anatproc.AnatProc(stdreg=globals._regType_)
myanatproc.inputs.inputspec.bet_fract_int_thr = bet_fract_int_thr #0.3  # feel free to adjust, a nice bet is important!
myanatproc.inputs.inputspec.bet_vertical_gradient = bet_vertical_gradient #-0.3 # feel free to adjust, a nice bet is important!
# try scripts/opt_bet.py to optimise these parameters
totalWorkflow.connect(reorient_struct, 'out_file', myanatproc, 'inputspec.anat')

mybbr = bbr.bbr_workflow()
totalWorkflow.connect(myanatproc, 'outputspec.brain', mybbr, 'inputspec.skull') #ToDo ready: rather input the brain extracted here?
totalWorkflow.connect(reorient_func, 'out_file', mybbr, 'inputspec.func')
totalWorkflow.connect(myanatproc, 'outputspec.probmap_wm', mybbr, 'inputspec.anat_wm_segmentation')
totalWorkflow.connect(myanatproc, 'outputspec.probmap_csf', mybbr, 'inputspec.anat_csf_segmentation')
totalWorkflow.connect(myanatproc, 'outputspec.probmap_gm', mybbr, 'inputspec.anat_gm_segmentation')
totalWorkflow.connect(myanatproc, 'outputspec.probmap_ventricle', mybbr, 'inputspec.anat_ventricle_segmentation')

compcor_roi = cc.create_anat_noise_roi_workflow()
totalWorkflow.connect(mybbr, 'outputspec.wm_mask_in_funcspace', compcor_roi, 'inputspec.wm_mask')
totalWorkflow.connect(mybbr, 'outputspec.ventricle_mask_in_funcspace', compcor_roi, 'inputspec.ventricle_mask')

# Preprocessing of functional data
myfuncproc = funcproc.FuncProc_despike_afni(carpet_plot="carpet_plots")
totalWorkflow.connect(reorient_func, 'out_file', myfuncproc, 'inputspec.func')
totalWorkflow.connect(compcor_roi, 'outputspec.noise_roi', myfuncproc, 'inputspec.cc_noise_roi')

# Pick atlas
pickatlas = tsext.PickAtlas()
pickatlas.inputs.inputspec.labelmap = _MISTDIR_ + "/Parcellations/MIST_122.nii.gz"
pickatlas.inputs.inputspec.modules = _ATLAS_MODULES
pickatlas.inputs.inputspec.labels = _ATLAS_LABELS

# Extract timeseries
extract_timeseries = tsext.extract_timeseries_nativespace()
totalWorkflow.connect(pickatlas, 'outputspec.relabeled_atlas', extract_timeseries, 'inputspec.atlas')
totalWorkflow.connect(pickatlas, 'outputspec.reordered_labels', extract_timeseries, 'inputspec.labels')
totalWorkflow.connect(pickatlas, 'outputspec.reordered_modules', extract_timeseries, 'inputspec.modules')
totalWorkflow.connect(myanatproc, 'outputspec.brain', extract_timeseries, 'inputspec.anat')
totalWorkflow.connect(mybbr, 'outputspec.anat_to_func_linear_xfm', extract_timeseries, 'inputspec.inv_linear_reg_mtrx')
totalWorkflow.connect(myanatproc, 'outputspec.mni2anat_warpfield', extract_timeseries, 'inputspec.inv_nonlinear_reg_mtrx')
totalWorkflow.connect(mybbr, 'outputspec.gm_mask_in_funcspace', extract_timeseries, 'inputspec.gm_mask')
totalWorkflow.connect(myfuncproc, 'outputspec.func_preprocessed', extract_timeseries, 'inputspec.func')
totalWorkflow.connect(myfuncproc, 'outputspec.FD', extract_timeseries, 'inputspec.confounds')

# Extract timeseries - scrubbed
#extract_timeseries_scrubbed = tsext.extract_timeseries_nativespace(SinkTag="connectivity_scrubbed", wf_name="extract_timeseries_nativespace_scribbed")
#totalWorkflow.connect(pickatlas, 'outputspec.relabeled_atlas', extract_timeseries_scrubbed, 'inputspec.atlas')
#totalWorkflow.connect(pickatlas, 'outputspec.reordered_labels', extract_timeseries_scrubbed, 'inputspec.labels')
#totalWorkflow.connect(pickatlas, 'outputspec.reordered_modules', extract_timeseries_scrubbed, 'inputspec.modules')
#totalWorkflow.connect(myanatproc, 'outputspec.brain', extract_timeseries_scrubbed, 'inputspec.anat')
#totalWorkflow.connect(mybbr, 'outputspec.anat_to_func_linear_xfm', extract_timeseries_scrubbed, 'inputspec.inv_linear_reg_mtrx')
#totalWorkflow.connect(myanatproc, 'outputspec.mni2anat_warpfield', extract_timeseries_scrubbed, 'inputspec.inv_nonlinear_reg_mtrx')
#totalWorkflow.connect(mybbr, 'outputspec.gm_mask_in_funcspace', extract_timeseries_scrubbed, 'inputspec.gm_mask')
#totalWorkflow.connect(myfuncproc, 'outputspec.func_preprocessed_scrubbed', extract_timeseries_scrubbed, 'inputspec.func')
#totalWorkflow.connect(myfuncproc, 'outputspec.FD', extract_timeseries_scrubbed, 'inputspec.confounds')

# Calculate RPN-score: prediction of pain sensitivity
def calculate_connectivity(ts_files, fd_files):
    # load FD data
    import pandas as pd
    import numpy as np
    import PAINTeR.connectivity as conn

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

    df = pd.DataFrame()

    df['ts_files'] = ts_files
    df['meanFD'] = mean_FD
    df['medianFD'] = median_FD
    df['maxFD'] = max_FD

    # load timeseries data
    ts, labels = conn.load_timeseries(ts_files, df, scrubbing=True,
                                 scrub_threshold=0.15)
    features, cm = conn.connectivity_matrix(np.array(ts))

    return features, df


conn = pe.Node(util.Function(input_names=['ts_files', 'fd_files'],
                             output_names=['features', 'df'],
                             function=calculate_connectivity), name="calculate_connectivity")
totalWorkflow.connect(extract_timeseries, 'outputspec.timeseries', conn, 'ts_files')
totalWorkflow.connect(myfuncproc, 'outputspec.FD', conn, 'fd_files')

def predict_pain_sensitivity(X, df):
    # load trained model
    from PAINTeR import global_vars
    from sklearn.externals import joblib
    model = joblib.load(global_vars._RES_PRED_MOD_FIXED_)
    predicted = model.predict(X)

    df['RPN'] = predicted

    pred_file="prediction.csv"
    df.to_csv()

    return pred_file


predict = pe.Node(util.Function(input_names=['X'],
                                output_names=['pred_file'],
                                function=predict_pain_sensitivity),
                  name='predict')

totalWorkflow.connect(conn, 'features', predict, 'X')

ds_pred = pe.Node(interface=io.DataSink(), name='ds_pred')
ds_pred.inputs.regexp_substitutions = [("(\/)[^\/]*$", "results.csv")]
ds_pred.inputs.base_directory = globals._SinkDir_
totalWorkflow.connect(predict, 'pred_file', ds_pred, 'RPN')


# RUN!

totalWorkflow.write_graph('graph-orig.dot', graph2use='orig', simple_form=True)
totalWorkflow.write_graph('graph-exec-detailed.dot', graph2use='exec', simple_form=False)
totalWorkflow.write_graph('graph.dot', graph2use='colored')

#from nipype import config
#config.enable_resource_monitor()
from nipype.utils.profiler import log_nodes_cb
#import logging
#callback_log_path = 'run_stats.log'
#logger = logging.getLogger('callback')
#logger.setLevel(logging.DEBUG)
#handler = logging.FileHandler(callback_log_path)
#logger.addHandler(handler)

plugin_args = {'n_procs' : 7,
               'memory_gb' : 10
              #'status_callback' : log_nodes_cb
               }
totalWorkflow.run(plugin='MultiProc', plugin_args=plugin_args)

#import PUMI.utils.resource_profiler as rp
#rp.generate_gantt_chart('run_stats.log', cores=8)