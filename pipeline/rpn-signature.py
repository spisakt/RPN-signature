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
# - computing the pain sensitivity score based on the serialized predictive model


# start it like:
# /Users/tspisak/src/PUMI/scripts/pipeline_PAINTeR.py "/Users/tspisak/projects/PAINTeR/data/nii-szeged/data/PA*/highres.nii" "/Users/tspisak/projects/PAINTeR/data/nii-szeged/data/PA*/rest.nii" /Users/tspisak/projects/PAINTeR/res/szeged /Users/tspisak/data/atlases/MIST/


import os
import psutil

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

############ this part follows the BIDS-app specification ##############################################################
# parse command line arguments
from argparse import ArgumentParser
from argparse import RawTextHelpFormatter

ver = open(os.path.join(os.path.dirname(__file__),"../_version"), "r")
__version__ = ver.readline()
ver.close()

parser = ArgumentParser(description='RPN-signature: Resting-state Pain susceptibility Network signature'
                        'to predict individual pain sensitivity based on resting-state fMRI.\n'
                        'Webpage: https://spisakt.github.io/RPN-signature/',
                            formatter_class=RawTextHelpFormatter)

parser.add_argument('--version', '-v', '--ver', action='version', version='RPN-signature ' + __version__)

parser.add_argument('bids_dir', action='store',
                        help='the root folder of a BIDS valid dataset (sub-XXXXX folders should '
                             'be found at the top level in this folder).')
parser.add_argument('output_dir', action='store',
                        help='the output path for the outcomes of preprocessing and visual '
                             'reports')
parser.add_argument('analysis_level', choices=['participant'],
                        help='processing stage to be run, only "participant" in the case of '
                             'the rpn-signature')

g_bids = parser.add_argument_group('Options for filtering BIDS queries')
g_bids.add_argument('--participant_label', '--participant-label', action='store', nargs='+',
                    help='a space delimited list of participant identifiers or a single '
                         'identifier (the sub- prefix can be removed)')
g_bids.add_argument('-t', '--task-id', '--task_id', action='store',
                        help='select a specific task to be processed (resting-state recommended for the rpn-signature)')
g_bids.add_argument('--echo_idx', '--echo-idx', action='store', type=int,
                    help='select a specific echo to be processed in a multiecho series')

g_set = parser.add_argument_group('Settings for the RPN-signature calculation')
g_set.add_argument('--atlas', action='store',
                    default=os.path.abspath(os.path.join(os.path.dirname(__file__),"../data/atlas/MIST")),
                    help='MIST brain atlas directory')
g_set.add_argument('--keep_derivatives', '--keep_derivatives', '--keep_der', '--keep-der', action='store_true', default=False,
                         help='keep derivatives (preprocessed image files and regional timeseries)')

g_comp = parser.add_argument_group('Options for optimizing computations')
g_comp.add_argument('--bet_fract_int_thr', action='store', type=float, default=0.3,
                    help='fractional intensity threshold for FSL brain extraction')
g_comp.add_argument('--bet_vertical_gradient', action='store', type=float, default=-0.3,
                    help='vertical gradient value for FSL brain extraction')

g_perfm = parser.add_argument_group('Options to handle performance')
g_perfm.add_argument('--nthreads', '--n_cpus', '-n-cpus', action='store', type=int,
                     default=psutil.cpu_count(logical=True),
                     help='maximum number of threads across all processes')
#g_perfm.add_argument('--omp-nthreads', action='store', type=int, default=2, # ToDo: implement this
#                         help='maximum number of threads per-process')
g_perfm.add_argument('--mem_gb', '--mem-gb', action='store', default=psutil.virtual_memory().total/(1024*1024*1024), type=int,
                         help='upper bound memory limit for ROPN-signature processes')
g_perfm.add_argument('--template_2mm', '--template-2mm', '--2mm', action='store_true', default=False,
                         help='normalize to 2mm template (faster but less accurate prediction)')

opts = parser.parse_args()


globals._SinkDir_ = opts.output_dir
_MISTDIR_ = opts.atlas

# ToDo: make motion thresholds parametrisable

############ this part follows the BIDS-app specification ##############################################################

##############################
if (opts.template_2mm):
    globals._brainref = "/data/standard/MNI152_T1_2mm_brain.nii.gz"
    globals._headref = "/data/standard/MNI152_T1_2mm.nii.gz"
    globals._brainref_mask = "/data/standard/MNI152_T1_2mm_brain_mask_dil.nii.gz"
else:
    globals._brainref = "/data/standard/MNI152_T1_1mm_brain.nii.gz"
    globals._headref = "/data/standard/MNI152_T1_1mm.nii.gz"
    globals._brainref_mask = "/data/standard/MNI152_T1_1mm_brain_mask_dil.nii.gz"
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

print("Starting RPN-signature...")
print("Memory usage limit: " + str(opts.mem_gb) + "GB")
print("Number of CPUs used: " + str(opts.nthreads))

totalWorkflow = nipype.Workflow('RPN')
totalWorkflow.base_dir = '.'

########################
# parse command line args
bids_dir = opts.bids_dir

# create BIDS data grabber
datagrab = pe.Node(io.BIDSDataGrabber(), name='data_grabber')
datagrab.inputs.base_dir = bids_dir

# BIDS filtering
if opts.task_id and opts.echo_idx:
    datagrab.inputs.output_query['bold'] = dict(datatype='func', task=opts.t, echo=opts.echo_idx)
elif opts.task_id:
    datagrab.inputs.output_query['bold'] = dict(datatype='func', task=opts.t)
elif opts.echo_idx:
    datagrab.inputs.output_query['bold'] = dict(datatype='func', echo=opts.echo_idx)

print "Participants selected:"
if (opts.participant_label):
    datagrab.inputs.subject = opts.participant_label
    print opts.participant_label
else:
    print "all participants in dataset"

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
reorient_struct = pe.MapNode(fsl.utils.Reorient2Std(output_type='NIFTI_GZ'),
                      iterfield=['in_file'],
                      name="reorient_struct")
totalWorkflow.connect(datagrab, 'T1w', reorient_struct, 'in_file')

reorient_func = pe.MapNode(fsl.utils.Reorient2Std(output_type='NIFTI_GZ'),
                      iterfield=['in_file'],
                      name="reorient_func")
totalWorkflow.connect(datagrab, 'bold', reorient_func, 'in_file')

myanatproc = anatproc.AnatProc(stdreg=globals._regType_)
myanatproc.inputs.inputspec.bet_fract_int_thr = opts.bet_fract_int_thr #0.3  # feel free to adjust, a nice bet is important!
myanatproc.inputs.inputspec.bet_vertical_gradient = opts.bet_vertical_gradient #-0.3 # feel free to adjust, a nice bet is important!
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
def calculate_connectivity(ts_files, fd_files, _scrub_threshold_ = 0.15):
    # load FD data
    import pandas as pd
    import numpy as np
    import PAINTeR.connectivity as conn

    FD = []
    mean_FD = []
    median_FD = []
    max_FD = []
    perc_scrubbed = []
    for f in fd_files:
        fd = pd.read_csv(f, sep="\t").values.flatten()
        fd = np.insert(fd, 0, 0)
        FD.append(fd.ravel())
        mean_FD.append(fd.mean())
        median_FD.append(np.median(fd))
        max_FD.append(fd.max())
        perc_scrubbed.append(100 - 100 * len(fd) / len(fd[fd <= _scrub_threshold_]))

    df = pd.DataFrame()

    df['ts_file'] = ts_files
    df['fd_file'] = fd_files
    df['meanFD'] = mean_FD
    df['medianFD'] = median_FD
    df['maxFD'] = max_FD
    df['perc_scrubbed'] = perc_scrubbed

    # load timeseries data
    ts, labels = conn.load_timeseries(ts_files, df, scrubbing=True,
                                 scrub_threshold=_scrub_threshold_)
    features, cm = conn.connectivity_matrix(np.array(ts))

    mot_file = "motion.csv"
    df.to_csv(mot_file)

    return features, os.path.abspath(mot_file)


conn = pe.Node(util.Function(input_names=['ts_files', 'fd_files'],
                             output_names=['features', 'motion'],
                             function=calculate_connectivity), name="calculate_connectivity")
totalWorkflow.connect(extract_timeseries, 'outputspec.timeseries', conn, 'ts_files')
totalWorkflow.connect(myfuncproc, 'outputspec.FD', conn, 'fd_files')

def predict_pain_sensitivity(X, in_files):
    # load trained model
    from PAINTeR import global_vars
    from sklearn.externals import joblib
    import pandas as pd
    import os

    model = joblib.load(global_vars._RES_PRED_MOD_FIXED_)
    predicted = model.predict(X)

    df = pd.DataFrame()
    df['in_file'] = in_files
    df['RPN'] = predicted

    pred_file = "prediction.csv"
    df.to_csv(pred_file)

    return os.path.abspath(pred_file)


predict = pe.Node(util.Function(input_names=['X', 'in_files'],
                                output_names=['pred_file'],
                                function=predict_pain_sensitivity),
                  name='predict')

totalWorkflow.connect(conn, 'features', predict, 'X')
totalWorkflow.connect(datagrab, 'bold', predict, 'in_files')

ds_mot = pe.Node(interface=io.DataSink(), name='ds_mot')
ds_mot.inputs.regexp_substitutions = [("(\/)[^\/]*$", "summary.csv")]
ds_mot.inputs.base_directory = globals._SinkDir_
totalWorkflow.connect(conn, 'motion', ds_mot, 'motion_')

ds_pred = pe.Node(interface=io.DataSink(), name='ds_pred')
ds_pred.inputs.regexp_substitutions = [("(\/)[^\/]*$", "results.csv")]
ds_pred.inputs.base_directory = globals._SinkDir_
totalWorkflow.connect(predict, 'pred_file', ds_pred, 'RPN')


# RUN!

#totalWorkflow.write_graph('graph-orig.dot', graph2use='orig', simple_form=True)
#totalWorkflow.write_graph('graph-exec-detailed.dot', graph2use='exec', simple_form=False)
#totalWorkflow.write_graph('graph.dot', graph2use='colored')

#from nipype import config
#config.enable_resource_monitor()
from nipype.utils.profiler import log_nodes_cb
#import logging
#callback_log_path = 'run_stats.log'
#logger = logging.getLogger('callback')
#logger.setLevel(logging.DEBUG)
#handler = logging.FileHandler(callback_log_path)
#logger.addHandler(handler)

plugin_args = {'n_procs' : opts.nthreads,
               'memory_gb' : opts.mem_gb
              #'status_callback' : log_nodes_cb
               }
totalWorkflow.run(plugin='MultiProc', plugin_args=plugin_args)

# post-pipeline tasks
import shutil
if not opts.keep_derivatives:
    try:
        shutil.rmtree(globals._SinkDir_ + "/" + "regional_timeseries")
    except OSError as ex:
        print(ex)

    try:
        shutil.rmtree(globals._SinkDir_ + "/" + "func_preproc")
    except OSError as ex:
        print(ex)

    try:
        shutil.rmtree(globals._SinkDir_ + "/" + "anat_preproc")
    except OSError as ex:
        print(ex)

    try:
        shutil.rmtree(globals._SinkDir_ + "/" + "connectivity")
    except OSError as ex:
        print(ex)

    #try:
    #    os.remove(globals._SinkDir_ + "/" + "subjectsIDs.txt")
    #except OSError as ex:
    #    print(ex)

    try:
        os.remove(globals._SinkDir_ + "/" + "atlas.nii.gz")
    except OSError as ex:
        print(ex)

from shutil import copyfile

import glob
import shutil
for file in glob.glob('crash*'):
    print("crash files:")
    print(file)
    shutil.copy(file, globals._SinkDir_)


#import PUMI.utils.resource_profiler as rp
#rp.generate_gantt_chart('run_stats.log', cores=8)