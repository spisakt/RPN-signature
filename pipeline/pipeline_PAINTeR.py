#!/usr/bin/env python
# This is a pipeline doing resting-state fMRI preprocessing as done in the PAINTeR study
# https://github.com/spisakt/PAINTeR
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

# - computing the pain sensitivity score from the matrix is not included here and can be found in
# https://github.com/spisakt/PAINTeR

# start it like:
# /Users/tspisak/src/PUMI/scripts/pipeline_PAINTeR.py "/Users/tspisak/projects/PAINTeR/data/nii-szeged/data/PA*/highres.nii" "/Users/tspisak/projects/PAINTeR/data/nii-szeged/data/PA*/rest.nii" /Users/tspisak/projects/PAINTeR/res/szeged /Users/tspisak/data/atlases/MIST/


import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

import sys
# sys.path.append("/home/balint/Dokumentumok/phd/github/") #PUMI should be added to the path by install or by the developer
# az importalasnal az ~_preproc utan a .fajlnev-et kell megadni
import nipype
import nipype.pipeline as pe
# import the defined workflows from the anat_preproc folder
import nipype.interfaces.io as io
import nipype.interfaces.fsl as fsl
import nipype.interfaces.afni as afni
import nipype.interfaces.utility as utility
import PUMI.AnatProc as anatproc
import PUMI.FuncProc as funcproc
import PUMI.func_preproc.Compcor as cc
# import the necessary workflows from the func_preproc folder
import PUMI.anat_preproc.Func2Anat as bbr
import PUMI.func_preproc.func2standard as transform
import PUMI.utils.utils_convert as utils_convert
import os
import PUMI.utils.globals as globals
import PUMI.connectivity.TimeseriesExtractor as tsext
import nipype.interfaces.nilearn as learn
import PUMI.utils.QC as qc
import PUMI.connectivity.NetworkBuilder as nw


# parse command line arguments
if (len(sys.argv) <= 2):
    print("Please specify command line arguments!")
    print("Usage:")
    print(sys.argv[0] + " <\"highres_data_template\"> <\"func_data_template\"> [results_sink_directory]")
    print("Example:")
    print(sys.argv[0] + " \"highres_data/subject_*.nii.gz\" \"func_data/subject_*.nii.gz\"")
    quit()

if (len(sys.argv) > 3):
    globals._SinkDir_ = sys.argv[3]

if (len(sys.argv) > 4):
    _MISTDIR_=sys.argv[4]
else:
    _MISTDIR_ = '/home/analyser/Documents/mistatlases/'

##############################
globals._brainref="/data/standard/MNI152_T1_1mm_brain.nii.gz"
globals._headref="/data/standard/MNI152_T1_1mm.nii.gz"
globals._brainref_mask="/data/standard/MNI152_T1_1mm_brain_mask_dil.nii.gz"
##############################
_refvolplace_ = globals._RefVolPos_.first

#
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

totalWorkflow = nipype.Workflow('pumi')
totalWorkflow.base_dir = '.'

# create data grabber
datagrab = pe.Node(io.DataGrabber(outfields=['func', 'struct']), name='data_grabber')

datagrab.inputs.base_directory = os.getcwd()  # do we need this?
datagrab.inputs.template = "*"  # do we need this?
datagrab.inputs.field_template = dict(func=sys.argv[2],
                                      struct=sys.argv[1])  # specified by command line arguments
datagrab.inputs.sort_filelist = True

# sink: file - idx relationship!!
pop_id = pe.Node(interface=utils_convert.List2TxtFile,
                     name='pop_id')
pop_id.inputs.rownum = 0
pop_id.inputs.out_file = "subject_IDs.txt"
totalWorkflow.connect(datagrab, 'func', pop_id, 'in_list')

ds_id = pe.Node(interface=io.DataSink(), name='ds_pop_id')
ds_id.inputs.regexp_substitutions = [("(\/)[^\/]*$", "IDs.txt")]
ds_id.inputs.base_directory = globals._SinkDir_
totalWorkflow.connect(pop_id, 'txt_file', ds_id, 'subjects')

# build the actual pipeline
reorient_struct = pe.MapNode(fsl.utils.Reorient2Std(output_type='NIFTI'),
                      iterfield=['in_file'],
                      name="reorient_struct")
totalWorkflow.connect(datagrab, 'struct', reorient_struct, 'in_file')

reorient_func = pe.MapNode(fsl.utils.Reorient2Std(output_type='NIFTI'),
                      iterfield=['in_file'],
                      name="reorient_func")
totalWorkflow.connect(datagrab, 'func', reorient_func, 'in_file')

myanatproc = anatproc.AnatProc(stdreg=globals._regType_)
myanatproc.inputs.inputspec.bet_fract_int_thr = 0.4 #0.3  # feel free to adjust, a nice bet is important!
myanatproc.inputs.inputspec.bet_vertical_gradient = -0.1 #-0.3 # feel free to adjust, a nice bet is important!
# try scripts/opt_bet.py to optimise these parameters
totalWorkflow.connect(reorient_struct, 'out_file', myanatproc, 'inputspec.anat')

mybbr = bbr.bbr_workflow()
totalWorkflow.connect(myanatproc, 'outputspec.brain', mybbr, 'inputspec.skull') #ToDo ready: rather input the brain extracted here?
totalWorkflow.connect(reorient_func, 'out_file', mybbr, 'inputspec.func')
totalWorkflow.connect(myanatproc, 'outputspec.probmap_wm', mybbr, 'inputspec.anat_wm_segmentation')
totalWorkflow.connect(myanatproc, 'outputspec.probmap_csf', mybbr, 'inputspec.anat_csf_segmentation')
totalWorkflow.connect(myanatproc, 'outputspec.probmap_gm', mybbr, 'inputspec.anat_gm_segmentation')
totalWorkflow.connect(myanatproc, 'outputspec.probmap_ventricle', mybbr, 'inputspec.anat_ventricle_segmentation')

# Add arbitrary number of nii images wthin the same space. The default is to add csf and wm masks for anatcompcor calculation.
#myadding=adding.addimgs_workflow(numimgs=2)

# ToDo_ready: put compcor-related mask handling into a nested pipeline
# TODO_ready: erode compcor noise mask!!!!
# NOTE: more CSF voxels are retained for compcor when only WM signal is eroded and csf is added to it

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


# compute connectivity
#measure = "partial correlation"
#mynetmat = nw.build_netmat(wf_name=measure.replace(" ", "_"))
#mynetmat.inputs.inputspec.measure = measure

#totalWorkflow.connect(extract_timeseries, 'outputspec.timeseries', mynetmat, 'inputspec.timeseries')
#totalWorkflow.connect(pickatlas, 'outputspec.reordered_modules', mynetmat, 'inputspec.modules')
#totalWorkflow.connect(pickatlas, 'outputspec.relabeled_atlas', mynetmat, 'inputspec.atlas')

# Extract timeseries
extract_timeseries_scrubbed = tsext.extract_timeseries_nativespace(SinkTag="connectivity_scrubbed", wf_name="extract_timeseries_nativespace_scribbed")
totalWorkflow.connect(pickatlas, 'outputspec.relabeled_atlas', extract_timeseries_scrubbed, 'inputspec.atlas')
totalWorkflow.connect(pickatlas, 'outputspec.reordered_labels', extract_timeseries_scrubbed, 'inputspec.labels')
totalWorkflow.connect(pickatlas, 'outputspec.reordered_modules', extract_timeseries_scrubbed, 'inputspec.modules')
totalWorkflow.connect(myanatproc, 'outputspec.brain', extract_timeseries_scrubbed, 'inputspec.anat')
totalWorkflow.connect(mybbr, 'outputspec.anat_to_func_linear_xfm', extract_timeseries_scrubbed, 'inputspec.inv_linear_reg_mtrx')
totalWorkflow.connect(myanatproc, 'outputspec.mni2anat_warpfield', extract_timeseries_scrubbed, 'inputspec.inv_nonlinear_reg_mtrx')
totalWorkflow.connect(mybbr, 'outputspec.gm_mask_in_funcspace', extract_timeseries_scrubbed, 'inputspec.gm_mask')
totalWorkflow.connect(myfuncproc, 'outputspec.func_preprocessed_scrubbed', extract_timeseries_scrubbed, 'inputspec.func')
totalWorkflow.connect(myfuncproc, 'outputspec.FD', extract_timeseries_scrubbed, 'inputspec.confounds')


# compute connectivity
#measure = "partial correlation"
#mynetmat_scrubbed = nw.build_netmat(SinkTag="connectivity_scrubbed", wf_name=measure.replace(" ", "_") + "_scrubbed")
#mynetmat_scrubbed.inputs.inputspec.measure = measure

#totalWorkflow.connect(extract_timeseries_scrubbed, 'outputspec.timeseries', mynetmat_scrubbed, 'inputspec.timeseries')
#totalWorkflow.connect(pickatlas, 'outputspec.reordered_modules', mynetmat_scrubbed, 'inputspec.modules')
#totalWorkflow.connect(pickatlas, 'outputspec.relabeled_atlas', mynetmat_scrubbed, 'inputspec.atlas')

# RUN!

totalWorkflow.write_graph('graph-orig.dot', graph2use='orig', simple_form=True)
totalWorkflow.write_graph('graph-exec-detailed.dot', graph2use='exec', simple_form=False)
totalWorkflow.write_graph('graph.dot', graph2use='colored')

from nipype import config
config.enable_resource_monitor()
from nipype.utils.profiler import log_nodes_cb
import logging
callback_log_path = 'run_stats.log'
logger = logging.getLogger('callback')
logger.setLevel(logging.DEBUG)
handler = logging.FileHandler(callback_log_path)
logger.addHandler(handler)

plugin_args = {'n_procs' : 8,
               'memory_gb' : 8,
              'status_callback' : log_nodes_cb
               }
totalWorkflow.run(plugin='MultiProc', plugin_args=plugin_args)

import PUMI.utils.resource_profiler as rp
rp.generate_gantt_chart('run_stats.log', cores=8)