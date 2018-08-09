#!/usr/bin/env python
import sys
# sys.path.append("/home/balint/Dokumentumok/phd/github/") #PUMI should be added to the path by install or by the developer
# az importalasnal az ~_preproc utan a .fajlnev-et kell megadni
import nipype
import nipype.pipeline as pe
# import the defined workflows from the anat_preproc folder
import nipype.interfaces.io as nio
import nipype.interfaces.fsl as fsl
import nipype.interfaces.afni as afni
import PUMI.AnatProc as anatproc
import PUMI.FuncProc as funcproc
# import the necessary workflows from the func_preproc folder
import PUMI.anat_preproc.Func2Anat as bbr
import PUMI.func_preproc.func2standard as transform
import PUMI.utils.utils_convert as utils_convert
import os
import PUMI.utils.globals as globals
import PUMI.connectivity.TimeseriesExtractor as tsext
import PUMI.connectivity.NetworkBuilder as nw
#import PUMI.utils.addimages as adding

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

##############################
#_regtype_ = globals._RegType_.FSL
_regtype_ = globals._RegType_.ANTS
##############################
# specify atlas for network construction:
# name of labelmap nii (or list of probmaps)
_ATLAS_FILE = '/Users/tspisak/data/atlases/MIST/Parcellations/MIST_122.nii.gz'
# a list of labels, where index+1 corresponds to the label in the labelmap
_ATLAS_LABELS = tsext.mist_labels(mist_directory='/Users/tspisak/data/atlases/MIST/', resolution="122")
# a list of labels, where index i corresponds to the module of the i+1th region, this is optional
_ATLAS_MODULES = tsext.mist_modules(mist_directory='/Users/tspisak/data/atlases/MIST/', resolution="122")
##############################


# create data grabber
datagrab = pe.Node(nio.DataGrabber(outfields=['func', 'struct']), name='data_grabber')

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
pop_id.inputs.filelist = False
ds_id = pe.Node(interface=nio.DataSink(), name='ds_pop_id')
ds_id.inputs.regexp_substitutions = [("(\/)[^\/]*$", "IDs.txt")]
ds_id.inputs.base_directory = globals._SinkDir_

# build the actual pipeline
reorient_struct = pe.MapNode(fsl.utils.Reorient2Std(),
                      iterfield=['in_file'],
                      name="reorient_struct")
reorient_func = pe.MapNode(fsl.utils.Reorient2Std(),
                      iterfield=['in_file'],
                      name="reorient_func")

myanatproc = anatproc.AnatProc(stdreg=_regtype_)
myanatproc.inputs.inputspec.bet_fract_int_thr = 0.3  # feel free to adjust, a nice bet is important!
myanatproc.inputs.inputspec.bet_vertical_gradient = -0.3 # feel free to adjust, a nice bet is important!
# try scripts/opt_bet.py to optimise these parameters

mybbr = bbr.bbr_workflow()
# Add arbitrary number of nii images wthin the same space. The default is to add csf and wm masks for anatcompcor calculation.
#myadding=adding.addimgs_workflow(numimgs=2)
add_masks = pe.MapNode(fsl.ImageMaths(op_string=' -add'),
                       iterfield=['in_file', 'in_file2'],
                       name="addimgs")

# TODO: erode compcor noise mask!!!!
erode_mask = pe.MapNode(fsl.ErodeImage(),
                        iterfield=['in_file'],
                        name="erode_compcor_mask")
# TODO: add skull voxels??

def pickindex(vec, i):
    return [x[i] for x in vec]

myfuncproc = funcproc.FuncProc()

#create atlas matching this space
resample_atlas = pe.Node(interface=afni.Resample(outputtype = 'NIFTI_GZ',
                                          in_file="/Users/tspisak/data/atlases/MIST/Parcellations/MIST_7.nii.gz",
                                          master=globals._FSLDIR_ + '/data/standard/MNI152_T1_3mm_brain.nii.gz'),
                         name='resample_atlas') #default interpolation is nearest neighbour

# standardize what you need
myfunc2mni = transform.func2mni(stdreg=_regtype_, carpet_plot="1_original", wf_name="func2mni_1")
#myfunc2mni_cc = transform.func2mni(stdreg=_regtype_, carpet_plot="2_cc", wf_name="func2mni_2_cc")
#myfunc2mni_cc_bpf = transform.func2mni(stdreg=_regtype_, carpet_plot="3_cc_bpf", wf_name="func2mni_3_cc_bpf")
#myfunc2mni_cc_bpf_cens = transform.func2mni(stdreg=_regtype_, carpet_plot="4_cc_bpf_cens", wf_name="func2mni_4_cc_bpf_cens")
myfunc2mni_cc_bpf_cens_mac = transform.func2mni(stdreg=_regtype_, carpet_plot="5_cc_bpf_cens_mac", wf_name="func2mni_5_cc_bpf_cens_mac")

###################

#create matrices
myextract = tsext.extract_timeseries()
myextract.inputs.inputspec.atlas_file = _ATLAS_FILE
myextract.inputs.inputspec.labels = _ATLAS_LABELS
myextract.inputs.inputspec.modules = _ATLAS_MODULES

measure = "partial correlation"
mynetmat = nw.build_netmat(wf_name=measure.replace(" ", "_"))
mynetmat.inputs.inputspec.measure = measure


totalWorkflow = nipype.Workflow('preprocess_all')
totalWorkflow.base_dir = '.'

# anatomical part and func2anat
totalWorkflow.connect([
    (datagrab, pop_id,
     [('func', 'in_list')]),
    (pop_id, ds_id,
     [('txt_file', 'subjects')]),
    (datagrab, reorient_struct,
     [('struct', 'in_file')]),
    (reorient_struct, myanatproc,
     [('out_file', 'inputspec.anat')]),
    (reorient_struct, mybbr,
     [('out_file', 'inputspec.skull')]),
    (datagrab, reorient_func,
     [('func', 'in_file')]),
    (reorient_func, mybbr,
     [('out_file', 'inputspec.func')]),
    (myanatproc, mybbr,
      [('outputspec.probmap_wm', 'inputspec.anat_wm_segmentation'),
       ('outputspec.probmap_csf', 'inputspec.anat_csf_segmentation'),
       ('outputspec.probmap_gm', 'inputspec.anat_gm_segmentation'),
       ('outputspec.probmap_ventricle', 'inputspec.anat_ventricle_segmentation')])

    ])

# functional part
totalWorkflow.connect([
    (reorient_func, myfuncproc,
     [('out_file', 'inputspec.func')]),
    (mybbr, add_masks,
     [('outputspec.ventricle_mask_in_funcspace','in_file'),
      ('outputspec.wm_mask_in_funcspace','in_file2')]),
    (add_masks, erode_mask,
     [('out_file','in_file')]),
    (erode_mask, myfuncproc,
     [('out_file','inputspec.cc_noise_roi')]),

    # push func to standard space
    (myfuncproc, myfunc2mni,
     [('outputspec.func_mc', 'inputspec.func'),
      ('outputspec.FD', 'inputspec.confounds')]),
    (mybbr, myfunc2mni,
     [('outputspec.func_to_anat_linear_xfm', 'inputspec.linear_reg_mtrx')]),
    (myanatproc, myfunc2mni,
     [('outputspec.anat2mni_warpfield', 'inputspec.nonlinear_reg_mtrx'),
      #('outputspec.std_template', 'inputspec.reference_brain'),
      ('outputspec.brain', 'inputspec.anat')]),
    (resample_atlas, myfunc2mni,
     [('out_file', 'inputspec.atlas')]),

    (myfuncproc, myfunc2mni_cc_bpf_cens_mac,
     [('outputspec.func_mc_nuis_bpf_cens_medang', 'inputspec.func'),
      ('outputspec.FD', 'inputspec.confounds')]),
    (mybbr, myfunc2mni_cc_bpf_cens_mac,
     [('outputspec.func_to_anat_linear_xfm', 'inputspec.linear_reg_mtrx')]),
    (myanatproc, myfunc2mni_cc_bpf_cens_mac,
     [('outputspec.anat2mni_warpfield', 'inputspec.nonlinear_reg_mtrx'),
      #('outputspec.std_template', 'inputspec.reference_brain'),
      ('outputspec.brain', 'inputspec.anat')]),
    (resample_atlas, myfunc2mni_cc_bpf_cens_mac,
     [('out_file', 'inputspec.atlas')])
    ])

# connect network analysis part
totalWorkflow.connect(myfunc2mni_cc_bpf_cens_mac, 'outputspec.func_std', myextract, 'inputspec.std_func')
totalWorkflow.connect(myextract, 'outputspec.timeseries_file', mynetmat, 'inputspec.timeseries')
totalWorkflow.connect(myextract, 'outputspec.reordered_modules', mynetmat, 'inputspec.modules')
totalWorkflow.connect(myextract, 'outputspec.relabelled_atlas_file', mynetmat, 'inputspec.atlas')

# RUN #
#from nipype.utils.profiler import log_nodes_cb
#import logging
#callback_log_path = 'run_stats.log'
#logger = logging.getLogger('callback')
#logger.setLevel(logging.DEBUG)
#handler = logging.FileHandler(callback_log_path)
#logger.addHandler(handler)
totalWorkflow.write_graph('graph-orig.dot', graph2use='orig', simple_form=True)
totalWorkflow.write_graph('graph-exec-detailed.dot', graph2use='exec', simple_form=False)
totalWorkflow.write_graph('graph.dot', graph2use='colored')
totalWorkflow.run(plugin='MultiProc')

#from nipype.utils.draw_gantt_chart import generate_gantt_chart
#generate_gantt_chart('run_stats.log', cores=8)
