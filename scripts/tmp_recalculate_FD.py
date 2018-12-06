#!/usr/bin/env python

#Todo: put this all into PUMI as a sub-workflow for FD calculation and logging

import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

import sys
import os
import nipype
import nipype.pipeline as pe
import nipype.interfaces.io as io
import nipype.algorithms.confounds as conf
import PUMI.utils.utils_convert as utils_convert
import PUMI.utils.utils_math as utils_math

totalWorkflow = nipype.Workflow('recalculate_FD')
totalWorkflow.base_dir = '.'

SinkDir = sys.argv[2]
QCDir = SinkDir + "/QC"

# create data grabber
datagrab = pe.Node(io.DataGrabber(outfields=['mc_params']), name='data_grabber')

datagrab.inputs.base_directory = os.getcwd()  # do we need this?
datagrab.inputs.template = "*"  # do we need this?
datagrab.inputs.field_template = dict(mc_params=sys.argv[1])  # specified by command line arguments
datagrab.inputs.sort_filelist = True

#sink: file - idx relationship!!
pop_id = pe.Node(interface=utils_convert.List2TxtFile,
                     name='pop_id')
pop_id.inputs.rownum = 0
pop_id.inputs.out_file = "subject_IDs.txt"
totalWorkflow.connect(datagrab, 'mc_params', pop_id, 'in_list')

calculate_FD = pe.MapNode(conf.FramewiseDisplacement(parameter_source='FSL', save_plot=True),
                                         iterfield=['in_file'],
                                         name='calculate_FD_Power'
                                         )
totalWorkflow.connect(datagrab, 'mc_params', calculate_FD, 'in_file')

# compute mean and max FD
meanFD = pe.MapNode(interface=utils_math.Txt2meanTxt,
                        iterfield=['in_file'],
                        name='meanFD')
meanFD.inputs.axis = 0  # global mean
meanFD.inputs.header = True  # global mean
totalWorkflow.connect(calculate_FD, 'out_file', meanFD, 'in_file')

maxFD = pe.MapNode(interface=utils_math.Txt2maxTxt,
                        iterfield=['in_file'],
                        name='maxFD')
maxFD.inputs.axis = 0  # global mean
maxFD.inputs.header = True
totalWorkflow.connect(calculate_FD, 'out_file', maxFD, 'in_file')

pop_FD = pe.Node(interface=utils_convert.List2TxtFileOpen,
                     name='pop_FD')
totalWorkflow.connect(meanFD, 'mean_file', pop_FD, 'in_list')

pop_FDmax = pe.Node(interface=utils_convert.List2TxtFileOpen,
                     name='pop_FDmax')
totalWorkflow.connect(maxFD, 'max_file', pop_FDmax, 'in_list')

# save data out with Datasink
ds_fd = pe.Node(interface=io.DataSink(), name='ds_pop_fd')
ds_fd.inputs.regexp_substitutions = [("(\/)[^\/]*$", "FD.txt")]
ds_fd.inputs.base_directory = SinkDir
totalWorkflow.connect(pop_FD, 'txt_file', ds_fd, 'pop')

# save data out with Datasink
ds_fd_max = pe.Node(interface=io.DataSink(), name='ds_pop_fd_max')
ds_fd_max.inputs.regexp_substitutions = [("(\/)[^\/]*$", "FD_max.txt")]
ds_fd_max.inputs.base_directory = SinkDir
totalWorkflow.connect(pop_FDmax, 'txt_file', ds_fd_max, 'pop')

# Save outputs which are important
ds_qc_fd = pe.Node(interface=io.DataSink(),
                        name='ds_qc_fd')
ds_qc_fd.inputs.base_directory = QCDir
ds_qc_fd.inputs.regexp_substitutions = [("(\/)[^\/]*$", "_FD.pdf")]
totalWorkflow.connect(calculate_FD, 'out_figure', ds_qc_fd, 'FD')

# save data out with Datasink
ds_FDs = pe.Node(interface=io.DataSink(), name='ds_fd')
ds_FDs.inputs.regexp_substitutions = [("(\/)[^\/]*$", "FD.txt")]
ds_FDs.inputs.base_directory = SinkDir
totalWorkflow.connect(calculate_FD, 'out_file', ds_FDs, 'FD')

totalWorkflow.run(plugin='MultiProc')