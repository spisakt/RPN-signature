# A brain-based predictive signature of individual pain sensitivity
## *based on the Resting-state Pain susceptibility Network (RPN)*

Welcome to website of the RPN-signature, a resting-state network-based predictive signature of individual pain sensitivity.
The project is a joint effort of the Predictive Neuroimagiong Laboratory, ([PNI-lab](https://pni-lab.github.io), Tamas Spisak) and the [Bingel-lab](https://www.uk-essen.de/clinical_neurosciences_bingel/) (Ulrike Bingel), University Hospital Essen, Germany.

**_This site is under construction._**

## News
- **10.01.2020.** Our paper describing the RPN-signature has been published by Nature Communications:
*Spisak, T. et al. Pain-free resting-state functional brain connectivity predicts individual pain sensitivity. Nat Commun 11, 187 (2020)* [![DOI:10.1016/j.neuroimage.2018.09.078](https://zenodo.org/badge/DOI/10.1038/s41467-019-13785-z.svg)](https://doi.org/10.1038/s41467-019-13785-z)

## Contents
* [1. Summary](#summary)
* [2. Inputs of the the RPN-signature](#inputs)
* [3. Usage via Docker](#usage-with-docker)
* [4. Output](#output)
* [5. Running the source code (advanced)](#running-the-source-code)
* [6. Authors and Citation](#authors)

## Summary
Individual differences in pain percetheption are of key interest in both basic and clinical research as altered pain sensitivity is both a characteristic and a risk factor for many pain conditions. Individual susceptibility to pain is reflected in the pain-free resting-state activity and functional connectivity of the brain.
The RPN-signature is a network pattern in the pain-free resting-state functional brain connectome that is predictive of interindividual differences in pain sensitivity.
The RPN-signature allows assessing the individual susceptibility to pain without applying any painful stimulation, as might be valuable in patients where reliable behavioural pain reports cannot be obtained. Additionally, as a direct, non-invasive readout of the supraspinal neural contribution to pain sensitivity, it may have broad implications for translational research and the development of analgesic treatment strategies.

The Resting-state Pain susceptibility Network signature consists of an fMRI image preprocessing pipeline and a prediction based on (a linear combination of) specific functional connectivity values. Its output is a single number: a predicted pain sensitivity score, to be interpreted on the scale of the QST-based (Quantitative Sensory Testing) pain sensitivity score. See the paper (under review) for details.

- The list of predictive functional connections is to be found [here](https://github.com/spisakt/PAINTeR/blob/master/res/predictive_connections.csv).
(Note that a sufficient predictive performance is expected only with our dedicated preprocessing pipeline, see below)

- The nodes of the predictive network (with nodal predictive strength) can be downloaded [here](https://github.com/spisakt/PAINTeR/blob/master/res/RPN_predictive_network_nodes.nii.gz). Note that this map is not predictive on it's own, just a spatial map of the RPN-nodes.

[![Back to Top](https://iaibloggertips.files.wordpress.com/2014/05/e5406-back-to-top-button-for-blogger3-1.png)](#a-brain-based-predictive-signature-of-individual-pain-sensitivity)
## Inputs


This "research product" allows making predictions on the individual's pain sensitivity based on their resting-state fMRI measutrements. For the image preprocessing step, the T1-weighted anatomical images are additionally needed.

All data must be structures according to the **Brain Imaging Data Structure** [BIDS](http://bids.neuroimaging.io).
Consider validating your data with the [BIDS validator](https://bids-standard.github.io/bids-validator/) before running the RPN-signature.

The predictive model should be robust for variations in imaging sequences. Neverthless, we have the following suggestions (which shouldn't be hard to meet):

### In general:
- 3T field strength

### Anatomical image:
- high-resolution, "close-to-isovoxel" T1-weighted anatomical image, e.g. 1x1x1mm

### Functional image:
- 8-12 min long resting-state fMRI scan
- whole brain coverage (actually, a few millimeter can be missed from the ventral part of the cerebellum, see the RPN regional connectivity map for reference: [https://github.com/spisakt/PAINTeR/blob/master/res/RPN_predictive_network_nodes.nii.gz]
- TR around 2.5 sec (the model might be robust to this, though)
- interleaved slice order
- approximately 3mm voxel
- carefull fixation to prevent motion artifacts

[![Back to Top](https://iaibloggertips.files.wordpress.com/2014/05/e5406-back-to-top-button-for-blogger3-1.png)](#a-brain-based-predictive-signature-of-individual-pain-sensitivity)

## Usage with docker

The usage of the RPN-siganture with Docker is simple and platform-independent.
You can run it like any other [BIDS-app](http://bids-apps.neuroimaging.io/tutorial/).

1. Get the Docker Engine (https://docs.docker.com/engine/installation/)
2. Have your data organized in BIDS (get BIDS specification, see BIDS paper).
3. Validate your data (http://incf.github.io/bids-validator/). You can safely use the BIDS-validator since no data is uploaded to the server, works locally in your browser.

4. Have a look at the help page to test if it works.
It will start to download the docker image from docker hub (approximately 5.3Gb).
```bash
docker run -it tspisak/rpn-signature:latest -h
```

5. Run it by mounting and specifying your BIDS directory, output directory and level of analysis, like for any other BIDS-app.
E.g.:
```bash
docker run -it --rm -v /data/nii-bids/:/data:ro -v /data/nii-bids/derivatives:/out \
tspisak/rpn-signature:latest /data /out participant
```

_**NOTE 1** Have a look at the help, there are some useful command line options:_

E.g.:
```bash
docker run -it --rm -v /data/nii-bids/:/data:ro -v /data/nii-bids/derivatives:/out \
tspisak/rpn-signature:latest /data /out participant \
--participant_label  001 002 003 005 008 013 021 034 --mem_gb 10 --nthreads 7 --2mm
```

_**NOTE 2** Output directory must be specified as an absolute path._

_**NOTE 3** Note that the --2mm command line option performs spatial co-registration to a 2mm-resolution template (instead of 1mm), which is much faster (total running time is approximately 50 min instead of 8 hours per subject), but was not validasted and gives slighly different (preassumably less accurate) predictions._

_**NOTE 4** Make sure to configure Docker's resource availability to take adavantage of parallell processing._

_**NOTE 5** Make sure to have enough free space for storing temporary files (1.5GB per subject)._

_**NOTE 6** Consider using the option --keep_derivatives, if you need the timeseries and connectivity data for further processing._

_**NOTE 7** Do quality checking (see below) before using the predicted values and adjust brain extraction parameters with the options --bet_fract_int_thr and --bet_vertical_gradient if neccessary._

[![Back to Top](https://iaibloggertips.files.wordpress.com/2014/05/e5406-back-to-top-button-for-blogger3-1.png)](#a-brain-based-predictive-signature-of-individual-pain-sensitivity)

## Output

The main output of the RPN-signature is the RPN-score, a prediction of the composite pain sensitivity score of (Zunhammer et al., 2016).
Click [here](https://raw.githack.com/spisakt/RPN-signature/master/notebooks/Supplementary_Analysis_1.html) for more information on the composite score and its feasability as prediction target.

Additionally, the RPN-signature has a rich outpout of quality-check images and, if *--keep_derivatives* is specified, it outputs the processed image files and can be used as an fMRI processing workflow.

```
output_directory/
│    RPNresults.csv                CSV-file containing the predicted pain sensitivity scores
│    subjectsID.txt                text file linking data files to QC indices
└─── QC/                           directory for quality check images
│   └─── anat2mni/                 standardized anatomical image + the standard template
│   └─── brain_extraction/         anatomical image + the result of brain extraction 
│   └─── brain_extraction_func/    functional image + the result of brain extraction 
│   └─── carpet_plots/             carpet plots of preproicessing stages
│   └─── compcor_noiseroi/         aCompCor noise ROI overlaid on the functional image
│   └─── FD/                       framewise displacement plots
│   └─── func2anat/                functional image in anatomical space + anatomical image 
│   └─── func2mni/                 functional image in stnadard space + standard template
│   └─── motion_correction/        rotational and translational motion estimates
│   └─── regional_timeseries/      carpet plot of the atlas-based regional timeseries
│   └─── timeseries/               mean global signal timeseries of preprocessing stages
│   └─── tissue_segmentation/      tissue segmentation maximum probability images
:
:    if --keep_derivatives is specified:
:   
:    atlas.nii.gz                  brain atlas (MIST122) to define ROIs
:... anat_preproc/                 anatomical derivatives
:   :... anat2mni_std/             standard-space anatomical image
:   :... anat2mni_warpfield/       warpinf-field for standardisation (contains all steps)
:   :... bet_brain/                brain extracted anatomical image
:   :... brain_mask/               anatomical brain mask
:   :... fast_csf/                 CSF probability map
:   :... fast_gm/                  grey matter probability map
:   :... fast_wm/                  white matter probability map
:... func_preproc/                 functional derivatives
:   :    popFD_max.txt             mean FD values per subject
:   :    popFD.txt                 max FD values per subject
:   :    pop_percent_scrubbed.txt  percent of volumes scrubbed per subject
:   :... bet_brain/                brain extracted functional image
:   :... brain_mask/               functional brain mask
:   :... FD_scrubbed/              FD timeseries after scrubbing
:   :... mc_fd/                    FD timeseries
:   :... mc_frist24/               Friston-24 expansion of motion parameters
:   :... mc_func/                  motion corrected funcrtional image
:   :... mc_par/                   6 motion parameters (3 rotation, 3 translation)
:   :... mc_rms/                   root mean squared motion estimates
:... regional-timeseries           regional timeseries in tab separated format

```

[![Back to Top](https://iaibloggertips.files.wordpress.com/2014/05/e5406-back-to-top-button-for-blogger3-1.png)](#a-brain-based-predictive-signature-of-individual-pain-sensitivity)
## Running the source code


- Running the RPN-signature from source can be tricky for non-developers, because a lot of dependencies must be resolved and installtion is undocumented.
- Image preprocessing and network calculation is based on our PUMI (https://github.com/spisakt/PUMI) python module (see page for installation and dependencies), which has to be installed first.
- Run pipeline script for image preprocessing and network calculation: [https://github.com/spisakt/PAINTeR/blob/master/pipeline/pipeline_PAINTeR-BIDS.py]
- For help, please contact Tamas Spisak (spisak.tms@uk-essen.de)

[![Back to Top](https://iaibloggertips.files.wordpress.com/2014/05/e5406-back-to-top-button-for-blogger3-1.png)](#a-brain-based-predictive-signature-of-individual-pain-sensitivity)
## Authors


Tamas Spisak<sup>1</sup>, Balint Kincses<sup>2</sup>, Frederik Schlitt<sup>1</sup>, Matthias Zunhammer<sup>1</sup>, Tobias Schmidt-Wilcke<sup>2</sup>, Zsigmond Tamas Kincses<sup>2</sup>, Ulrike Bingel<sup>1</sup>

1.	Department of Neurology, University Hospital Essen, Essen, Germany
2.	Faculty of Medicine, Ruhr-University Bochum, Bochum, Germany
3.	Department of Neurology, University of Szeged, Szeged, Hungary

## Citation:
_Tamas Spisak, Balint Kincses, Frederik Schlitt, Matthias Zunhammer, Tobias Schmidt-Wilcke, Zsigmond Tamas Kincses, Ulrike Bingel, Pain-free resting-state functional brain connectivity predicts individual pain sensitivity, Nat Commun 11, 187 (2020)._
[![DOI:10.1016/j.neuroimage.2018.09.078](https://zenodo.org/badge/DOI/10.1038/s41467-019-13785-z.svg)](https://doi.org/10.1038/s41467-019-13785-z)

[![GitHub license](https://img.shields.io/github/license/spisakt/RPN-signature.svg)](https://github.com/spisakt/RPN-signature/blob/master/LICENSE)
[![GitHub release](https://img.shields.io/github/release/spisakt/RPN-signature.svg)](https://github.com/spisakt/RPN-signature/releases/)
[![CircleCI](https://circleci.com/gh/spisakt/pTFCE.svg?style=svg)](https://circleci.com/gh/spisakt/ptfce)
![CircleCI2](https://img.shields.io/circleci/project/github/RedSparr0w/node-csgo-parser.svg)
![Docker Build](https://img.shields.io/docker/cloud/build/tspisak/rpn-signature.svg)
![Docker Pulls](https://img.shields.io/docker/pulls/tspisak/rpn-signature.svg)
[![GitHub issues](https://img.shields.io/github/issues/spisakt/RPN-signature.svg)](https://GitHub.com/spisakt/RPN-signature/issues/)
[![GitHub issues-closed](https://img.shields.io/github/issues-closed/spisakt/RPN-signature.svg)](https://GitHub.com/spisakt/RPN-signature/issues?q=is%3Aissue+is%3Aclosed)

Maintained by the [PNI-lab](https://pni-lab.github.io).

[![Back to Top](https://iaibloggertips.files.wordpress.com/2014/05/e5406-back-to-top-button-for-blogger3-1.png)](#a-brain-based-predictive-signature-of-individual-pain-sensitivity)
