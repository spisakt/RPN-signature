# A brain-based predictive signature of individual pain sensitivity
## *based on the Resting-state Pain susceptibility Network (RPN)*

Welcome to website of the RPN-signature!

**_This site is under construction._**

## Contents
* [1. Summary](#summary)
* [2. Inputs of the the RPN-signature](#inputs)
* [3. Usage via Docker](#usage-with-docker)
* [4. Running the source code (advanced)](#running-the-source-code)
* [5. Authors and Citation](#authors)

## Summary
Individual differences in pain percetheption are of key interest in both basic and clinical research as altered pain sensitivity is both a characteristic and a risk factor for many pain conditions. Individual susceptibility to pain is reflected in the pain-free resting-state activity and functional connectivity of the brain.
The RPN-signature is a network pattern in the pain-free resting-state functional brain connectome that is predictive of interindividual differences in pain sensitivity.
The RPN-signature allows assessing the individual susceptibility to pain without applying any painful stimulation, as might be valuable in patients where reliable behavioural pain reports cannot be obtained. Additionally, as a direct, non-invasive readout of the supraspinal neural contribution to pain sensitivity, it may have broad implications for translational research and the development of analgesic treatment strategies.

The Resting-state Pain susceptibility Network signature consists of an fMRI image preprocessing pipeline and a prediction based on (a linear combination of) specific functional connectivity values. Its output is a single number: a predicted pain sensitivity score, to be interpreted on the scale of the QST-based (Quantitative Sensory Testing) pain sensitivity score. See the paper (under review) for details.

- The list of predictive functional connections is to be found [here](https://github.com/spisakt/PAINTeR/blob/master/res/predictive_connections.csv).
(Note that a sufficient predictive performance is expected only with our dedicated preprocessing pipeline, see below)

- The nodes of the predictive network (with nodal predictive strength) can be downloaded [here](https://github.com/spisakt/PAINTeR/blob/master/res/RPN_predictive_network_nodes.nii.gz). Note that this map is not predictive on it's own, just a spatial map of the RPN-nodes.

## Inputs [![Back to Top](https://iaibloggertips.files.wordpress.com/2014/05/e5406-back-to-top-button-for-blogger3-1.png)](#a-brain-based-predictive-signature-of-individual-pain-sensitivity)

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

## Usage with docker
[Back to Contents](#contents)

The usage of the RPN-siganture with Docker is simple and platform-independent.
You can run it like any other [BIDS-app](http://bids-apps.neuroimaging.io/tutorial/).

1. Get the Docker Engine (https://docs.docker.com/engine/installation/)
2. Have your data organized in BIDS (get BIDS specification, see BIDS paper).
3. Validate your data (http://incf.github.io/bids-validator/). You can safely use the BIDS-validator since no data is uploaded to the server, works locally in your browser.

4. Have a look at the help page to test if it works.
It will start to download the docker image from docker hub (approximately 2Gb).
```bash
docker run -it tspisak/rpn-signature:latest -h
```

5. Run it by mounting and specifying your BIDS directory, output directory and level of analysis, like for any other BIDS-app.

E.g.:
```bash
docker run -it --rm -v /data/nii-bids/:/data:ro -v /data/nii-bids/derivatives:/out tspisak/rpn-signature:latest /data /out participant
```

Have a look at the help, there are some useful command line options:

E.g.:
```bash
docker run -it --rm -v /data/nii-bids/:/data:ro -v /data/nii-bids/derivatives:/out tspisak/rpn-signature:latest /data /out participant --participant_label  004 006 007 008 009 011 --mem_gb 10 --nthreads 7 --2mm
```

Note that the --2mm command line option performs spatial co-registration to a 2mm-resolution template (instead of 1mm), which is much faster (total running time is approximately 50 min instead of 8 hours / subject), but was not validasted and gives slighly different (preassumably less accurate) predictions.

## Running the source code
[Back to Contents](#contents)

- Running the RPN-signature from source can be tricky for non-developers, because a lot of dependencies must be resolved and installtion is undocumented.
- Image preprocessing and network calculation is based on our PUMI (https://github.com/spisakt/PUMI) python module (see page for installation and dependencies), which has to be installed first.
- Run pipeline script for image preprocessing and network calculation: [https://github.com/spisakt/PAINTeR/blob/master/pipeline/pipeline_PAINTeR-BIDS.py]
- For help, please contact Tamas Spisak (spisak.tms@uk-essen.de)

## Authors
| |[Back to Contents](#contents)|

Tamas Spisak<sup>1</sup>, Balint Kincses<sup>2</sup>, Frederik Schlitt<sup>1</sup>, Matthias Zunhammer<sup>1</sup>, Tobias Schmidt-Wilcke<sup>2</sup>, Zsigmond Tamas Kincses<sup>2</sup>, Ulrike Bingel<sup>1</sup>

1.	Department of Neurology, University Hospital Essen, Essen, Germany
2.	Faculty of Medicine, Ruhr-University Bochum, Bochum, Germany
3.	Department of Neurology, University of Szeged, Szeged, Hungary

## Citation:
_Tamas Spisak, Balint Kincses, Frederik Schlitt, Matthias Zunhammer, Tobias Schmidt-Wilcke, Zsigmond Tamas Kincses, Ulrike Bingel, Pain-free resting-state functional brain connectivity predicts individual pain sensitivity, under review, 2019._

[![Open Source Love svg1](https://badges.frapsoft.com/os/v1/open-source.svg?v=103)](https://github.com/spisakt/RPN-signature)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/spisakt/RPN-signature/graphs/commit-activity)
[![GitHub license](https://img.shields.io/github/license/spisakt/RPN-signature.svg)](https://github.com/spisakt/RPN-signature/blob/master/LICENSE)
[![GitHub release](https://img.shields.io/github/release/spisakt/RPN-signature.svg)](https://github.com/spisakt/RPN-signature/releases/)
[![GitHub issues](https://img.shields.io/github/issues/spisakt/RPN-signature.svg)](https://GitHub.com/spisakt/RPN-signature/issues/)
[![GitHub issues-closed](https://img.shields.io/github/issues-closed/spisakt/RPN-signature.svg)](https://GitHub.com/spisakt/RPN-signature/issues?q=is%3Aissue+is%3Aclosed)
[![HitCount](http://hits.dwyl.io/spisakt/RPN-signature.svg)](http://hits.dwyl.io/spisakt/RPN-signature)

2019.
[Back to Contents](#contents)
