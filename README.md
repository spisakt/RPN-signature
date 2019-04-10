# PAINTeR
## Predicting PAIN Thresholds based on Resting-state fMRI

### Abstract
Individual differences in pain perception are of key interest in both basic and clinical research as altered pain sensitivity is both a characteristic and a risk factor for many pain conditions.
It is, however, unclear how individual susceptibility to pain is reflected in the pain-free resting-state activity and functional connectivity of the brain.
Here, we identified and validated a network pattern in the pain-free resting-state functional brain connectome that is predictive of interindividual differences in pain sensitivity and provides insights into the contribution of regional connectivity changes.
Our method allows assessing the individual susceptibility to pain without applying any painful stimulation, as might be valuable in patients where reliable behavioural pain reports cannot be obtained. Additionally, as a direct, non-invasive readout of the supraspinal neural contribution to pain sensitivity, it may have broad implications for translational research and the development of analgesic treatment strategies.

### Inputs
This "research product" allows making predictions on the individual's pain sensitivity based on their resting-state fMRI measutrements. To calculate the prediction, you need an additional input, the T1-weighted anatomical images.

- The predictive network-signature is to be found here:
https://github.com/spisakt/PAINTeR/blob/master/res/predictive_connections.csv

- The nodes of the predictive network (with nodel predictive strength):
https://github.com/spisakt/PAINTeR/blob/master/res/RPN_predictive_network_nodes.nii.gz

The predictive model should be robust for variations in imaging sequences.
Neverthless, we have the following suggestions (which shouldn't be hard to meet):
* Anatomical image:
 - high-resolution, "close-to-isovoxel" T1-weighted anatomical image, e.g. 1x1x1mm
* Functional image:
 - 8-12 min long resting-state fMRI scan
 - whole brain coverage (actually, a few millimeter can be missed from the ventral part of the cerebellum, see the RPN regional connectivity map for reference: https://github.com/spisakt/PAINTeR/blob/master/res/RPN_predictive_network_nodes.nii.gz)
 - TR around 2.5 sec (the model might be robust to this, though)
 - interleaved slice order
 - Approximately 3mm voxel
 
 ### Usage with docker
 *** Coming soon ***
 
 ### Running the source code
 1. image preprocessing and network calculation is based on our PUMI (https://github.com/spisakt/PUMI) python module (see page for installation and dependencies).
 2. Run pipeline script for image preprocessing and network calculation: https://github.com/spisakt/PAINTeR/blob/master/pipeline/pipeline_PAINTeR.py
 3. Adjust (if neccessary) and run https://github.com/spisakt/PAINTeR/blob/master/scripts/run_all.sh to perform the prediction.
 
 For a more detailed instruction, please contact Tamas Spisak (spisak.tms@uk-essen.de)
 
 
 
