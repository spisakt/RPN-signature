# PAINTeR
Predicting PAIN Thresholds based on Resting-state fMRI

Individual differences in pain perception are of key interest in both basic and clinical research as altered pain sensitivity is both a characteristic and a risk factor for many pain conditions.
It is, however, unclear how individual susceptibility to pain is reflected in the pain-free resting-state activity and functional connectivity of the brain.
Here, we identified and validated a network pattern in the pain-free resting-state functional brain connectome that is predictive of interindividual differences in pain sensitivity and provides insights into the contribution of regional connectivity changes.
Our method allows assessing the individual susceptibility to pain without applying any painful stimulation, as might be valuable in patients where reliable behavioural pain reports cannot be obtained. Additionally, as a direct, non-invasive readout of the supraspinal neural contribution to pain sensitivity, it may have broad implications for translational research and the development of analgesic treatment strategies.

This "research product" allows making predictions on the individual's pain sensitivity based on their resting-state fMRI measutrements.
Additional input needed: T1-weighted anatomical images.

The predictive model should be robust for variations in imaging sequences.
Neverthless, we have the following suggestions (which shouldn't be hard to meet):
 - high-resolution, "close-to-isovoxel" T1-weighted anatomical image, e.g. 1x1x1mm
 - 8-12 min long resting-state fMRI scan
 - whole brain coverage (actually, a few millimeter can be missed from the ventral part of the cerebellum, see the RPN regional connectivity map for reference)
