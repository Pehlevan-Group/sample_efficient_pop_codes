# sample_efficient_pop_codes

Code for the paper [Population Codes Enable Learning from Few Examples By Shaping Inductive Bias](https://www.biorxiv.org/content/10.1101/2021.03.30.437743v1).

The analysis of Mouse V1 responses can be reproduced with code in the [KPCA_EXPT](KPCA_EXPT) directory. Responses of V1 cells to [natural images](https://janelia.figshare.com/articles/dataset/Recordings_of_ten_thousand_neurons_in_visual_cortex_in_response_to_2_800_natural_images/6845348) and [grating stimuli](https://janelia.figshare.com/articles/dataset/Recordings_of_20_000_neurons_from_V1_in_response_to_oriented_stimuli/8279387) should be downloaded and placed in the [KPCA_EXPT](KPCA_EXPT) directory in subdirectories titled [natural_images](KPCA_EXPT/natural_images) and [grating_data](KPCA_EXPT/grating_data) respectively. Our utils.py file relied on some preprocessing code from [Mouseland Github](https://github.com/MouseLand/stringer-et-al-2019), which can be referenced for more details on data loading and preprocessing.

Code to reproduce results relating to the Gabor model is contained in [Gabor_Model](Gabor_Model) while the reservoir computing results are in [RNN](RNN). The tuning curve width results can be reproduced from the code in [Bandwidth_Expt](Bandwidth_Expt).
