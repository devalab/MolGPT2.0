# MolGPT2.0

This repository contains code and datasets for the paper "MolGPT2.0:
Multi-objective molecule generation via transformer encoder-decoder model and direct preference optimization."

---

`data` contains dataset(dockstring) used for training base models in the file `lck_dockstring_data1.csv`, while Preference data used for performing DPO is present in pickle file `PreferenceData_affinity.pkl` .

---

`src` contains the code to train the base model, perform DPO, and analyse results.

---

## How To Run The Code

1. Install required libraries using the `requirements.txt`. For dockstring, we recommend using conda from conda-forge, as this also installs correct version of `openbabel` which cannot be installed by pip currently.  
Use following commands for installing dockstring and other required packages 

        conda install -c conda-forge dockstring

        pip install -r requirements.txt

2. For training base model, run following command in `src` directory (*properties* argument determines on which properties model is conditioned, and *temp* determines temp used for sampling molecules ) : 

        python train_models.py --properties affinity --temp 1.0

This command trains an encoder - decoder model trained using conditioning properties, and samples molecules for some target values for that properties.

Base model and sampled molecules will be stored in checkpoints directory , inside the directory encoder_decoder_8_layer_affinity.  

For multiple property conditioning, add desired properties after --properties. eg.
        python train_models.py --properties affinity logps sas tpsas qeds

3. For performing DPO on base model, use following command : 

        python dpo_training.py --model_properties affinity --base_model_dir encoder_decoder_8_layer_affinity --output_dir dpo_encoder_decoder_8_layer_affinity

Here, the arguments are :
`--model_properties` : properties on which base model is conditioned
`--base_model_dir` : directory in which base model and results are stored
`--output_dir` : directory in which to store results for dpo model

4. For computing properties of generated molecules and generating results and plots for them, use the command :

        python analysis_and_plots.py --checkpoint_dir encoder_decoder_8_layer_affinity_logps --properties affinity logps --plot_targets affinity=9.0,6.0 logps=1.0,3.0

Here, the arguments are :
`--checkpoint_dir` : directory containing sampled molecules
`--properties` : properties on which sampled molecules were conditioned
`--plot_targets` : target values of property for which to plot kdes