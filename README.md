# MolGPT2.0

This repository contains code and datasets for the paper "MolGPT2.0:
Multi-objective molecule generation via transformer encoder-decoder model and direct preference optimization."

---

`data` contains dataset(dockstring) used for training base models in the file `lck_dockstring_data1.csv`, while Preference data used for performing DPO is present in pickle file `PreferenceData_affinity.pkl` .

---

`src` contains the code to train the base model, perform DPO, and analyse results.

---

## How To Run The Code

### 1. Installing required packages/libraries
Install required libraries using the `requirements.txt`. For dockstring, we recommend using conda from conda-forge, as this also installs correct version of `openbabel` which cannot be installed by pip currently.  
Use following commands for installing dockstring and other required packages 

        conda install -c conda-forge dockstring

        pip install -r requirements.txt

### 2. Training Models
For training base model, run following command in `src` directory (*properties* argument determines on which properties model is conditioned) : 

        python train_models.py --properties affinity

This command trains an encoder - decoder model trained for conditioning on affinity. You can replace affinity here with whatever property (or properties) you want to condition on.

Base model will be stored in checkpoints directory (as best_model.pt), inside the subdirectory *encoder_decoder_{conditioning_properties}*. (by default, can be changed using --checkpoint_dir argument)  

For multiple property conditioning, add desired properties after --properties. eg.

        python train_models.py --properties affinity logps sas tpsas qeds
    
You can also provide following arguments : 
    
- --checkpoint_dir : subdirectory inside checkpoints directory where to save model(by default it is stored in encoder_decoder_{property_names})
- --epochs
- --batch_size
- --d_model : Token embedding dimension
- --n_heads : Number of attention heads
- --n_layers : Number of encoder and decoder layers
- --hidden_units : Number of hidden units in feedforward layers
- --lr : Learning rate
- --temp : Sampling temperature

### 3. Sampling Molecules
For generating(sampling) molecules using trained models, run following command inside `src` directory : 

        python3 generate_molecules.py --properties {conditioning_properties}

This command stores sampled molecules in *generated_molecules_temp_{temp}.pkl* in checkpoint_dir, and also saves generation metrics like validity, novelty etc. in the file *final_metrics_temp_{temp}.csv*.

You can also provide following arguments : 

- --checkpoint_dir : Directory where trained model is stored.
- --affinity_targets : Target affinity values for molecule generation (similarly for other properties, eg --logp_targets etc. )
- --num_samples : Number of molecules to generate per target property combination
- arguments like batch size,d_model,n_heads etc. describing model architecture.


### 4. Performing Direct Preference Optimisation
For performing DPO on base model, use following command : 

        python dpo_training.py --model_properties affinity --base_model_dir encoder_decoder_8_layer_affinity --output_dir dpo_encoder_decoder_8_layer_affinity

Here, the arguments are :

- --model_properties : properties on which base model is conditioned
- --base_model_dir : directory in which base model and results are stored
- --output_dir : directory in which to store results for dpo model

### 5. Computing properties of generated molecules
For computing properties of generated molecules and generating results and plots for them, use the command :

        python analysis_and_plots.py --checkpoint_dir encoder_decoder_8_layer_affinity_logps --properties affinity logps --plot_targets affinity=9.0,6.0 logps=1.0,3.0

Here, the arguments are :
- --checkpoint_dir : directory containing sampled molecules
- --properties : properties on which sampled molecules were conditioned
- --plot_targets : target values of property for which to plot kdes