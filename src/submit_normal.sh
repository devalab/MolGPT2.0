#!/bin/bash

#SBATCH --job-name=MultiGPU-TestJob
#SBATCH -A research
#SBATCH -p long
#SBATCH -c 40
#SBATCH -w gnode041
#SBATCH --mem-per-cpu=2G
#SBATCH --time=4-00:00:00
#SBATCH --gres=gpu:4
#SBATCH --mail-user=kanakala.ganesh@research.iiit.ac.in
#SBATCH --mail-type=ALL
#SBATCH --output=outputs/op_file_all_props_rawpreferencedata.txt

echo "Starting ..."

# python ConvertSMILESTo3D.py
#python DPO_Finetuning_MOSES_Decoder.py

# python bindingAffinity_dockstring_multiproperty_decoder_arc.py
# python bindingAffinity_dockstring_multiproperty.py
# python BindingAffinity_Distribution.py 0.35
# python DUMP_RESULTS_DOCKSTRING.py
# python CreateRawPreferenceDataDockstring.py
# echo "Raw preference data done"
python CreatePreferenceDataAffinities.py
echo "Preference data affinities done"
# python CreatePreferenceDataFullProperties.py
# echo "Preference data full properties done"
