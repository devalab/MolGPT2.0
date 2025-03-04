#!/bin/bash

#SBATCH --job-name=DPO
#SBATCH -A plafnet2
#SBATCH -p plafnet2
#SBATCH -c 20
#SBATCH --mem-per-cpu=2G
#SBATCH --time=4-00:00:00
#SBATCH --gres=gpu:0
#SBATCH --mail-user=kanakala.ganesh@research.iiit.ac.in
#SBATCH --mail-type=ALL
SBATCH --output=outputs/op_file_IPO.txt
#SBATCH -w gnode116

echo "Starting ..."

# jupyter lab
#python bindingAffinity_dockstring_multiproperty_decoder_arc_EGFR.py 
# python bindingAffinity_dockstring_multiproperty_decoder_arc.py

#python DUMP_RESULTS_DOCKSTRING.py
# python DockingMolecules.py

# python CreateRawPreferenceDataDockstring.py
# echo "Raw preference data done"
# python CreatePreferenceDataAffinities.py
# echo "Preference data affinities done"
# python CreatePreferenceDataFullProperties.py
# echo "Preference data full properties done"

# python PlotAffinityDistribution.py
# python CreatePreferenceDataDockstring.py
# python CreatePreferenceDataAffinities.py

# python BindingAffinity_Distribution.py 

python BindingAffinity_Distribution.py 0.11
# echo "Done with 0.1"

# python BindingAffinity_Distribution.py 0.2
# echo "Done with 0.2"

# python BindingAffinity_Distribution.py 0.3
# echo "Done with 0.3"

# python BindingAffinity_Distribution.py 0.4
# echo "Done with 0.4"

# python BindingAffinity_Distribution.py 0.5
# echo "Done with 0.5"

# python BindingAffinity_Distribution.py 0.6
# echo "Done with 0.6"

# python BindingAffinity_Distribution.py 0.7
# echo "Done with 0.7"

# python BindingAffinity_Distribution.py 0.8
# echo "Done with 0.8"

# python BindingAffinity_Distribution.py 0.9
# echo "Done with 0.9"



