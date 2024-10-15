# Phosphorylation
2 CNN models that take SCDM and SCDM+SHDM maps respectivly as inputs and predict Phosphorylation (binary 1 for yes 0 for no)
SCDM and SHDM maps are created from the Phos dataset(located in the kings group shared folder) 

Train models by running the submission script on the DU SLURM cluster using the command

sbatch submitter.sh

this script runs 10 gpu jobs in parallel to do a 10 fold cross validation, and saves all the respective models and their testing scores (AUPRC MCC and accuracy)

the models are saved in their respective folders.

