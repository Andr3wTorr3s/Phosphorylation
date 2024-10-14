# Phosphorylation
2 models presented here that train and validate on SCDM and SHDM maps based on the Phos data set. (located in the kings group shared folder) 

Train models by running the submission script on the DU SLURM cluster using the command

sbatch submitter.sh

this script runs 10 gpu jobs in parallel to do a 10 fold cross validation, and saves all the respective models and their testing scores (AUPRC and accuracy)

the models are saved in their respective folders.

