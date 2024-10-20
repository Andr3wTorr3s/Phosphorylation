## What is Phosphorylation? 

Simply put, it is the attachment of a phosphate group to a molecule or ion. A common bioinformatics task is to try and predict the phosphorylation site given a sequence. Most models predict this phosphorylation with a model that looks at trends in sequence alone or with sequence trends alongside shape information or disorder info.  

## What is the model presented here?
Instead of predicting phosphoylation from sequence alone or ssequence+shape/disorder, here I present a model that predicts phosphorylation with more "physics" information in the form of SCDM or SHDM or both. What is SCDM? (S)equence (C)harge (D)ecoration (M)atricies are matrix representations of protein sequences that detail coulomb or "charge" interactions and their respective distances in sequence. Similarly, (S)equence (H)hydrophobicity (Decoration) (M)atricies (SHDM) detail hydrophobic interactions and their respective distances. More detail on both of these can be found here: https://pmc.ncbi.nlm.nih.gov/articles/PMC9190209/ 

The idea with this model is the following:  Protein shape has an effect on phosphorylation. since hydrophobicity, hydrophobicity placement, charge, and charge placement have an effect on phosphorylation then hydrophobicity decoration and charge decoration must have an effect on phosphorylation.  

Here I present a Convolutional Neural Network, that trains and vaildates on SCDM, SHDM or both and predicts phosphorylation. 

##Data set

The model was trained and validated on the PhosphoELM dataset https://academic.oup.com/nar/article/36/suppl_1/D240/2506170
The datasets covers 12025 phospho-serine, 2362 phospho-threonine and 2083 phospho-tyrosine sites and the CNN model was trained and tested seperatly on the PELM set and the PPA set for each of the amino acid sites seperatly in the same manner used to train the PARROT model (https://elifesciences.org/articles/70576#content).



## How to use
2 CNN models that take SCDM and SCDM+SHDM maps respectivly as inputs and predict Phosphorylation (binary 1 for yes 0 for no)
SCDM and SHDM maps are created from the Phos dataset(located in the kings group shared folder) 

Train models by running the submission script on the DU SLURM cluster using the command

sbatch submitter.sh

this script runs 10 gpu jobs in parallel to do a 10 fold cross validation, and saves all the respective models and their testing scores (AUPRC MCC and accuracy)

the models are saved in their respective folders.

