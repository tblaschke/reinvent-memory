# Implementation of "Memory-assisted reinforcement learning for diverse molecular de novo design"

This repository holds the code used to train, sample, transfer learn and reinforcement learn models described in [Memory-assisted reinforcement learning for diverse molecular de novo design](https://chemrxiv.org/).
This code is based on our [updated REINVENT code](https://github.com/tblaschke/reinvent) used in "Molecular De Novo Design through Deep Reinforcement
Learning".

The scripts and folders are the following:

1) Python files in the main folder are all scripts. Run them with `-h` for usage information.
2) `./priors` folder: Contains the trained priors used in the manuscript.
3) `./scaffold` folder: Contains the code of the memory units.
4) `./results` folder: Contains all the used datasets and generated molecules form the manuscript

## Requirements

The repository includes a Conda `environment.yml` file with the required libraries to run all the scripts. It should work fine with Linux setups and a mid-high range GPU.

### Install

A [Conda](https://conda.io/miniconda.html) `environment.yml` is supplied with all the required libraries.
~~~~bash
git clone https://github.com/tblaschke/reinvent-memory
cd reinvent-memory
conda env create -f environment.yml
conda activate reinvent
~~~~
From here the general usage applies.

## General Usage

1) Create Model (`create_model.py`): Creates a blank model file.
2) Train Model (`train_model.py`): Trains the model with the specified parameters.
3) Transfer Learn Model (`transfer_learn_model.py`): Performs transfer learning of an already trained model with the specified parameters.
4) Sample Model (`sample_from_model.py`): Samples an already trained model for a given number of SMILES.
5) Reinforcement Learn Model (`reinforce_model.py`): Performs reinforcement learning of an already trained model with the specified parameters.

## Memory Unit

This code contains the memory unit aka scaffold filter. It's a hash table which dynamically changes the reward function during the reinforcement learning such the model gets penalized when it generates too similar compounds. 

The basic principle goes as follows. Highly scored compounds get added to the memory. Inside the memory the compounds are assigned into different buckets. Each bucket is represented by an index compound. If a bucket is "full" we alter the value of the scoring function to be 0. To find out into which bucket a compound belongs, it's compared to all index compounds. 

There are multiple modes to assign compounds into buckets.

1) CompoundSimilarity: Calculates the ECFP Tanimoto similarity to all index compounds.
2) MurckoScaffold: Checks for exact matches of Bemis-Murcko Scaffold 
3) TopologicalScaffold:  Checks for exact matches of Topological Scaffold / Carbon Skeleton (replace all heavy atoms in a BM scaffold with carbons, and connect them only via single bonds)
4) ScaffoldSimilarity: Checks for Tanimoto similarity between topological scaffolds. Uses the Atom Pair fingerprint.
5) NoFilter: Only saves the generated molecules. Does nothing on it's own. This mode is equivalent to a normal RL.

## Reproducibility

The results used for the publication can be found in the `results` folder. It contains all saved memories for all the different runs and the result of our MMP analysis.
If you want to reproduce all the numbers for yourself please checkout the other code repositories [for the data extraction and SVM training](https://github.com/tblaschke/reinvent-classifiers) and [the MMP analysis](https://github.com/tblaschke/reinvent-mmp-analysis). To make it as easy as possible to run all of the code copy all three code repositories into your ~/projects folder (if it does not exists, create it)
~~~~bash
mkdir -f ~/projects
git clone https://github.com/tblaschke/reinvent-memory ~/projects/reinvent-memory
git clone https://github.com/tblaschke/reinvent-classifiers ~/projects/reinvent-classifiers
git clone https://github.com/tblaschke/reinvent-mmp-analysis ~/projects/reinvent-mmp-analysis
~~~~

### Data extraction and SVM training

This command will download ExCAPE, filter the activity data, create the datasets and train the SVM models.
~~~~bash
cd ~/projects/reinvent-classifiers
conda activate reinvent
bash run_extraction_and_svm_training.sh
~~~~

### Run all RL experiments

After the datasets are constructed and SVMs are trained, this command will run all RL experiments. (Use a GPU for this)
~~~~bash
cd ~/projects/reinvent-classifiers
conda activate reinvent
bash run_all_reinvent_experiments.sh
~~~~

### MMP analysis

After the RL experiments. This will collect all the data and perform the MMP analysis.
~~~~bash
cd ~/projects/reinvent-mmp-analysis
conda activate reinvent
bash run_mmpa.sh
~~~~

## Support

If you have any questions please feel free to open an issue on GitHub or write an mail to thomas.blaschke@uni-bonn.de