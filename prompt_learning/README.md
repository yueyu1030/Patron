# Prompt-based Learning Experiments
## Requirements
```
python 3.8
transformers==4.2.0
pytorch==1.6.0
tqdm
scikit-learn
faiss-cpu==1.6.4
openprompt
```

## Dataset Preprocessing
The input for prompt-based learning is the same as the vanilla fine-tuning. 
Suppose we have the selected id from the training data from `patron_sample.py`. An example of the file name should be `train_idx_roberta-base_round2_rho0.01_gamma0.5_beta0.2_mu0.5_sample32.json`. 

Then, we need to sample the unlabeled data into the *training* set and _validation_ set. 
The number in the above file is the *index* of the selected data, which will be used as the training set. 
The validation set is randomly selected from the unlabeled data.

The corresponding train/dev dataset are `train_[budget].json` and `valid.json`.


## Training Commands
Run the following commands `commands/run.sh` for fine-tuning the PLM with the selected data.

_Note_: the relevant configurations for prompts (e.g. verbalizers, templates) are in `utils.py`. 

## Hyperparameters
Note: the three numbers in each grid indicate the parameter for 32/64/128 labels.

|  | IMDB | Yelp-full | AG News | Yahoo! | DBPedia | TREC |
| ------ | ------ | ------ | ------ | ------ | ------  |------  |
| BSZ | 4/8/16 | 4/8/16  | 4/8/16  | 4/8/16  | 4/8/16  | 4/8/16  |
| LR | 2e-5/1e-5/1e-5 | 1e-5 | 2e-5 | 1e-5 | 1e-5 | 5e-5/5e-5/1e-5 
| EPOCH | 18 | 18 | 15 | 15 | 18 | 15 
| Max Tokens | 256 | 256 | 128 | 128 | 128 | 64