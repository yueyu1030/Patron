# Patron
This repo contains the code for our paper [Cold-Start Data Selection for Few-shot Language Model Fine-tuning: A Prompt-Based Uncertainty Propagation Approach]() (arXiv preprint).

## Model Framework
![Figure](Figure/PATRON.png)



## Dependencies
```
python 3.8
transformers==4.2.0
pytorch==1.8.0
tqdm
scikit-learn
faiss-cpu==1.6.4
sentencepiece==0.1.96
tqdm>=4.62.2
tensorboardX
nltk
pip install openprompt
```

## Datasets
We use the following four datasets for the main experiments.
|   Dataset   | Task  | Number of Classes | Number of Unlabeled Data/Test Data|
|---------------- | -------------- |-------------- | -------------- |
| [IMDB](https://huggingface.co/datasets/imdb)       |     Sentiment           |     2   |  25k/25k  |
| [Yelp-full](https://github.com/yumeng5/WeSHClass)       |     Sentiment           |     5   |  39k/10k  |
| [AG News](https://huggingface.co/datasets/ag_news) |    News Topic       |      4      |  119k/7.6k   |
| [Yahoo! Answers](https://huggingface.co/datasets/yahoo_answers_topics)  |  QA Topic  |     5        |     180k/30.1k    |
| [DBPedia](https://huggingface.co/datasets/dbpedia_14)     |     Ontology Topic      |      14      |     280k/70k      |
| [TREC](https://huggingface.co/datasets/trec)     |     Question Topic      |      6      |     5k/0.5k      |

The processed data can be found at [this link](https://drive.google.com/drive/folders/1qSGGxVlxmy1-T1RLDlwGlGHKrw2kEKKm?usp=sharing). The folder to put these datasets will be discribed in the following parts.

## Data Selection Pipeline
### a) Generating Unsupervised Text Embeddings via SimCSE
Run the following commands
```
python gen_embedding_simcse.py --dataset [the dataset you use] --gpuid [the id of gpu you use] --batchsize [the number of data processed in one time]
```

### b) Estimating Uncertainty via Prompts
We provide the *pseudo prediction* obtained via prompts in the above link for datasets. Please refer to the original papers for details. 

### c) Data Selection with Uncertainty Propagation and PTR
Run the following commands (example on AG News dataset)
```
python patron_sample.py --dataset agnews --k 50 --rho 0.01 --gamma 0.5 --beta 0.5
```
Some important hyperparameters:
- `rho`: the parameter used for uncertainty propagation in Eq. 6 of the paper 
- `beta`: the regularization of distance in Eq. 8 of the paper 
- `gamma`: the weight of the  regularization term in Eq. 10 of the paper

## Experiments
### Running Fine-tuning Experiments
See `finetune` folder for detailed instructions.


### Running Prompt-based Learning Experiments
See `prompt_learning` folder for detailed instructions.


## Running on a New Dataset

See [this link](https://github.com/thunlp/OpenPrompt/blob/ca27491101df0108a8dd753e5b1e79bf591f65d3/docs/source/notes/examples.rst#introduction-with-an-example) as the pipeline for generating the prompt-based predictions. 

## Citation
Please kindly cite the following paper if you find this repo helpful for your research. Thanks in advance!

## Acknowledgements 
The author would like to thank the authors from the repo [SimCSE](https://github.com/princeton-nlp/SimCSE) and [OpenPrompt](https://github.com/thunlp/OpenPrompt) for the well-organized code.