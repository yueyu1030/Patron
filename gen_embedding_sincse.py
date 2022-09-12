import torch
from scipy.spatial.distance import cosine
from transformers import AutoModel, AutoTokenizer
import argparse
import json
from tqdm import trange
import numpy as np
import pickle
# Tokenize input texts
def get_arguments():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--dataset",
        default='agnews',
        type=str,
        required=True,
        help="The input data dir. Should contain the cached passage and query files",
    )
    parser.add_argument(
        "--type",
        default='unlabeled',
        type=str,
        help="The input data dir. Should contain the cached passage and query files",
    )

    parser.add_argument(
        "--batch_size",
        default=64,
        type=int,
        help="The input data dir. Should contain the cached passage and query files",
    )

    parser.add_argument(
        "--gpuid",
        default=0,
        type=int,
        help="The input data dir. Should contain the cached passage and query files",
    )
    args = parser.parse_args()
    return args

args = get_arguments()
text = []
label = []
model_name = "princeton-nlp/unsup-simcse-roberta-base"

text_a = []
text_b = []
with open(f"{args.dataset}/{args.type}.json", 'r') as f:
    for lines in f:
        lines = json.loads(lines)
        text.append(lines["text"])
        label.append(lines["_id"])
    # print number of unlabeled data/classes
    print(len(text), len(label))

# Import our models. The package will take care of downloading the models automatically
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

model = model.to(f"cuda:{args.gpuid}")

embedding = []


num_iter = len(text)//args.batch_size if len(text) % args.batch_size == 0 else (len(text)//args.batch_size + 1)
for i in trange(len(text)//args.batch_size + 1):
    inputs = tokenizer(text[i*args.batch_size:(i+1)*args.batch_size], padding=True, truncation=True, return_tensors="pt").to(f"cuda:{args.gpuid}")
    # Get the embeddings
    with torch.no_grad():
        embeddings = model(**inputs, output_hidden_states=True, return_dict=True).pooler_output
        embedding.append(embeddings.cpu().numpy())

embedding = np.concatenate(embedding, axis = 0)
print(embedding.shape)

with open(f"{args.dataset}/embedding_{args.model}_simcse_{args.type}.pkl", 'wb') as handle:
    pickle.dump(embedding, handle, protocol=4)


    



