# %% [markdown]
# ## Text Classification with LM-BFF.
# In this tutorial, we do sentiment analysis with automatic template and verbalizer generation. We use SST-2 as an example.

# %% [markdown]
# ### 1. load dataset

# %%
import json 
import argparse
import torch 
parser = argparse.ArgumentParser("")
from openprompt.data_utils.utils import InputExample

from openprompt.data_utils.text_classification_dataset import SST2Processor
from utils import load_datasets, index_select, load_pt_utils
from openprompt.plms import load_plm
from openprompt.pipeline_base import PromptDataLoader, PromptForClassification
from openprompt.prompts import ManualTemplate
from openprompt.trainer import ClassificationRunner
import copy
import torch
from transformers import  AdamW, get_linear_schedule_with_warmup
import copy
import random 
import numpy as np 
from tqdm import tqdm 
from sklearn.metrics import f1_score
def set_seed(args):
    random.seed(args.train_seed)
    np.random.seed(args.train_seed)
    torch.manual_seed(args.train_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.train_seed)
        torch.cuda.manual_seed(args.train_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

parser.add_argument("--lr", type=float, default=1e-4)

parser.add_argument("--train_seed", default=0, type=int, help="which seed to use")
parser.add_argument("--task", default="agnews", type=str, help="The name of the task to train")
parser.add_argument("--data_dir", default="../datasets", type=str,
                    help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
parser.add_argument("--model_dir", default="./model", type=str, help="Path to model")
parser.add_argument("--eval_dir", default="./eval", type=str, help="Evaluation script, result directory")
parser.add_argument("--train_file", default="train.tsv", type=str, help="Train file")
parser.add_argument("--dev_file", default="dev.tsv", type=str, help="dev file")
parser.add_argument("--test_file", default="test.tsv", type=str, help="Test file")
parser.add_argument("--unlabel_file", default="unlabeled.tsv", type=str, help="Test file")
parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the test set.")

parser.add_argument("--dev_labels", default=100, type=int, help="number of labels for dev set")
parser.add_argument("--cache_dir", default="", type=str, help="Where do you want to store the pre-trained models downloaded from s3",)
parser.add_argument("--output_dir", default=None, type=str, required=True, help="The output directory where the model predictions and checkpoints will be written.",)

parser.add_argument('--logging_steps', type=int, default=20, help="Log every X updates steps.")
parser.add_argument('--self_train_logging_steps', type=int, default=20, help="Log every X updates steps.")
parser.add_argument('--save_steps', type=int, default=200, help="Save checkpoint every X updates steps.")
parser.add_argument("--model_type", default="bert-base-uncased", type=str)
parser.add_argument("--add_sep_token", action="store_true", help="Add [SEP] token at the end of the sentence")
parser.add_argument("--max_seq_len", default=128, type=int, help="The maximum total input sequence length after tokenization.")

parser.add_argument("--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform.")
parser.add_argument("--max_steps", default=100, type=int, help="Training steps for initialization.")
parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
parser.add_argument("--dropout_rate", default=0.1, type=float, help="Dropout for fully-connected layers")
parser.add_argument("--batch_size", default=32, type=int, help="Batch size for training and evaluation.")
parser.add_argument("--eval_batch_size", default=256, type=int, help="Batch size for training and evaluation.")

parser.add_argument("--al_method", default='random', type=str, help="The initial learning rate for Adam.")
parser.add_argument("--sample_labels", default=32, type=int, help="The initial learning rate for Adam.")
args = parser.parse_args()
args.model_name_or_path = args.model_type

set_seed(args)

dataset = {}
unlabeled =  load_datasets(args, mode = 'train', template_id = 0)
dev =  load_datasets(args, mode = 'dev', template_id = 0)
test = load_datasets(args, mode = 'test', template_id = 0)
with open(f"../datasets/{args.task}-0-0/train_idx_roberta-base_{args.al_method}_{args.sample_labels}.json", 'r') as f:
    train_id = json.load(f)
    print("Number of labeled data: ", len(train_id))


train = index_select(train_id, unlabeled)
print(len(train), len(dev), len(test))
template, verbalizer = load_pt_utils(dataset = args.task)
print(f"n_classes: {len(verbalizer)}, template: {template}, verbalizer: {verbalizer}")
dataset['train'] = train
dataset['validation'] = dev
dataset['test'] = test

# %% [markdown]
# ### 2. build initial verbalizer and template
# - note that if you wish to do automatic label word generation, the verbalizer is not the final verbalizer, and is only used for template generation.
# - note that if you wish to do automatic template generation, the template text may desirably include `{"meta":"labelword"}` so that label word can be used and remember to use `LMBFFTemplateGenerationTemplate` class so that "labelword" can be handled properly. Else you can just use `ManualTemplate`
# - below is a template that expects plain text generation at each "mask" token position

# %%
print('load model...')
from openprompt.plms import load_plm
# load mlm model for main tasks
plm, tokenizer, model_config, WrapperClass = load_plm("roberta", "roberta-base")

# load generation model for template generation
# template_generate_model, template_generate_tokenizer, template_generate_model_config, template_tokenizer_wrapper = load_plm('t5', 't5-large')

from openprompt.prompts import ManualVerbalizer, ManualTemplate

verbalizer = ManualVerbalizer(tokenizer=tokenizer, num_classes=len(verbalizer), label_words=verbalizer) # [['terrible'],['great']])

# from openprompt.prompts.prompt_generator import LMBFFTemplateGenerationTemplate
# template = LMBFFTemplateGenerationTemplate(tokenizer=template_generate_tokenizer, verbalizer=verbalizer, text='{"placeholder":"text_a"} {"mask"} {"meta":"labelword"} {"mask"}.')
template = ManualTemplate(tokenizer=tokenizer, text=template) # '{"placeholder":"text_a"} It is {"mask"}.')

# view wrapped example
wrapped_example = template.wrap_one_example(dataset['train'][0])
print(wrapped_example)

# %%
# parameter setting
cuda = True
auto_t = False # whether to perform automatic template generation
auto_v = False # whether to perform automatic label word generation


# %%
# train util function
def fit(args, model, train_dataloader, val_dataloader, loss_func, optimizer):
    best_score = 0.0
    for epoch in range(int(args.num_train_epochs)):
        # train_epoch(model, train_dataloader, loss_func, optimizer)
        for step, inputs in enumerate(train_dataloader):
            model.train()
            training_len = len(train_dataloader)
            if cuda:
                inputs = inputs.cuda()
            logits = model(inputs)
            labels = inputs['label']
            loss = loss_func(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            if step in [0, training_len//4, training_len//2, 3*training_len//4]:
                dev_score, dev_f1 = evaluate(model, val_dataloader)
                print(f"epoch = {epoch}, step = {step}, dev acc = {dev_score}, dev f1 = {dev_f1}")
                if dev_score > best_score:
                    best_score = dev_score
                    best_model = copy.deepcopy(model)
    return best_score, best_model


def train_epoch(model, train_dataloader, loss_func, optimizer):
    model.train()
    for step, inputs in tqdm(enumerate(train_dataloader)):
        training_len = len(train_dataloader)
        if cuda:
            inputs = inputs.cuda()
        logits = model(inputs)
        labels = inputs['label']
        loss = loss_func(logits, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


def evaluate(model, val_dataloader):
    model.eval()
    allpreds = []
    alllabels = []
    with torch.no_grad():
        for step, inputs in tqdm(enumerate(val_dataloader)):
            if cuda:
                inputs = inputs.cuda()
            logits = model(inputs)
            labels = inputs['label']
            alllabels.extend(labels.cpu().tolist())
            allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
    acc = sum([int(i==j) for i,j in zip(allpreds, alllabels)])/len(allpreds)
    f1 = f1_score(alllabels, allpreds, average='macro')
    return acc, f1


# %% [markdown]
# ### 3. automatic template and verbalizer generation

# %%
from tqdm import tqdm
# template generation
# %% [markdown]
# ### 4. main training loop

# %%
# main training loop
train_dataloader = PromptDataLoader(dataset['train'], template, tokenizer=tokenizer, tokenizer_wrapper_class=WrapperClass, max_seq_length = args.max_seq_len, batch_size = args.batch_size)
valid_dataloader = PromptDataLoader(dataset['validation'], template, tokenizer=tokenizer, tokenizer_wrapper_class=WrapperClass, max_seq_length= args.max_seq_len, batch_size = args.eval_batch_size)
test_dataloader = PromptDataLoader(dataset['test'], template, tokenizer=tokenizer, tokenizer_wrapper_class=WrapperClass, max_seq_length= args.max_seq_len, batch_size = args.eval_batch_size)


model = PromptForClassification(copy.deepcopy(plm), template, verbalizer)
loss_func = torch.nn.CrossEntropyLoss()
no_decay = ['bias', 'LayerNorm.weight']
# it's always good practice to set no decay to biase and LayerNorm parameters
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)

if cuda:
    model = model.cuda()

score, best_model = fit(args, model, train_dataloader, valid_dataloader, loss_func, optimizer)
test_score, test_f1 = evaluate(best_model, test_dataloader)
print("==================")

print(f"Dataset: {args.task}, Labels: {len(train)}, Seed: {args.train_seed}, Acc: {test_score}, F1: {test_f1}")

result_dict = {'seed': args.train_seed, 'labels': len(train)}
result_dict['acc'] = test_score
result_dict['f1'] = test_f1

result_dict['lr'] = args.lr
result_dict['bsz'] = args.batch_size
line = json.dumps(result_dict)
with open(f'{args.output_dir}_pt_{args.al_method}.json', 'a+') as f:
    f.write(line + '\n')
