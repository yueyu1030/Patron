import os
import logging
from tqdm import tqdm, trange
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, ConcatDataset, TensorDataset
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm, trange
from transformers import BertConfig, AdamW, get_linear_schedule_with_warmup, AutoConfig, AutoModelForSequenceClassification
import copy
import math
import os
import random 
from sklearn.metrics import f1_score
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter
from collections import Counter

logger = logging.getLogger(__name__)


def compute_metrics(preds, labels):
    assert len(preds) == len(labels)
    return acc_and_f1(preds, labels)


def acc_and_f1(preds, labels, average='macro'):
    acc = (preds == labels).mean()
    macro_recall = f1_score(y_true=labels, y_pred = preds, average = 'macro')
    micro_recall = f1_score(y_true=labels, y_pred = preds, average = 'micro')
    #print(acc, macro_recall, micro_recall)

    return {
        "acc": acc,
        "f1": macro_recall,
        "f1_micro": micro_recall
    }

class Trainer(object):
    def __init__(self, args, train_dataset = None, dev_dataset = None, test_dataset = None, unlabeled = None, contra_datasets= [], \
                num_labels = 10, data_size = 100, n_gpu = 1):
        self.args = args
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset
        self.unlabeled = unlabeled
        self.contra_datasets = contra_datasets
        self.data_size = data_size

        self.num_labels = num_labels
        self.config_class = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=self.num_labels)
        self.model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, num_labels=self.num_labels)
        self.n_gpu = 1
        
    def init_model(self):
        # GPU or CPU
        self.device = "cuda" if torch.cuda.is_available() and self.n_gpu > 0 else "cpu"
        if self.n_gpu > 1:
            self.model = nn.DataParallel(self.model)
        self.model = self.model.to(self.device)
    
    def load_model(self, path = None):
        if path is None:
            logger.info("No ckpt path, load from original ckpt!")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.args.model_name_or_path,
                config=self.config_class,
                cache_dir=self.args.cache_dir if self.args.cache_dir else None,
            )
        else:
            logger.info(f"Loading from {path}!")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                path,
                config=self.config_class,
                cache_dir=self.args.cache_dir if self.args.cache_dir else None,
            )


    def save_model(self, stage = 0):
        output_dir = os.path.join(
            self.args.output_dir, f"{self.args.al_method}", f"seed{self.args.train_seed}", "checkpoint-{}".format(stage))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model_to_save = (
            self.model.module if hasattr(self.model, "module") else self.model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(output_dir)

        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
        logger.info("Saving model checkpoint to %s", output_dir)

   
    def train(self, n_sample = 20):
        train_sampler = RandomSampler(self.train_dataset)
        train_dataloader = DataLoader(self.train_dataset, sampler=train_sampler, batch_size=self.args.batch_size)
        
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        training_steps = max(self.args.max_steps, int(self.args.num_train_epochs) * len(train_dataloader))

        # Train!
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(self.train_dataset))
        logger.info("  Num Epochs = %d", self.args.num_train_epochs)
        logger.info("  Total train batch size = %d", self.args.batch_size)
        logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", training_steps)
        global_step = 0
        tr_loss = 0.0

        self.model.zero_grad()

        train_iterator = trange(int(self.args.num_train_epochs), desc="Epoch")
        best_dev = -np.float('inf')
        for _ in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            local_step = 0
            training_len = len(epoch_iterator)
            for step, batch in enumerate(epoch_iterator):
                self.model.train()
                batch = tuple(t.to(self.device) for t in batch)  # GPU or CPU
                inputs = {
                            'input_ids': batch[0],
                            'attention_mask': batch[1],
                            'token_type_ids': batch[2],
                            'labels': batch[3],
                          }

                outputs = self.model(**inputs)
                loss = outputs[0]
                logits = outputs[1]
                # loss = criterion(input = F.log_softmax(logits), target = batch[3].to(self.device))
                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps           
                if torch.cuda.device_count() > 1:
                    loss = loss.mean()
                loss.backward()
                tr_loss += loss.item()
                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    local_step += 1
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                    optimizer.step()
                    self.model.zero_grad()
                    global_step += 1
                    epoch_iterator.set_description("iteration:%d, Loss:%.3f, best dev:%.3f" % (_, tr_loss/global_step, 100*best_dev))
                    if local_step in [training_len//2]: 
                        loss_dev, acc_dev, f1_dev = self.evaluate('dev', global_step)
                        if acc_dev > best_dev:
                            logger.info("Best model updated!")
                            self.best_model = copy.deepcopy(self.model.state_dict())
                            best_dev = acc_dev
                
                if 0 < training_steps < global_step:
                    epoch_iterator.close()
                    break
            loss_dev, acc_dev, f1_dev = self.evaluate('dev', global_step)
            if acc_dev > best_dev:
                logger.info("Best model updated!")
                self.best_model = copy.deepcopy(self.model.state_dict())
                best_dev = acc_dev
            print(f'Dev: Loss: {loss_dev}, Acc: {acc_dev}, F1: {f1_dev}', f'Test: Loss: {loss_test}, Acc: {acc_test}')
        result_dict = {'seed': self.args.train_seed, 'labels': self.args.sample_labels}
        self.model.load_state_dict(self.best_model)
        loss_test, acc_test, f1_test = self.evaluate('test', global_step)
        result_dict['acc'] = acc_test
        result_dict['f1'] = f1_test
        result_dict['lr'] = self.args.learning_rate
        result_dict['bsz'] = self.args.batch_size
        if len(self.contra_datasets) > 0:
            for i, dataset in enumerate(self.contra_datasets):
                loss_, acc_i, f1_i = self.evaluate(mode = 'contra', dataset = dataset, global_step=global_step) 
                result_dict[f'acc_contra_{i}'] = acc_i
                print(f'Test Contra {i}: Loss: {loss_}, Acc: {acc_i}')
        print(f'Test: Loss: {loss_test}, Acc: {acc_test}, F1: {f1_test}')
        import json 
        line = json.dumps(result_dict)
        with open(f'{self.args.output_dir}_{self.args.model_type}_{self.args.al_method}.json', 'a+') as f:
            f.write(line + '\n')
        self.save_model(stage = n_sample)
        return global_step, tr_loss / global_step
   

    def evaluate(self, mode, dataset = None, global_step=-1):
        # We use test dataset because semeval doesn't have dev dataset
        if mode == 'test':
            dataset = self.test_dataset
        elif mode == 'dev':
            dataset = self.dev_dataset
        elif mode == 'contra':
            dataset = dataset
        else:
            raise Exception("Only dev and test dataset available")

        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=self.args.eval_batch_size)

        # Eval!
        logger.info("***** Running evaluation on %s dataset *****", mode)
        logger.info("  Num examples = %d", len(dataset))
        # logger.info("  Batch size = %d", self.args.batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        self.model.eval()
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                inputs = {
                            'input_ids': batch[0],
                            'attention_mask': batch[1],
                            'token_type_ids': batch[2],
                            'labels': batch[3],
                          }
                outputs = self.model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1

            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(
                    out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        results = {
            "loss": eval_loss
        }
        preds = np.argmax(preds, axis=1)
        
        result = compute_metrics(preds, out_label_ids)
        result.update(result)
        logger.info("***** Eval results *****")

    
        return results["loss"], result["acc"], result["f1"]
