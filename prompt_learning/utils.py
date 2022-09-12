# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This file contains the pattern-verbalizer pairs (PVPs) for all tasks.
"""
import random
import string
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Tuple, List, Union, Dict

import torch
import json
import re
import json
import logging
import copy
import csv,os

from torch.utils.data import TensorDataset

from transformers import PreTrainedTokenizer, GPT2Tokenizer
from transformers import AutoTokenizer, RobertaTokenizer, RobertaConfig, RobertaForSequenceClassification, BertTokenizer, BertForSequenceClassification, BertConfig
from transformers import AlbertForSequenceClassification, AlbertTokenizer, AlbertConfig, DistilBertTokenizer
# from pet.task_helpers import MultiMaskTaskHelper
# from pet.tasks import TASK_HELPERS
# from pet.utils import InputExample, get_verbalization_ids
from openprompt.data_utils.utils import InputExample

# import log
# from pet import wrapper as wrp
import logging

logger = logging.getLogger(__name__)

FilledPattern = Tuple[List[Union[str, Tuple[str, bool]]], List[Union[str, Tuple[str, bool]]]]

logger = logging.getLogger(__name__)

def init_logger():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

def load_tokenizer(args):
    # tokenizer_dict = {'bert': BertTokenizer, 'roberta': RobertaTokenizer, 'albert': AlbertTokenizer, 'distillbert': DistilBertTokenizer,}
    # if 'roberta' in args.model_name_or_path:
        # tokenizer = tokenizer_dict['roberta'].from_pretrained(args.model_name_or_path)
    # else:
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    # tokenizer.add_special_tokens({"additional_special_tokens": ADDITIONAL_SPECIAL_TOKENS})
    return tokenizer

# class InputExample(object):
#     """
#     A single training/test example for simple sequence classification.
#     Args:
#         guid: Unique id for the example.
#         text_a: string. The untokenized text of the first sequence. For single
#         sequence tasks, only this sequence must be specified.
#         label: (Optional) string. The label of the example. This should be
#         specified for train and dev examples, but not for test examples.
#     """

#     def __init__(self, guid, text_a, label, task, text_b = None):
#         self.guid = guid
#         self.text_a = text_a
#         self.text_b = text_b
#         self.task = task
#         self.label = label

#     def __repr__(self):
#         return str(self.to_json_string())

#     def to_dict(self):
#         """Serializes this instance to a Python dictionary."""
#         output = copy.deepcopy(self.__dict__)
#         return output

#     def to_json_string(self):
#         """Serializes this instance to a JSON string."""
#         return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


def get_verbalization_ids(word, tokenizer, force_single_token: bool) -> Union[int, List[int]]:
    """
    Get the token ids corresponding to a verbalization

    :param word: the verbalization
    :param tokenizer: the tokenizer to use
    :param force_single_token: whether it should be enforced that the verbalization corresponds to a single token.
           If set to true, this method returns a single int instead of a list and throws an error if the word
           corresponds to multiple tokens.
    :return: either the list of token ids or the single token id corresponding to this word
    """
    # kwargs = {'add_prefix_space': True} if isinstance(tokenizer, RobertaTokenizer) else {}
    # print(tokenizer,tokenizer.name_or_path)
    # if 'roberta' in tokenizer.name_or_path:
    #     ids = tokenizer.encode(word, add_special_tokens=False, **kwargs)
    # else:
    ids = tokenizer.encode(word, add_special_tokens=False)
    print(word, ids)
    if not force_single_token:
        return ids
    # if 'roberta' in tokenizer.name_or_path:
    #     assert len(ids) == 1, \
    #     f'Verbalization "{word}" does not correspond to a single token, got {tokenizer.convert_ids_to_tokens(ids)}'
    #     verbalization_id = ids[0]
    # else:
    assert len(ids) == 1, \
        f'Verbalization "{word}" does not correspond to a single token, got {tokenizer.convert_ids_to_tokens(ids)}'
    verbalization_id = ids[0]
    assert verbalization_id not in tokenizer.all_special_ids, \
        f'Verbalization {word} is mapped to a special token {tokenizer.convert_ids_to_tokens(verbalization_id)}'
    return verbalization_id

class PVP(ABC):
    """
    This class contains functions to apply patterns and verbalizers as required by PET. Each task requires its own
    custom implementation of a PVP.
    """

    def __init__(self, tokenizer, pattern_id: int = 0, verbalizer_file: str = None, seed: int = 42):
        """
        Create a new PVP.

        :param wrapper: the wrapper for the underlying language model
        :param pattern_id: the pattern id to use
        :param verbalizer_file: an optional file that contains the verbalizer to be used
        :param seed: a seed to be used for generating random numbers if necessary
        """
        # self.wrapper = wrapper
        self.pattern_id = pattern_id
        self.rng = random.Random(seed)

        if verbalizer_file:
            self.verbalize = PVP._load_verbalizer_from_file(verbalizer_file, self.pattern_id)
        self.tokenizer = tokenizer
        # use_multimask = (self.wrapper.config.task_name in TASK_HELPERS) and (
        #     issubclass(TASK_HELPERS[self.wrapper.config.task_name], MultiMaskTaskHelper)
        # )
        # if not use_multimask and self.wrapper.config.wrapper_type in [wrp.MLM_WRAPPER, wrp.PLM_WRAPPER]:
        #     self.mlm_logits_to_cls_logits_tensor = self._build_mlm_logits_to_cls_logits_tensor()

    # def _build_mlm_logits_to_cls_logits_tensor(self):
    #     label_list = self.wrapper.config.label_list
    #     m2c_tensor = torch.ones([len(label_list), self.max_num_verbalizers], dtype=torch.long) * -1

    #     for label_idx, label in enumerate(label_list):
    #         verbalizers = self.verbalize(label)
    #         for verbalizer_idx, verbalizer in enumerate(verbalizers):
    #             verbalizer_id = get_verbalization_ids(verbalizer, self.wrapper.tokenizer, force_single_token=True)
    #             assert verbalizer_id != self.wrapper.tokenizer.unk_token_id, "verbalization was tokenized as <UNK>"
    #             m2c_tensor[label_idx, verbalizer_idx] = verbalizer_id
    #     return m2c_tensor

    @property
    def mask(self) -> str:
        """Return the underlying LM's mask token"""
        return self.tokenizer.mask_token

    @property
    def mask_id(self) -> int:
        """Return the underlying LM's mask id"""
        return self.tokenizer.mask_token_id

    @property
    def max_num_verbalizers(self) -> int:
        """Return the maximum number of verbalizers across all labels"""
        return max(len(self.verbalize(label)) for label in self.wrapper.config.label_list)

    @staticmethod
    def shortenable(s):
        """Return an instance of this string that is marked as shortenable"""
        return s, True

    @staticmethod
    def remove_final_punc(s: Union[str, Tuple[str, bool]]):
        """Remove the final punctuation mark"""
        if isinstance(s, tuple):
            return PVP.remove_final_punc(s[0]), s[1]
        return s.rstrip(string.punctuation)

    @staticmethod
    def lowercase_first(s: Union[str, Tuple[str, bool]]):
        """Lowercase the first character"""
        if isinstance(s, tuple):
            return PVP.lowercase_first(s[0]), s[1]
        return s[0].lower() + s[1:]

    def encode(self, example: InputExample, \
            priming: bool = False, \
            labeled: bool = False, \
            max_seq_len: int = 512) \
            -> Tuple[List[int], List[int]]:
        """
        Encode an input example using this pattern-verbalizer pair.

        :param example: the input example to encode
        :param priming: whether to use this example for priming
        :param labeled: if ``priming=True``, whether the label should be appended to this example
        :return: A tuple, consisting of a list of input ids and a list of token type ids
        """

        if not priming:
            assert not labeled, "'labeled' can only be set to true if 'priming' is also set to true"

        tokenizer = self.tokenizer  # type: PreTrainedTokenizer
        parts_a, parts_b = self.get_parts(example)

        kwargs = {'add_prefix_space': True} if isinstance(tokenizer, GPT2Tokenizer) else {}

        parts_a = [x if isinstance(x, tuple) else (x, False) for x in parts_a]
        parts_a = [(tokenizer.encode(x, add_special_tokens=False, **kwargs), s) for x, s in parts_a if x]

        if parts_b:
            parts_b = [x if isinstance(x, tuple) else (x, False) for x in parts_b]
            parts_b = [(tokenizer.encode(x, add_special_tokens=False, **kwargs), s) for x, s in parts_b if x]

        self.truncate(parts_a, parts_b, max_length = max_seq_len)

        tokens_a = [token_id for part, _ in parts_a for token_id in part]
        tokens_b = [token_id for part, _ in parts_b for token_id in part] if parts_b else None

        if priming:
            input_ids = tokens_a
            if tokens_b != []:
                input_ids += tokens_b
            if labeled:
                mask_idx = input_ids.index(self.mask_id)
                assert mask_idx >= 0, 'sequence of input_ids must contain a mask token'
                assert len(self.verbalize(example.label)) == 1, 'priming only supports one verbalization per label'
                verbalizer = self.verbalize(example.label)[0]
                verbalizer_id = get_verbalization_ids(verbalizer, self.tokenizer, force_single_token=True)
                # input_ids[mask_idx] = verbalizer_id
            return input_ids, [], mask_idx, verbalizer_id

        input_ids = tokenizer.build_inputs_with_special_tokens(tokens_a, tokens_b)
        token_type_ids = tokenizer.create_token_type_ids_from_sequences(tokens_a, tokens_b)

        return input_ids, token_type_ids

    @staticmethod
    def _seq_length(parts: List[Tuple[str, bool]], only_shortenable: bool = False):
        return sum([len(x) for x, shortenable in parts if not only_shortenable or shortenable]) if parts else 0

    @staticmethod
    def _remove_last(parts: List[Tuple[str, bool]]):
        last_idx = max(idx for idx, (seq, shortenable) in enumerate(parts) if shortenable and seq)
        parts[last_idx] = (parts[last_idx][0][:-1], parts[last_idx][1])

    def truncate(self, parts_a: List[Tuple[str, bool]], parts_b: List[Tuple[str, bool]], max_length: int):
        """Truncate two sequences of text to a predefined total maximum length"""
        total_len = self._seq_length(parts_a) + self._seq_length(parts_b)
        total_len += self.tokenizer.num_special_tokens_to_add(bool(parts_b))
        num_tokens_to_remove = total_len - max_length

        if num_tokens_to_remove <= 0:
            return parts_a, parts_b

        for _ in range(num_tokens_to_remove):
            if self._seq_length(parts_a, only_shortenable=True) > self._seq_length(parts_b, only_shortenable=True):
                self._remove_last(parts_a)
            else:
                self._remove_last(parts_b)

    @abstractmethod
    def get_parts(self, example: InputExample) -> FilledPattern:
        """
        Given an input example, apply a pattern to obtain two text sequences (text_a and text_b) containing exactly one
        mask token (or one consecutive sequence of mask tokens for PET with multiple masks). If a task requires only a
        single sequence of text, the second sequence should be an empty list.

        :param example: the input example to process
        :return: Two sequences of text. All text segments can optionally be marked as being shortenable.
        """
        pass

    @abstractmethod
    def verbalize(self, label) -> List[str]:
        """
        Return all verbalizations for a given label.

        :param label: the label
        :return: the list of verbalizations
        """
        pass

    def get_mask_positions(self, input_ids: List[int]) -> List[int]:
        label_idx = input_ids.index(self.mask_id)
        labels = [-1] * len(input_ids)
        labels[label_idx] = 1
        return labels

    def convert_mlm_logits_to_cls_logits(self, mlm_labels: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        masked_logits = logits[mlm_labels >= 0]
        cls_logits = torch.stack([self._convert_single_mlm_logits_to_cls_logits(ml) for ml in masked_logits])
        return cls_logits

    def _convert_single_mlm_logits_to_cls_logits(self, logits: torch.Tensor) -> torch.Tensor:
        m2c = self.mlm_logits_to_cls_logits_tensor.to(logits.device)
        # filler_len.shape() == max_fillers
        filler_len = torch.tensor([len(self.verbalize(label)) for label in self.wrapper.config.label_list],
                                  dtype=torch.float)
        filler_len = filler_len.to(logits.device)

        # cls_logits.shape() == num_labels x max_fillers  (and 0 when there are not as many fillers).
        cls_logits = logits[torch.max(torch.zeros_like(m2c), m2c)]
        cls_logits = cls_logits * (m2c > 0).float()

        # cls_logits.shape() == num_labels
        cls_logits = cls_logits.sum(axis=1) / filler_len
        return cls_logits

    def convert_plm_logits_to_cls_logits(self, logits: torch.Tensor) -> torch.Tensor:
        assert logits.shape[1] == 1
        logits = torch.squeeze(logits, 1)  # remove second dimension as we always have exactly one <mask> per example
        cls_logits = torch.stack([self._convert_single_mlm_logits_to_cls_logits(lgt) for lgt in logits])
        return cls_logits

    @staticmethod
    def _load_verbalizer_from_file(path: str, pattern_id: int):

        verbalizers = defaultdict(dict)  # type: Dict[int, Dict[str, List[str]]]
        current_pattern_id = None

        with open(path, 'r') as fh:
            for line in fh.read().splitlines():
                if line.isdigit():
                    current_pattern_id = int(line)
                elif line:
                    label, *realizations = line.split()
                    verbalizers[current_pattern_id][label] = realizations

        logger.info("Automatically loaded the following verbalizer: \n {}".format(verbalizers[pattern_id]))

        def verbalize(label) -> List[str]:
            return verbalizers[pattern_id][label]

        return verbalize


class AgnewsPVP(PVP):
    VERBALIZER = {
        "0": ["politics"], # 'world'
        "1": ["sports"],
        "2": ["business"],
        "3": ["technology"]
    }

    def get_parts(self, example: InputExample) -> FilledPattern:

        text_a = self.shortenable(example.text_a)
        text_b = self.shortenable(example.text_b)
        # print(text_a, 'tb:', text_b)
        if self.pattern_id == 0:
            return [self.mask, ':', text_a], []
        elif self.pattern_id == 1:
            return [self.mask, 'News:', text_a], []
        elif self.pattern_id == 2:
            return ['(', self.mask, ')', text_a], []
        elif self.pattern_id == 3:
            return [text_a, '(', self.mask, ')'], []
        elif self.pattern_id == 4:
            return ['[ Category:', self.mask, ']', text_a], []
        elif self.pattern_id == 5:
            return [self.mask, '-', text_a], []
        elif self.pattern_id == 6: # 72.06
            return ['The category of', text_a, 'is', self.mask], []
        else:
            raise ValueError("No pattern implemented for id {}".format(self.pattern_id))

    def verbalize(self, label) -> List[str]:
        return AgnewsPVP.VERBALIZER[label]
    
    def get_all_label_id(self) -> List:
        idx = []
        for v in AgnewsPVP.VERBALIZER:
            text = AgnewsPVP.VERBALIZER[v][0]
            id = get_verbalization_ids(" " +text, self.tokenizer, force_single_token =True)
            idx.append(id)
        return idx

    def get_pmi_parts(self, example: InputExample) -> FilledPattern:
        text_a = self.shortenable(example.text_a)
        text_b = self.shortenable(example.text_b)
        if self.pattern_id == 0: # 73.96
            return [self.mask, ':'], []
        elif self.pattern_id == 1: # 0.7215
            return [self.mask, 'News:'], []
        elif self.pattern_id == 2:
            return ['(', self.mask, ')'], []
        elif self.pattern_id == 3: # 72.06
            return ['(', self.mask, ')'], []
        elif self.pattern_id == 4: # 
            return ['[ Category:', self.mask, ']'], []
        elif self.pattern_id == 5:
            return [self.mask, '-'], []
        elif self.pattern_id == 6:
            return ['The category of is', self.mask], []
        elif self.pattern_id == -1:
            return [self.mask], []
        else:
            raise ValueError("No pattern implemented for id {}".format(self.pattern_id))

            
class DBPediaPVP(PVP):
    VERBALIZER = {
        "0": ["company"],
        "1": ["school"],
        "2": ["artist"],
        "3": ["athlete"],
        "4": ["politics"],
        "5": ["transportation"],
        "6": ["building"],
        "7": ["mountain"],
        "8": ["village"],
        "9": ["animal"],
        "10": ["plant"],
        "11": ["album"],
        "12": ["film"],
        "13": ["book"],
    }

    def get_parts(self, example: InputExample) -> FilledPattern:

        text_a = self.shortenable(example.text_a)
        text_b = self.shortenable(example.text_b)
        if self.pattern_id == 0: # 73.96
            return [text_a, text_b, text_a, 'is a', self.mask], []
        elif self.pattern_id == 1: # 0.7215
            return [text_a, text_b, 'the type of', text_a, 'is', self.mask], []
        elif self.pattern_id == 2:
            return [text_a, '(', self.mask, ')', text_b], []
        elif self.pattern_id == 3: # 72.06
            return [text_a, text_b, 'the category of', text_a, 'is', self.mask], []
        elif self.pattern_id == 4: # 
            return ['[ Category:', self.mask, ']', text_a, text_b], []
        elif self.pattern_id == 5:
            return [self.mask, '-', text_a, text_b], []
        else:
            raise ValueError("No pattern implemented for id {}".format(self.pattern_id))

    def verbalize(self, label) -> List[str]:
        return DBPediaPVP.VERBALIZER[label]
    
    def get_all_label_id(self) -> List:
        idx = []
        for v in DBPediaPVP.VERBALIZER:
            text = DBPediaPVP.VERBALIZER[v][0]
            id = get_verbalization_ids(" " +text, self.tokenizer, force_single_token = True)
            idx.append(id)
        return idx

    def get_pmi_parts(self, example: InputExample) -> FilledPattern:

        text_a = self.shortenable(example.text_a)
        text_b = self.shortenable(example.text_b)
        if self.pattern_id == 0: # 73.96
            return [text_a, 'is a', self.mask], []
        elif self.pattern_id == 1: # 0.7215
            return ['the type of', text_a, 'is', self.mask], []
        elif self.pattern_id == 2:
            return ['(', self.mask, ')'], []
        elif self.pattern_id == 3: # 72.06
            return ['the category of', text_a, 'is', self.mask], []
        elif self.pattern_id == 4: # 
            return ['[ Category:', self.mask, ']'], []
        elif self.pattern_id == 5:
            return [self.mask, '-'], []
        elif self.pattern_id == -1:
            return [self.mask], []
        else:
            raise ValueError("No pattern implemented for id {}".format(self.pattern_id))


class YahooPVP(PVP):
    VERBALIZER = {
        "0": ["society"],
        "1": ["science"],
        "2": ["health"],
        "3": ["education"],
        "4": ["computer"],
        "5": ["sports"],
        "6": ["business"],
        "7": ["entertainment"],
        "8": ["relationship"],
        "9": ["politics"],
    }

    def get_parts(self, example: InputExample) -> FilledPattern:

        text_a = self.shortenable(example.text_a)
        text_b = self.shortenable(example.text_b)
        # print(text_a, text_b)
        if self.pattern_id == 0:
            return [self.mask, ':', text_a, text_b], []
        elif self.pattern_id == 1:
            return [self.mask, 'Question:', text_a, text_b], []
        elif self.pattern_id == 2:
            return [text_a, '(', self.mask, ')', text_b], []
        elif self.pattern_id == 3:
            return [text_a, text_b, '(', self.mask, ')'], []
        elif self.pattern_id == 4: # 49
            return ['[ Category:', self.mask, ']', text_a, text_b], []
        elif self.pattern_id == 5:
            return [self.mask, '-', text_a, text_b], []
        elif self.pattern_id == 6:
            return [text_a, text_b, text_a, 'is a', self.mask], []
        elif self.pattern_id == 7: # 47
            return [text_a, text_b, 'the type of', text_a, 'is', self.mask], []
        elif self.pattern_id == 8: # 49
            return [text_a, text_b, 'the category of', text_a, 'is', self.mask], []
        else:
            raise ValueError("No pattern implemented for id {}".format(self.pattern_id))

    def verbalize(self, label) -> List[str]:
        return YahooPVP.VERBALIZER[label]

    def get_all_label_id(self) -> List:
        idx = []
        for v in YahooPVP.VERBALIZER:
            text = YahooPVP.VERBALIZER[v][0]
            id = get_verbalization_ids(" " + text, self.tokenizer, force_single_token = True)
            idx.append(id)
        return idx
    
    def get_pmi_parts(self, example: InputExample) -> FilledPattern:

        text_a = self.shortenable(example.text_a)
        text_b = self.shortenable(example.text_b)
        # print(text_a, text_b)
        if self.pattern_id == 0:
            return [self.mask, ':'], []
        elif self.pattern_id == 1:
            return [self.mask, 'Question:'], []
        elif self.pattern_id == 2:
            return ['(', self.mask, ')'], []
        elif self.pattern_id == 3:
            return ['(', self.mask, ')'], []
        elif self.pattern_id == 4: # 49
            return ['[ Category:', self.mask, ']'], []
        elif self.pattern_id == 5:
            return [self.mask, '-'], []
        elif self.pattern_id == 6:
            return [ text_a, 'is a', self.mask], []
        elif self.pattern_id == 7: # 47
            return ['the type of', text_a, 'is', self.mask], []
        elif self.pattern_id == 8: # 49
            return ['the category of', text_a, 'is', self.mask], []
        elif self.pattern_id == -1:
            return [self.mask], []
        else:
            raise ValueError("No pattern implemented for id {}".format(self.pattern_id))


class MnliPVP(PVP):
    VERBALIZER_A = {
        "0": ["wrong"],
        "2": ["right"],
        "1": ["maybe"]
    }
    VERBALIZER_B = {
        "0": ["no"],
        "2": ["yes"],
        "1": ["maybe"]
    }

    def get_parts(self, example: InputExample) -> FilledPattern:
        text_a = self.shortenable(self.remove_final_punc(example.text_a))
        text_b = self.shortenable(example.text_b)

        if self.pattern_id == 0 or self.pattern_id == 2:
            return ['"', text_a, '" ?', self.mask, ', "', text_b, '"'], []
        elif self.pattern_id == 1 or self.pattern_id == 3:
            return [text_a, '?', self.mask, ',', text_b], []

    def verbalize(self, label) -> List[str]:
        if self.pattern_id == 0 or self.pattern_id == 1:
            return MnliPVP.VERBALIZER_A[label]
        return MnliPVP.VERBALIZER_B[label]
    
    def get_all_label_id(self) -> List[int]:
        idx = []
        if self.pattern_id in [0,1 ,4]: # or self.pattern_id == 1 or :
            MnliPVP.VERBALIZER = MnliPVP.VERBALIZER_A
        else:
            MnliPVP.VERBALIZER = MnliPVP.VERBALIZER_B

        for v in MnliPVP.VERBALIZER:
            text = MnliPVP.VERBALIZER[v][0]
            id = get_verbalization_ids(" " + text, self.tokenizer, force_single_token = True)
            idx.append(id)
        return idx

    def get_pmi_parts(self, example: InputExample) -> FilledPattern:
        text_a = self.shortenable(self.remove_final_punc(example.text_a))
        text_b = self.shortenable(example.text_b)

        if self.pattern_id == 0 or self.pattern_id == 2:
            return ['"', '" ?', self.mask, ', "', '"'], []
        elif self.pattern_id == 1 or self.pattern_id == 3:
            return ['?', self.mask, ','], []
        elif self.pattern_id == 4 or self.pattern_id == 5:
            return ['?', self.mask, ','], []
        elif self.pattern_id == -1:
            return [self.mask], []

class SST5PVP(PVP):
    VERBALIZER = {
        "0": ["terrible"],
        "1": ["bad"],
        "2": ["okay"],
        "3": ["good"],
        "4": ["great"]
    }

    def get_parts(self, example: InputExample) -> FilledPattern:
        text = self.shortenable(example.text_a)

        if self.pattern_id == 0:
            return ['It\'s', self.mask, '.', text], []
        elif self.pattern_id == 1:
            return [text, '. All in all, it was', self.mask, '.'], []
        elif self.pattern_id == 2:
            return ['A', self.mask, "movie, ", text], []
        elif self.pattern_id == 3:
            return [text, 'In summary, the movie is', self.mask, '.'], []
        elif self.pattern_id == 4:
            return [text, '. In other words, it is', self.mask, '.'], []
        elif self.pattern_id == 5:
            return [text, '. In other words, the movie is', self.mask, '.'], []
        else:
            raise ValueError("No pattern implemented for id {}".format(self.pattern_id))

    def get_pmi_parts(self, example: InputExample) -> FilledPattern:
        text = self.shortenable(example.text_a)

        if self.pattern_id == 0:
            return ['It\'s', self.mask, '.'], []
        elif self.pattern_id == 1:
            return ['All in all, it was', self.mask, '.'], []
        elif self.pattern_id == 2:
            return ['A', self.mask, "movie"], []
        elif self.pattern_id == 3:
            return ['In summary, the movie is', self.mask, '.'], []
        elif self.pattern_id == 4:
            return ['. In other words, it is', self.mask, '.'], []
        elif self.pattern_id == 5:
            return ['. In other words, the movie is', self.mask, '.'], []
        elif self.pattern_id == -1:
            return [self.mask], []
        else:
            raise ValueError("No pattern implemented for id {}".format(self.pattern_id))
    
    def verbalize(self, label) -> List[str]:
        return SST5PVP.VERBALIZER[label]
    
    def get_all_label_id(self) -> List[int]:
        idx = []
        for v in SST5PVP.VERBALIZER:
            text = SST5PVP.VERBALIZER[v][0]
            id = get_verbalization_ids(" " + text, self.tokenizer, force_single_token = True)
            idx.append(id)
        return idx

class SST2PVP(PVP):
    VERBALIZER = {
        "0": ["terrible"], # terrible
        "1": ["great"]
    }

    def get_parts(self, example: InputExample) -> FilledPattern:
        # print(example.text_a)
        text = self.shortenable(example.text_a)
        # text[0] = re.sub(r'[^\w\s]', '', text[0]).strip()
        # print(text)
        if self.pattern_id == 0: # 77.46
            return [text, 'It was', self.mask, '.'], []
        elif self.pattern_id == 1: # 67.59
            return [text, '. All in all, it was', self.mask, '.'], []
        elif self.pattern_id == 2: # 67.48
            return ['A', self.mask, "movie, ", text], []
        elif self.pattern_id == 3: # 76.80
            return [text, 'In summary, the movie is', self.mask, '.'], []
        elif self.pattern_id == 4: # 80.56
            return [text, 'In other words, it is', self.mask, '.'], []
        elif self.pattern_id == 5: # 77.57
            return [text, 'In summary, the movie was', self.mask, '.'], []
        elif self.pattern_id == 6: # 0.7782
            return [text, 'In other words, it was', self.mask, '.'], []
        else:
            raise ValueError("No pattern implemented for id {}".format(self.pattern_id))

    def verbalize(self, label) -> List[str]:
        return SST2PVP.VERBALIZER[label]
    
    def get_all_label_id(self) -> List[int]:
        idx = []
        for v in SST2PVP.VERBALIZER:
            text = SST2PVP.VERBALIZER[v][0]
            id = get_verbalization_ids(" " + text, self.tokenizer, force_single_token = True)
            idx.append(id)
        return idx
    
    def get_pmi_parts(self, example: InputExample) -> FilledPattern:
        text = self.shortenable(example.text_a)

        if self.pattern_id == 0:
            return ['It was', self.mask, '.'], []
        elif self.pattern_id == 1:
            return ['All in all, it was', self.mask, '.'], []
        elif self.pattern_id == 2:
            return ['A', self.mask, "movie"], []
        elif self.pattern_id == 3:
            return ['In summary, the movie is', self.mask, '.'], []
        elif self.pattern_id == 4:
            return ['In other words, it is', self.mask, '.'], []
        # elif self.pattern_id == 5:
            # return ['. In other words, the movie is', self.mask, '.'], []
        elif self.pattern_id == -1:
            return [self.mask], []
        else:
            raise ValueError("No pattern implemented for id {}".format(self.pattern_id))
    

class IMDBPVP(PVP):
    VERBALIZER = {
        "0": ["terrible"],
        "1": ["great"]
    }

    def get_parts(self, example: InputExample) -> FilledPattern:
        text = self.shortenable(example.text_a)

        if self.pattern_id == 0: # Accu: 0.6552
            return [text, '. It was', self.mask, '.'], []
        elif self.pattern_id == 1: # Accu: 0.6502
            return [text, '. All in all, it was', self.mask, '.'], []
        elif self.pattern_id == 2: # 50.66
            return ['A', self.mask, "movie", ".", text], []
        elif self.pattern_id == 3: # 0.7238
            return [text, 'In summary, the movie is', self.mask, '.'], []
        elif self.pattern_id == 4: # 0.7268
            return [text, 'In other words, it is', self.mask, '.'], []
        else:
            raise ValueError("No pattern implemented for id {}".format(self.pattern_id))

    def verbalize(self, label) -> List[str]:
        return IMDBPVP.VERBALIZER[label]

    def get_all_label_id(self) -> List:
        idx = []
        for v in IMDBPVP.VERBALIZER:
            text = IMDBPVP.VERBALIZER[v][0]
            id = get_verbalization_ids(" " + text, self.tokenizer, force_single_token= True)
            idx.append(id)
        return idx
    
    def get_pmi_parts(self, example: InputExample) -> FilledPattern:
        text = self.shortenable(example.text_a)

        if self.pattern_id == 0:
            return ['It was', self.mask, '.'], []
        elif self.pattern_id == 1:
            return ['All in all, it was', self.mask, '.'], []
        elif self.pattern_id == 2:
            return ['A', self.mask, "movie"], []
        elif self.pattern_id == 3:
            return ['In summary, the movie is', self.mask, '.'], []
        elif self.pattern_id == 4:
            return ['In other words, it is', self.mask, '.'], []
        elif self.pattern_id == -1:
            return [self.mask], []
        else:
            raise ValueError("No pattern implemented for id {}".format(self.pattern_id))
    
class TrecPVP(PVP):
    VERBALIZER = {
        "0": ["Expression"],
        "1": ["Entity"],
        "2": ["Description"],
        "3": ["Human"],
        "4": ["Location"],
        "5": ["Number"]
    }

    def get_parts(self, example: InputExample) -> FilledPattern:
        text = self.shortenable(example.text_a)

        if self.pattern_id == 0: # Accu: 0.6552
            return [text, '. It is', self.mask, '.'], []
        elif self.pattern_id == 1: # Accu: 0.6552
            return [text, '. It is a', self.mask, 'question .'], []
        elif self.pattern_id == 2: # Accu: 0.6552
            return [self.mask, ': ', text], []
        elif self.pattern_id == 3: # Accu: 
            return ['A', self.mask, 'Question : ', text], []
        else:
            raise ValueError("No pattern implemented for id {}".format(self.pattern_id))

    def verbalize(self, label) -> List[str]:
        return TrecPVP.VERBALIZER[label]

    def get_all_label_id(self) -> List:
        idx = []
        for v in TrecPVP.VERBALIZER:
            text = TrecPVP.VERBALIZER[v][0]
            id = get_verbalization_ids(" " + text, self.tokenizer, force_single_token= True)
            idx.append(id)
        return idx
    
    def get_pmi_parts(self, example: InputExample) -> FilledPattern:
        text = self.shortenable(example.text_a)
        if self.pattern_id == 0: # Accu: 
            return ['. It was', self.mask, '.'], []
        elif self.pattern_id == 1: # Accu: 0.6552
            return [text, '. It was', self.mask, '.'], []
        elif self.pattern_id == 2: # Accu: 
            return [self.mask, ': '], []
        elif self.pattern_id == 3: # Accu: 
            return ['A', self.mask, 'Question : '], []
        elif self.pattern_id == -1:
            return [self.mask], []
        else:
            raise ValueError("No pattern implemented for id {}".format(self.pattern_id))


class YelpPolarityPVP(PVP):
    VERBALIZER = {
        "0": ["terrible"],
        "1": ["great"]
    }

    def get_parts(self, example: InputExample) -> FilledPattern:
        text = self.shortenable(example.text_a)

        if self.pattern_id == 0:
            return [text, '. It was', self.mask, '.'], []
        elif self.pattern_id == 1:
            return [text, '. All in all, it was', self.mask, '.'], []
        elif self.pattern_id == 2:
            return ['Just', self.mask, "!", text], []
        elif self.pattern_id == 3:
            return [text, '. In summary, the restaurant is', self.mask, '.'], []
        elif self.pattern_id == 4:
            return [text, '. In other words, it is', self.mask, '.'], []
        else:
            raise ValueError("No pattern implemented for id {}".format(self.pattern_id))

    def verbalize(self, label) -> List[str]:
        return YelpPolarityPVP.VERBALIZER[label]

    def get_all_label_id(self) -> List:
        idx = []
        for v in YelpPolarityPVP.VERBALIZER:
            text = YelpPolarityPVP.VERBALIZER[v][0]
            id = get_verbalization_ids(" " + text, self.tokenizer, force_single_token=True)
            idx.append(id)
        return idx
    
    def get_pmi_parts(self, example: InputExample) -> FilledPattern:
        text = self.shortenable(example.text_a)

        if self.pattern_id == 0:
            return ['It was', self.mask, '.'], []
        elif self.pattern_id == 1:
            return ['All in all, it was', self.mask, '.'], []
        elif self.pattern_id == 2:
            return ['Just', self.mask, "!"], []
        elif self.pattern_id == 3:
            return ['In summary, the restaurant is', self.mask, '.'], []
        elif self.pattern_id == 4:
            return ['In other words, it is', self.mask, '.'], []
        elif self.pattern_id == -1:
            return [self.mask], []
        else:
            raise ValueError("No pattern implemented for id {}".format(self.pattern_id))
    

class YelpFullPVP(YelpPolarityPVP):
    VERBALIZER = {
        "0": ["terrible"],
        "1": ["bad"],
        "2": ["okay"],
        "3": ["good"],
        "4": ["great"]
    }

    def verbalize(self, label) -> List[str]:
        return YelpFullPVP.VERBALIZER[label]

    def get_all_label_id(self) -> List:
        idx = []
        for v in YelpFullPVP.VERBALIZER:
            text = YelpFullPVP.VERBALIZER[v][0]
            id = get_verbalization_ids(" " + text, self.tokenizer, force_single_token=True)
            idx.append(id)
        return idx



PVPS = {
    'agnews': AgnewsPVP,
    'SST-2': SST2PVP,
    'SST-5': SST5PVP,
    'imdb': IMDBPVP,
    # 'amazon-polarity' : AmazonPolarityPVP,
    'mnli': MnliPVP,
    # 'qnli': QnliPVP,
    'yelp-polarity': YelpPolarityPVP,
    'yelp-full': YelpFullPVP,
    'yahoo': YahooPVP,
    'dbpedia': DBPediaPVP,
    'trec': TrecPVP,
}

class Processor(object):
    """Processor for the text data set """
    def __init__(self, args):
        self.args = args
        #self.relation_labels = self.load_json(filename) # all possible labels
        #filename = args.data_dir + '/' + 'config.json'
        #label, num_label, label2id, id2label = self.load_info(filename)
        #self.relation_labels =
        #self.num_label = num_label
        #self.label2id = None
        #self.id2label = None
        self.n_sentence = 1
        if self.args.task in ['agnews']:
            self.num_label = 4
        elif self.args.task in ['mnli', 'qnli', 'snli']:
            self.n_sentence = 2
            self.num_label = 3
        elif self.args.task in ['yahoo', 'chemprot']:
            self.num_label = 10
        elif self.args.task in ['YelpReviewPolarity','imdb','youtube', 'amazon-polarity', 'SST-2', 'elec']:
            self.num_label = 2
        elif self.args.task in ['yelp-full','amazon-full', 'SST-5', 'pubmed']:
            self.num_label = 5
        elif self.args.task in ['trec']:
            self.num_label = 6
        elif self.args.task in ['dbpedia']:
            self.num_label = 14
        elif self.args.task in ['tacred']:
            self.num_label = 42
        #for i in range(self.num_label):
        self.relation_labels = [x for x in range(self.num_label)]
        self.label2id = {x:x for x in range(self.num_label)}
        self.id2label = {x:x for x in range(self.num_label)}

    def read_data(self, filename):
        path = filename
        with open(path, 'r') as f:
            data = f #json.load(f)
            for x in data:
                yield json.loads(x)
        # return data

    def _create_examples(self, data, set_type):
        examples = []
        for i, d in enumerate(data):
            guid = "%s-%s" % (set_type, i)
            if self.n_sentence == 1:
                if self.args.task in ['dbpedia', 'yahoo']:
                    text = d["text"].split("  ")
                    title, body = text[0], " ".join(text[1:])
                    text_a = title.replace('\\n', ' ').replace('\\', ' ').strip()
                    text_b = body.replace('\\n', ' ').replace('\\', ' ').strip()
                    text_b = re.sub(r'[^\w\s]', '', text_b)
                    # exit()
                else:
                    text_a = d["text"].replace('\\n', ' ').replace('\\', ' ').strip()

            else:
                text_a = d["text_a"].replace('\\n', ' ').replace('\\', ' ').strip()
                text_b = d["text_b"].replace('\\n', ' ').replace('\\', ' ').strip()
                text_b = re.sub(r'[^\w\s]', '', text_b)
            text_a = re.sub(r'[^\w\s]', '', text_a)
            # text_b = re.sub(r'[^\w\s]', '', text_b)
            label = d["_id"] 

            if i % 5000 == 0:
                logger.info(d)
            if self.n_sentence == 1 and self.args.task not in ['dbpedia', 'yahoo']:
                examples.append(InputExample(guid=guid, text_a=text_a, label=label, task = self.args.task))
            else:
                examples.append(InputExample(guid=guid, text_a=text_a, text_b = text_b, task = self.args.task, label=label))

        return examples

    def get_examples(self, mode):
        """
        Args:
            mode: train, dev, test
        """
        file_to_read = None
        if mode == 'train':
            file_to_read = self.args.train_file
        elif mode == 'dev':
            file_to_read = self.args.dev_file
        elif mode == 'test':
            file_to_read = self.args.test_file
        elif mode == 'unlabeled':
            file_to_read = self.args.unlabel_file

        logger.info("LOOKING AT {}".format(os.path.join(self.args.data_dir, file_to_read)))
        return self._create_examples(self.read_data(os.path.join(self.args.data_dir, file_to_read)), mode)

def load_and_cache_examples(args, tokenizer, mode, template_id = 0, size = -1):
    processor = Processor(args)
    if mode in ["dev", "test"]:
        cached_features_file = os.path.join(
            args.data_dir,
            'cached_{}_{}_{}_{}_temp{}_prompt.pt'.format(
                mode,
                args.task,
                list(filter(None, args.model_name_or_path.split("/"))).pop(),
                args.max_seq_len,
                template_id
            )
        )
    else:
        cached_features_file = os.path.join(
            args.data_dir,
            'cached_{}_{}_{}_{}_{}_{}_temp{}_prompt.pt'.format(
                mode,
                args.task,
                list(filter(None, args.model_name_or_path.split("/"))).pop(),
                args.max_seq_len,
                args.al_method,
                args.sample_labels,
                template_id
            )
        )
    if os.path.exists(cached_features_file) and args.auto_load:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        if mode == "train":
            examples = processor.get_examples("train")
        elif mode == "dev":
            examples = processor.get_examples("dev")
        elif mode == "test":
            examples = processor.get_examples("test")
        else:
            raise Exception("For mode, Only train, dev, test is available")
        features = convert_examples_to_features_prompt(examples, args.max_seq_len, tokenizer, pattern_id=template_id, add_sep_token=args.add_sep_token)
        # features_zero = convert_examples_to_features_prompt_zero(examples, args.max_seq_len, tokenizer, pattern_id=template_id, add_sep_token=args.add_sep_token)
        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)
        
    # Convert to Tensors and build dataset
    if size > 0:
        import random 
        random.shuffle(features)
        features = features[:size]
    else:
        size = len(features)
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)

    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    all_ids = torch.tensor([ _ for _,f in enumerate(features)], dtype=torch.long)
    all_mask_ids = torch.tensor([f.mask_id for f in features], dtype=torch.long)
    all_mask_label_ids = torch.tensor([f.all_label_id for f in features], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_attention_mask,
                                all_token_type_ids, all_label_ids, all_ids, all_mask_ids, all_mask_label_ids)
    return dataset, processor.num_label, size

def load_and_cache_unlabeled_examples(args, tokenizer, mode, template_id = 0, train_size = 100, size = -1):
    processor = Processor(args)

    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        args.data_dir,
        'cached_{}_{}_{}_{}_temp{}_unlabel_prompt.pt'.format(
            mode,
            args.task,
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            args.max_seq_len,
            template_id
        )
    )

    cached_features_file_zero = os.path.join(
            args.data_dir,
           'cached_{}_{}_{}_{}_no_temp_unlabel_prompt.pt'.format(
            mode,
            args.task,
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            args.max_seq_len,
            )
        )
    cached_features_file_temp = os.path.join(
            args.data_dir,
            'cached_{}_{}_{}_{}_temp{}_unlabel_prompt_pmi.pt'.format(
            mode,
            args.task,
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            args.max_seq_len,
            template_id
        )
        )

    if os.path.exists(cached_features_file) and os.path.exists(cached_features_file_temp) \
        and os.path.exists(cached_features_file_zero) and args.auto_load:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
        features_zero = torch.load(cached_features_file_zero)
        features_temp = torch.load(cached_features_file_temp)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)

        assert mode == "unlabeled"
        examples = processor.get_examples("unlabeled")
        
        features = convert_examples_to_features_prompt(examples, args.max_seq_len, tokenizer, pattern_id=template_id, add_sep_token=args.add_sep_token)
        features_zero = convert_examples_to_features_prompt_zero(examples, args.max_seq_len, tokenizer, pattern_id=-1, add_sep_token=args.add_sep_token)
        features_temp = convert_examples_to_features_prompt_zero(examples, args.max_seq_len, tokenizer, pattern_id=template_id, add_sep_token=args.add_sep_token)

        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)
        torch.save(features_zero, cached_features_file_zero)
        torch.save(features_temp, cached_features_file_temp)

        

    # Convert to Tensors and build dataset
    if size > 0:
        features = features[:size]
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)

    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    all_ids = torch.tensor([_+train_size for _ ,f in enumerate(features)], dtype=torch.long)
    all_mask_ids = torch.tensor([f.mask_id for f in features], dtype=torch.long)
    all_mask_label_ids = torch.tensor([f.all_label_id for f in features], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_attention_mask,
                            all_token_type_ids, all_label_ids, all_ids, all_mask_ids, all_mask_label_ids)


    z_all_input_ids = torch.tensor([f.input_ids for f in features_zero], dtype=torch.long)
    z_all_attention_mask = torch.tensor([f.attention_mask for f in features_zero], dtype=torch.long)
    z_all_token_type_ids = torch.tensor([f.token_type_ids for f in features_zero], dtype=torch.long)
    z_all_label_ids = torch.tensor([f.label_id for f in features_zero], dtype=torch.long)
    z_all_ids = torch.tensor([_+train_size for _ ,f in enumerate(features_zero)], dtype=torch.long)
    z_all_mask_ids = torch.tensor([f.mask_id for f in features_zero], dtype=torch.long)
    z_all_mask_label_ids = torch.tensor([f.all_label_id for f in features_zero], dtype=torch.long)
    z_dataset = TensorDataset(z_all_input_ids, z_all_attention_mask,
                            z_all_token_type_ids, z_all_label_ids, z_all_ids, z_all_mask_ids, z_all_mask_label_ids)

    t_all_input_ids = torch.tensor([f.input_ids for f in features_temp], dtype=torch.long)
    t_all_attention_mask = torch.tensor([f.attention_mask for f in features_temp], dtype=torch.long)
    t_all_token_type_ids = torch.tensor([f.token_type_ids for f in features_temp], dtype=torch.long)
    t_all_label_ids = torch.tensor([f.label_id for f in features_temp], dtype=torch.long)
    t_all_ids = torch.tensor([_+train_size for _ ,f in enumerate(features_temp)], dtype=torch.long)
    t_all_mask_ids = torch.tensor([f.mask_id for f in features_temp], dtype=torch.long)
    t_all_mask_label_ids = torch.tensor([f.all_label_id for f in features_temp], dtype=torch.long)
    t_dataset = TensorDataset(t_all_input_ids, t_all_attention_mask,
                            t_all_token_type_ids, t_all_label_ids, t_all_ids, t_all_mask_ids, t_all_mask_label_ids)

    return dataset, processor.num_label, len(features), z_dataset, t_dataset

class InputFeatures(object):
    """
    A single set of features of data.
    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            Usually  ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded) tokens.
        token_type_ids: Segment token indices to indicate first and second portions of the inputs.
    """

    def __init__(self, input_ids, attention_mask, token_type_ids, label_id, 
                all_label_id= None, e1_mask = None, e2_mask = None, keys=None, mask_id= None):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label_id = label_id
        self.all_label_id = all_label_id
        self.e1_mask = e1_mask
        self.e2_mask = e2_mask
        self.mask_id = mask_id
        self.keys=keys

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


def convert_examples_to_features_prompt(examples, max_seq_len, tokenizer,
                                 pattern_id = 0,
                                 cls_token_segment_id=0,
                                 pad_token=0,
                                 pad_token_segment_id=0,
                                 sequence_a_segment_id=0,
                                 add_sep_token=False,
                                 mask_padding_with_zero=True,
                                ):
    features = []
    sample_per_example = 3
    for (ex_index, example) in enumerate(examples):
        if ex_index == 0:
            pvp = PVPS[example.task](tokenizer, pattern_id = pattern_id)
            all_label_id = pvp.get_all_label_id()
        if ex_index % 5000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        #print(example.text_a)

        # input_ids, token_type_ids = PVPS[example.task].encode(example)
        if example.task in ['mnli']:
            parts_a, parts_b = pvp.get_parts(example)
            # TODO
            # tokens_a = tokenizer.tokenize(example.text_a)
            # tokens_b = tokenizer.tokenize(example.text_b)
        else:
            parts_a, parts_b = pvp.get_parts(example)
            # tokens_a = tokenizer.tokenize(example.text_a)
        kwargs = {'add_prefix_space': True} if isinstance(tokenizer, RobertaTokenizer) else {}
        # print(parts_a,)
        parts_a = [x if isinstance(x, tuple) else (x, False) for x in parts_a]
        # print(parts_a,)
        text_a = [(tokenizer.tokenize(x,  **kwargs), s) for x, s in parts_a if x]
        parts_a = [(tokenizer.encode(x, **kwargs), s) for x, s in parts_a if x]
        # print(parts_a,text_a)
        if parts_b:
            parts_b = [x if isinstance(x, tuple) else (x, False) for x in parts_b]
            text_b = [(tokenizer.tokenize(x,  **kwargs), s) for x, s in parts_b if x]
            parts_b = [(tokenizer.encode(x, **kwargs), s) for x, s in parts_b if x]
        else:
            text_b = []
        # print(parts_a, parts_b, text_a, text_b)
        pvp.truncate(parts_a, parts_b, max_length = max_seq_len)
        pvp.truncate(text_a, text_b, max_length = max_seq_len)

        # print(  parts_a, parts_b,text_a, text_b)
        tokens_a = [token_id for part, _ in parts_a for token_id in part]
        tokens_b = [token_id for part, _ in parts_b for token_id in part] if parts_b else None
        # print(tokens_a, tokens_b, all_label_id)
       
        input_text =  [token_id for part, _ in text_a for token_id in part]
        input_ids = tokens_a
        # print(len(tokens_a),  len(input_text), len(text_b))
        if tokens_b is not None:
            input_ids += tokens_b
            input_text += [token_id for part, _ in text_b for token_id in part]
        # assert len(input_ids) == len(input_text), 'sequence of input_ids must equal to text, now %d vs. %d'%(len(input_ids), len(input_text))
        assert len(pvp.verbalize(str(example.label))) == 1, 'priming only supports one verbalization per label'
        input_ids = tokenizer.build_inputs_with_special_tokens(tokens_a, tokens_b)
        mask_idx = input_ids.index(pvp.mask_id)
        assert mask_idx >= 0, 'sequence of input_ids must contain a mask token'
        token_type_ids = [sequence_a_segment_id] * len(input_ids)
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
        # print(input_ids, token_type_ids)
        # print(mask_idx, example.label)
        # Zero-pad up to the sequence length.
        padding_length = max_seq_len - len(input_ids)
        input_ids = input_ids + ([pad_token] * padding_length)
        attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)
        assert len(input_ids) == max_seq_len, "Error with input length {} vs {}".format(len(input_ids), max_seq_len)
        assert len(attention_mask) == max_seq_len, "Error with attention mask length {} vs {}".format(len(attention_mask), max_seq_len)
        assert len(token_type_ids) == max_seq_len, "Error with token type length {} vs {}".format(len(token_type_ids), max_seq_len)
        label_id = int(example.label)
        features.append(
            InputFeatures(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            label_id = label_id,
                            mask_id = mask_idx,
                            all_label_id=all_label_id
                          )
            )
        # exit()
        # print(text_a, text_b, z)
        # exit()

        # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        # if add_sep_token:
        #     special_tokens_count = 2
        # else:
        #     special_tokens_count = 1
        # if len(tokens_a) > max_seq_len - special_tokens_count:
        #     tokens_a = tokens_a[:(max_seq_len - special_tokens_count)]

        # tokens = tokens_a
        # if add_sep_token:
        #     sep_token = tokenizer.sep_token
        #     tokens += [sep_token]

        # token_type_ids = [sequence_a_segment_id] * len(tokens)
        # cls_token = tokenizer.cls_token
        # tokens = [cls_token] + tokens
        # token_type_ids = [cls_token_segment_id] + token_type_ids
        # #tokens[0] = "$"
        # #tokens[1] = "<e2>"
        # input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # # tokens are attended to.
        # attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # # Zero-pad up to the sequence length.
        # padding_length = max_seq_len - len(input_ids)
        # input_ids = input_ids + ([pad_token] * padding_length)
        # attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        # token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)


        # assert len(input_ids) == max_seq_len, "Error with input length {} vs {}".format(len(input_ids), max_seq_len)
        # assert len(attention_mask) == max_seq_len, "Error with attention mask length {} vs {}".format(len(attention_mask), max_seq_len)
        # assert len(token_type_ids) == max_seq_len, "Error with token type length {} vs {}".format(len(token_type_ids), max_seq_len)

        # label_id = int(example.label)

        if ex_index < 3:
            logger.info("*** Example ***")
            logger.info("guid: %s" % example.guid)
            logger.info("tokens: %s" % " ".join([str(x) for x in input_text]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            logger.info("mask_id: %d" % mask_idx)
            logger.info("label: %s (id = %d)" % (example.label, label_id))
    return features


def convert_examples_to_features_prompt_zero(examples, max_seq_len, tokenizer,
                                 pattern_id = 0,
                                 cls_token_segment_id=0,
                                 pad_token=0,
                                 pad_token_segment_id=0,
                                 sequence_a_segment_id=0,
                                 add_sep_token=False,
                                 mask_padding_with_zero=True,
                                ):
    features = []
    sample_per_example = 3
    for (ex_index, example) in enumerate(examples):
        if ex_index == 0:
            pvp = PVPS[example.task](tokenizer, pattern_id = pattern_id)
            all_label_id = pvp.get_all_label_id()
        # input_ids, token_type_ids = PVPS[example.task].encode(example)
        if example.task in ['mnli']:
            print(example)
            parts_a, parts_b = pvp.get_pmi_parts(example)
            # TODO
            # tokens_a = tokenizer.tokenize(example.text_a)
            # tokens_b = tokenizer.tokenize(example.text_b)
        else:
            parts_a, parts_b = pvp.get_pmi_parts(example)
            # tokens_a = tokenizer.tokenize(example.text_a)
        kwargs = {'add_prefix_space': True} if isinstance(tokenizer, RobertaTokenizer) else {}
        # print(parts_a,)
        parts_a = [x if isinstance(x, tuple) else (x, False) for x in parts_a]
        # print(parts_a,)
        text_a = [(tokenizer.tokenize(x,  **kwargs), s) for x, s in parts_a if x]
        parts_a = [(tokenizer.encode(x, **kwargs), s) for x, s in parts_a if x]
        # print(parts_a,text_a)
        if parts_b:
            parts_b = [x if isinstance(x, tuple) else (x, False) for x in parts_b]
            text_b = [(tokenizer.tokenize(x,  **kwargs), s) for x, s in parts_b if x]
            parts_b = [(tokenizer.encode(x, **kwargs), s) for x, s in parts_b if x]
        else:
            text_b = []
        pvp.truncate(parts_a, parts_b, max_length = max_seq_len)
        pvp.truncate(text_a, text_b, max_length = max_seq_len)

        tokens_a = [token_id for part, _ in parts_a for token_id in part]
        tokens_b = [token_id for part, _ in parts_b for token_id in part] if parts_b else None
       
        input_text =  [token_id for part, _ in text_a for token_id in part]
        input_ids = tokens_a
        # print(len(tokens_a),  len(input_text), len(text_b))
        if tokens_b is not None:
            input_ids += tokens_b
            input_text += [token_id for part, _ in text_b for token_id in part]
        # assert len(input_ids) == len(input_text), 'sequence of input_ids must equal to text, now %d vs. %d'%(len(input_ids), len(input_text))
        assert len(pvp.verbalize(str(example.label))) == 1, 'priming only supports one verbalization per label'
        input_ids = tokenizer.build_inputs_with_special_tokens(tokens_a, tokens_b)
        mask_idx = input_ids.index(pvp.mask_id)
        assert mask_idx >= 0, 'sequence of input_ids must contain a mask token'
        token_type_ids = [sequence_a_segment_id] * len(input_ids)
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
        # Zero-pad up to the sequence length.
        padding_length = max_seq_len - len(input_ids)
        input_ids = input_ids + ([pad_token] * padding_length)
        attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)
        assert len(input_ids) == max_seq_len, "Error with input length {} vs {}".format(len(input_ids), max_seq_len)
        assert len(attention_mask) == max_seq_len, "Error with attention mask length {} vs {}".format(len(attention_mask), max_seq_len)
        assert len(token_type_ids) == max_seq_len, "Error with token type length {} vs {}".format(len(token_type_ids), max_seq_len)
        label_id = int(example.label)
        features.append(
            InputFeatures(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            label_id = label_id,
                            mask_id = mask_idx,
                            all_label_id=all_label_id
                          )
            )
        break
    return features

# InputExample(
#         guid = 0,
#         text_a = "Fu Xin was one of the greatest intellects of his time.",
#         text_b = "xxxx",
#         label = 1,
#     ),


# InputExample(
#         guid = 0,
#         text_a = "Fu Xin was one of the greatest intellects of his time.",
#         text_b = "xxxx",
#         label = 1,
#     ),

def load_and_cache_unlabeled_examples(args, tokenizer, mode, template_id = 0, train_size = 100, size = -1):
    pass

def load_datasets(args, mode = 'train', template_id = 0):
    samples = []
    if mode in ['dev']:
        file_name =  os.path.join(args.data_dir, args.dev_file)
    elif mode in ['test']:
        file_name =  os.path.join(args.data_dir, args.test_file)
    else:
        file_name =  os.path.join(args.data_dir, args.unlabel_file)
        # train_file=train_8.json --dev_file=valid_64.json --test_file=test.json  --unlabel_file=unlabeled.json \
    if mode in ["dev", "test"]:
        cached_features_file = os.path.join(
            args.cache_dir,
            'cached_{}_{}_{}_{}.pt'.format(
                mode,
                args.task,
                list(filter(None, args.model_name_or_path.split("/"))).pop(),
                args.max_seq_len,
            )
        )
    else:
        cached_features_file = os.path.join(
            args.cache_dir,
            'cached_{}_{}_{}_{}_{}_{}.pt'.format(
                mode,
                args.task,
                list(filter(None, args.model_name_or_path.split("/"))).pop(),
                args.max_seq_len,
                args.al_method,
                args.sample_labels,
            )
        )

    if os.path.exists(cached_features_file): 
        logger.info("Loading features from cached file %s", cached_features_file)
        samples = torch.load(cached_features_file)
    else:
        with open(file_name, 'r') as f: 
            i = 0
            for lines in f:
                i += 1
                d = json.loads(lines.strip())
                label = int(d["_id"])
                if args.task in ['dbpedia', 'yahoo']:
                    text = d["text"].split("  ")
                    title, body = text[0], " ".join(text[1:])
                    text_a = title.replace('\\n', ' ').replace('\\', ' ').strip()
                    text_b = body.replace('\\n', ' ').replace('\\', ' ').strip()
                    text_b = re.sub(r'[^\w\s]', '', text_b)  
                    # print('a', text_a,'\t\t', 'b', text_b)          
                    # exit()
                    example = InputExample(
                            guid = i,
                            text_a = text_a,
                            text_b = text_b, 
                            label = label,
                        )

                else:
                    text_a = d["text"].replace('\\n', ' ').replace('\\', ' ').strip()
                    text_b = None 
                    example = InputExample(
                            guid = i,
                            text_a = text_a,
                            label = label,
                        )
                samples.append(example)
                if i % 5000 == 0 :
                    logger.info("example: %s", example)
                logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(samples, cached_features_file)
    return samples

def index_select(train_id, unlabeled):
    return [unlabeled[i] for i in train_id]

def load_pt_utils(dataset = 'yahoo'):
    if dataset == 'yahoo': #<4>
        # template = '[ Category : {"mask"} ] {"placeholder":"text_a"} {"placeholder":"text_b"} .'
        template = '[ Category : {"mask"} ] {"placeholder": "text_a"} {"placeholder": "text_b"}'
        verbalizer = [['society'],['science'], ['health'], ['education'], ['computer'], ['sports'], ['business'], ['entertainment'], ['relationship'], ['politics']]

    elif dataset == 'imdb': #<0>
        template = '{"placeholder":"text_a"} It is {"mask"}.'
        verbalizer = [['terrible'], ['great']]
 
    elif dataset == 'yelp-full': #<3>
        template = '{"placeholder":"text_a"} In summary, the restaurant is {"mask"}.'
        verbalizer = [['terrible'], ['bad'], ['okay'], ['good'], ['great']]
        
    elif dataset == 'agnews': #<0>
        # self.mask, 'News:', text_a
        template = '{"mask"}: {"placeholder": "text_a"}.'
        verbalizer = [['politics'], ['sports'], ['business'], ['technology']]
        
    elif dataset == 'trec': #<0>
        template = '{"placeholder":"text_a"} It is {"mask"}.'
        verbalizer = [['Expression'], ['Entity'], ["Description"], ["Human"], ["Location"], ["Number"]]

    elif dataset == 'dbpedia': #<0>
        template = '{"placeholder":"text_a"} {"placeholder":"text_b"} {"placeholder": "text_a", "shortenable": False} is a {"mask"}.'
        verbalizer = [['company'], ['school'], ['artist'], ['athlete'], ['politics'], ['transportation'], ['building'], ['mountain'], ['village'], ['animal'], ['plant'], ['album'] , ['film'], ['book']]
        
    return template, verbalizer
