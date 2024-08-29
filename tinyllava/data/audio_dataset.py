import copy
from dataclasses import dataclass
import json
from typing import Dict,  Sequence, TYPE_CHECKING
from PIL import Image, ImageFile
import os

from .text_preprocess import TextPreprocess
from ..utils.arguments import DataArguments
from ..utils.constants import *
import librosa
import warnings


import transformers
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence



ImageFile.LOAD_TRUNCATED_IMAGES = True

class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        super(LazySupervisedDataset, self).__init__()
        list_data_dict = json.load(open(data_path, "r"))
        print("\n\nDATASET INITITALIZED\n\n")

        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.text_preprocess = TextPreprocess(tokenizer, data_args.conv_version)
        warnings.warn("Length of audio returns in __getitem__")


    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        # raise NotImplementedError
        warnings.warn("using 128 tokens for audio. Does this need to modify???")
        length_list = []
        for sample in self.list_data_dict:
            audio_tokens = 128 if 'audio' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + audio_tokens)
        return length_list

    @property
    def modality_lengths(self):
        # raise NotImplementedError
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'audio' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        print('\n\n')
        print(self.list_data_dict[i])
        print('\n\n')
        sources = self.list_data_dict[i]
        data_dict = self.text_preprocess(copy.deepcopy(sources["conversations"]))
        if 'audio' in sources:
            audio_file = self.list_data_dict[i]['audio']
            flac, sample_rate = librosa.load(audio_file, sr=44100)
            data_dict['audio'] = flac

        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            # print(f'{i}:{sources}')
            # crop_size = getattr(self.data_args.image_processor, 'crop_size', getattr(self.data_args.image_processor, 'size'))
            # data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
            raise NotImplementedError
        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            for input_id in input_ids:
                input_id[input_id == self.tokenizer.eos_token_id] = -300
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        labels = labels[:, :self.tokenizer.model_max_length]
        # FIXME: This is a hack for handling phi and stablelm, as they have the same eos, pad and unk. We want the model
        # FIXME: to predict the eos in the input ids, but we also use the id of eos to pad sequence, so we use a temp
        # FIXME: eos id first, and convert them back.
        if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            for input_id in input_ids:
                input_id[input_id == -300] = self.tokenizer.eos_token_id

        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
        )

        if 'audio' in instances[0]:
            audios = [instance['audio'] for instance in instances]
            wav_lengths = torch.LongTensor([len(wav) for wav in [audios]])
            wavs = pad_sequence([wavs], batch_first=True)
            batch['audios'] = audios
            batch['lengths'] = wav_lengths

        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                          data_path=data_args.data_path,
                                          data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)
