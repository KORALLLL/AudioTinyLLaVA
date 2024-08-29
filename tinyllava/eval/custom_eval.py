import argparse
import time

import torch
import os, sys
import json
from tqdm import tqdm
import shortuuid
sys.path.append('/home/user27/AudioTinyLLaVA')

from tinyllava.utils import *
from tinyllava.data import *
from tinyllava.model import *
from tinyllava.training_recipe import TrainingRecipeFactory

from torch.utils.data import Dataset, DataLoader

from PIL import Image
import math

from evaluate import load



@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        # input_ids = [instance['input_ids'] for instance in instances]
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
            wav_lengths = torch.LongTensor([len(wav) for wav in audios])
            audios = pad_audios(audios, target_len=512_000)
            audios = torch.from_numpy(audios)
            # audios = pad_sequence([audios], batch_first=True)
            batch['audios'] = audios
            batch['lengths'] = wav_lengths

        return batch


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


# Custom dataset class
class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 num_chunk, chunk_id):
        super(LazySupervisedDataset, self).__init__()
        list_data_dict = get_chunk(json.load(open(data_path, "r")), num_chunk, chunk_id)


        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.text_preprocess = TextPreprocess(tokenizer, "phi")
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
        # print("\n\n             Trying to get element in batch\n\n")
        sources = self.list_data_dict[i]
        
        data_dict = self.text_preprocess(copy.deepcopy(sources["conversations"]), mode='eval')
        # print(self.tokenizer(sources["conversations"][1]['value']).input_ids)
        data_dict['labels'] = torch.tensor(self.tokenizer(sources["conversations"][1]['value']).input_ids)
        # print(data_dict)
        # print("\n\n\n")
        # print(i, sources["conversations"][1]['value'])
        # data_dict
        if 'audio' in sources:
            audio_file = self.list_data_dict[i]['audio']
            flac, sample_rate = sf.read(audio_file)
            data_dict['audio'] = flac
        
        return data_dict




# DataLoader
def create_data_loader(data_path,
                        tokenizer,
                        batch_size=1, 
                        num_workers=1, 
                        num_chunks=None, 
                        chunk_id=None):
    assert batch_size == 1, "batch_size must be 1"
    dataset = LazySupervisedDataset(data_path, tokenizer, num_chunks, chunk_id)
    collator = DataCollatorForSupervisedDataset(tokenizer)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collator)
    return data_loader

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    # TODO: make it normal, not this shiiiiiit
    training_recipe: str = field(default='common')
    tune_type_llm: str = field(default="frozen") # support only: frozen, full, lora, qlora_int4, qlora_int8
    tune_type_vision_tower: str = field(default="frozen") # support only: frozen, full, partially-tune
    tune_vision_tower_from_layer: Optional[int] = field(default=10)
    tune_type_connector: str = field(default="frozen") # support only: frozen, full
    tune_embed_tokens: Optional[int] = field(default=False)
    
    remove_unused_columns: bool = field(default=False)
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)
    vision_tower_lr: Optional[float] = None
    pretrained_model_path: Optional[str] = field(default="home/user27/outputs/checkpoints/llava_factory/tiny-llava-phi-2-xeus-first_try-pretrain/checkpoint-1000")


def eval_model(args):

    
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    training_arguments = {}
    training_recipe = TrainingRecipeFactory("common")(training_arguments) 
    model = TinyLlavaForConditionalGeneration.from_pretrained(model_path, trust_remote_code=True)

    model.vision_tower.load_model(None)

    tokenizer = model.tokenizer
    # break
    print(model)


    # answers_file = os.path.expanduser(args.answer_file)
    # os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    # ans_file = open(answers_file, "w")
    wer = load("wer")

    data_loader = create_data_loader(args.data_path,
                                    tokenizer,
                                    num_chunks=args.num_chunks, 
                                    chunk_id=args.chunk_idx)
    # print("Tokenizer's eos token: ", tokenizer.eos_token)
    model.to(device='cuda')
    for i, batch in tqdm(enumerate(data_loader)):
        # keywords = [tokenizer.eos_token]
        # stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        input_ids = batch['input_ids'].abs().to(device='cuda', non_blocking=True)
        # print(tokenizer.batch_decode(input_ids, skip_special_tokens=True))
        
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                audios=batch['audios'].to(dtype=torch.float16, device='cuda', non_blocking=True),
                wav_lengths=batch['lengths'],
                pad_token_id=tokenizer.pad_token_id,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                # stopping_criteria=[stopping_criteria],
                use_cache=True)

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        
        # print(tokenizer.batch_decode(input_ids, skip_special_tokens=True)[0].strip())
        # print(output_ids)
        # print(input_ids)
        input = tokenizer.batch_decode(batch['labels'].abs(), skip_special_tokens=True)
        print(input)
        print("\n")
        # print()
        print(repr(outputs))
        print('\n\n')
        
        # print(wer.compute(prediction=outputs, references=input))
        if i == 10: break
        # break
        # print("Printing outputs")
        # print(outputs)
        # time.sleep(5)
        # ans_id = shortuuid.uuid()
        # ans_file.write(json.dumps({"predicted_text": outputs,
        #                            }) + "\n")
        # ans_file.flush()
    # ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--data_path", type=str, default="")
    parser.add_argument("--answer_file", type=str, default="answers.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llama")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--image_aspect_ratio", type=str, default="pad")
    args = parser.parse_args()

    eval_model(args)
