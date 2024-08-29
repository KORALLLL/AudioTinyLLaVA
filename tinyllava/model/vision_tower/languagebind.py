from . import register_vision_tower
from .base import VisionTower
import torch

import sys
sys.path.append("/home/user27/AudioTinyLLaVA/LanguageBind")
from languagebind import LanguageBindAudio, LanguageBindAudioTokenizer, LanguageBindAudioProcessor

@register_vision_tower('languagebind')
class Languagebind(VisionTower):
    def __init__(self, cfg):
        super().__init__(cfg)
    
    def _load_model(self, vision_tower_name, **kwargs): # Do we actually need this args? probably yes
        pretrained_ckpt = 'LanguageBind/LanguageBind_Audio_FT'  # also 'LanguageBind/LanguageBind_Audio'
        
        model = LanguageBindAudio.from_pretrained(
            pretrained_ckpt, cache_dir='/home/user27/AudioTinyLLaVA/LanguageBind/cache_dir'
        )
        tokenizer = LanguageBindAudioTokenizer.from_pretrained(
            pretrained_ckpt, cache_dir='/home/user27/AudioTinyLLaVA/LanguageBind/cache_dir'
        )
        self.audio_process = LanguageBindAudioProcessor(model.config, tokenizer)
        
        self._vision_tower = model
        print("\n\n\n\n                        Languagebind loaded\n\n\n\n")
    
    def forward(self, x, **kwargs):
        x = list(x)
        x = [audio.unsqueeze(0) for audio in x]
        # print(f"\n\n\n\ntensors befor (audio_enc)    {x}     \n\n\n\n")
        # audio_proces принимает list из pt tensors размерности (channels, audio_len) формата float
        data = self.audio_process(x, text=len(x)*["my audio"], return_tensors="pt")
        # print(f"\n\n\n\ntensors after (audio_enc)    {data}    \n\n\n\n")
        data['input_ids'], data["attention_mask"] = data['input_ids'].to("cuda"), data["attention_mask"].to("cuda")
        output = self._vision_tower(**data)
        # print(f"\n\n\n\noutput after (audio_enc)   {data}     \n\n\n\n")
        return output.image_embeds # (batch_size, 768)