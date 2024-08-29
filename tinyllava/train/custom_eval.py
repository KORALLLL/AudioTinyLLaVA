import tokenizers
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoImageProcessor
import sys
import os
sys.path.append('/home/user27/AudioTinyLLaVA')

from tinyllava.train.tinyllava_trainer import LLaVATrainer
from tinyllava.training_recipe import TrainingRecipeFactory
from tinyllava.utils import *
from tinyllava.model import *
from tinyllava.data.dataset import make_supervised_data_module


def load_settings(model_arguments, data_arguments, training_arguments):
    model_arguments.tune_type_connector = training_arguments.tune_type_connector
    model_arguments.tune_type_llm = training_arguments.tune_type_llm
    model_arguments.tune_type_vision_tower = training_arguments.tune_type_vision_tower
    model_arguments.image_aspect_ratio = data_arguments.image_aspect_ratio


def validate():
    # Load arguments
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_arguments, data_arguments, training_arguments = parser.parse_args_into_dataclasses()
    
    logger_setting(getattr(training_arguments, 'output_dir', None))
    training_recipe = TrainingRecipeFactory(training_arguments.training_recipe)(training_arguments) 
    load_settings(model_arguments, data_arguments, training_arguments)

    # Load pretrained checkpoint
    print(training_arguments.pretrained_model_path)
    model = AutoModelForCausalLM.from_pretrained(training_arguments.pretrained_model_path, trust_remote_code=True)
    config = model.config
    
    tokenizer = AutoTokenizer.from_pretrained(
        training_arguments.pretrained_model_path,
        use_fast=False,
        model_max_length=config.tokenizer_model_max_length,
        padding_side=config.tokenizer_padding_side
    )
    model.tokenizer = tokenizer
    model = training_recipe(model)
    
    model.config.use_cache = False
    model.config.image_aspect_ratio = data_arguments.image_aspect_ratio
    
    data_arguments.image_processor = AutoImageProcessor.from_pretrained(config.vision_model_name_or_path)
    data_arguments.is_multimodal = True
    
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_arguments)
    log_trainable_params(model)

    trainer = LLaVATrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_arguments,
        **data_module
    )
    
    # Evaluate the model on the validation set
    metrics = trainer.evaluate()
    
    print(f"Validation metrics: {metrics}")
    
    training_recipe.save(model, trainer)


if __name__ == "__main__":
    validate()
