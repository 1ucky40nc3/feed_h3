from typing import Optional

import os
import sys
import json
import math
import random
import logging
from dataclasses import (
    dataclass,
    field
)
from itertools import chain

import flatten_dict
from tqdm import tqdm

import torch

import datasets
from datasets import (
    load_dataset,
    DatasetDict,
    get_dataset_infos,
    get_dataset_config_names
)

import evaluate

import transformers
from transformers import (
    HfArgumentParser,
    TrainingArguments,
    AutoConfig,
    AutoTokenizer,
    default_data_collator,
    get_scheduler
)

import accelerate
from accelerate import Accelerator
from accelerate.logging import get_logger


from feed_h3 import (
    SSMSeqConfig,
    SSMLMHeadModel
)


logger = get_logger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Name or path to a pretrained model."
                " Currently only paths can be provided."
                " Note: Don't provided this prohibited when training from scratch!"
            )
        }
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the model config (when training from scratch)."
                " Provide the config as comma seperated `key=value` pairs."
                " For example: n_embd=10,resid_pdrop=0.2,..."
            )
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Name of a pretrained tokenizer."
        }
    )

    def __post_init__(self):
        if self.model_config is not None and self.model_name_or_path is not None:
            raise ValueError("The parameters `--model_name_or_path` and `model_config` can't be combined!")


@dataclass
class DataTrainingArguments:
    dataset_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Name of a dataset (from the `datasets` lib)."
        }
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Name of a dataset config (from the `datasets` lib)."
        }
    )
    train_file: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Path to a training file (csv/txt/json format).",
                " If `--dataset_name` is provided this will be ignored."
            )
        }
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Path to a validation file (csv/txt/json format).",
                " If `--dataset_name` is provided this will be ignored."
            )
        }
    )
    valid_split_percentage: Optional[int] = field(
        default=None,
        metadata={
            "help": "Size of the validation split if no `--validation_file` is provided."
        }
    )
    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "Lenght of the inputs during training and validation."
        }
    )
    keep_linebreaks: Optional[bool] = field(
        default=True,
        metadata={
            "help": "State if you want to keep linebreaks during training and validation."
        }
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "Maximum number of training examples (mainly for testing)."
        }
    )
    max_validation_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "Maximum number of validation examples (mainly for testing)."
        }
    )
    overwrite_cache: Optional[bool] = field(
        default=False,
        metadata={
            "help": "State if you want to overwrite cached training and validation datasets."
        }
    )
    processing_num_workers: Optional[bool] = field(
        default=None,
        metadata={
            "help": (
                "Number of processed used to do processing."
                " If "
            )
        }
    )


def get_dataloaders(accelerator, tokenizer, training_args, model_args, data_args):
    # Prepare the raw datasets
    if data_args.dataset_name is not None:
        dataset_config_name = data_args.dataset_config_name
        if dataset_config_name is None:
            dataset_config_names = get_dataset_config_names(data_args.dataset_name)
            assert len(dataset_config_names) == 1, f'The dataset has multiple `configs`! Choose one of: {dataset_config_names}'
            dataset_config_name = dataset_config_names[0]
        infos = get_dataset_infos(data_args.dataset_name)[dataset_config_name]

        if 'validation' not in infos.splits.keys():
            raw_datasets = DatasetDict()
            raw_datasets['validation'] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f'train[:{data_args.validation_split_percentage}%]',
            )
            raw_datasets['train'] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f'train[{data_args.validation_split_percentage}%:]',
            )
        else:
            raw_datasets = load_dataset(data_args.dataset_name, data_args.dataset_config_name)
    else:
        data_files = {}
        dataset_args = {}
        if data_args.train_file is not None:
            data_files['train'] = data_args.train_file
        if data_args.validation_file is not None:
            data_files['validation'] = data_args.validation_file

        extension = data_args.train_file.split('.')[-1]
        if extension == 'txt':
            extension = 'text'
            dataset_args['keep_linebreaks'] = not data_args.no_keep_linebreaks
        
        if 'validation' not in data_files.keys():
            raw_datasets = DatasetDict()
            raw_datasets['validation'] = load_dataset(
                extension,
                data_files=data_files,
                split=f'train[:{data_args.validation_split_percentage}%]',
                **dataset_args,
            )
            raw_datasets['train'] = load_dataset(
                extension,
                data_files=data_files,
                split=f'train[{data_args.validation_split_percentage}%:]',
                **dataset_args,
            )
        else:
            raw_datasets = load_dataset(extension, data_files=data_files, **dataset_args)

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    column_names = raw_datasets['train'].column_names
    text_column_name = 'text' if 'text' in column_names else column_names[0]

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])

    with accelerator.main_process_first():
        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc='Running tokenizer on dataset',
        )

    if data_args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            print(
                'The chosen tokenizer supports a `model_max_length` that is longer than the default `block_size` value'
                ' of 1024. If you would like to use a longer `block_size` up to `tokenizer.model_max_length` you can'
                ' override this default with `--block_size xxx`.'
            )
        block_size = 1024
    else:
        if data_args.block_size > tokenizer.model_max_length:
            print(
                f'The block_size passed ({data_args.block_size}) is larger than the maximum length for the model'
                f'({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}.'
            )
        block_size = min(data_args.block_size, tokenizer.model_max_length)

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result['labels'] = result['input_ids'].copy()
        return result

    # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
    # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
    # to preprocess.
    #
    # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
    # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map

    with accelerator.main_process_first():
        lm_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
            desc=f'Grouping texts in chunks of {block_size}',
        )

    train_dataset = lm_datasets['train']
    eval_dataset = lm_datasets['validation']

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f'Sample {index} of the training set: {train_dataset[index]}.')

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, 
        shuffle=True, 
        collate_fn=default_data_collator, 
        batch_size=training_args.per_device_train_batch_size
    )
    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset, 
        collate_fn=default_data_collator, 
        batch_size=training_args.per_device_eval_batch_size
    )

    return train_dataset, eval_dataset, train_dataloader, eval_dataloader


def training_fn(training_args, model_args, data_args):
    accelerator_kwargs = {
        'mixed_precision': None if training_args.fp16 is False else 'fp16',
        'gradient_accumulation_steps': training_args.gradient_accumulation_steps,
        'log_with': training_args.report_to,
        'project_dir': training_args.logging_dir,
    }
    accelerator = Accelerator(**accelerator_kwargs)
    accelerator.wait_for_everyone()        

    # Initialize the config
    if model_args.config_name:
        config = SSMSeqConfig.from_pretrained(model_args.config_name)
    elif model_args.model_name_or_path:
        config = SSMSeqConfig.from_pretrained(model_args.model_name_or_path)
    else:
        logger.info('Initializing new config from scratch!')
        config = SSMSeqConfig()
        if model_args.config_overrides is not None:
            logger.info(f'Overriding config with: {model_args.config_overrides}')
            config.update_from_string(model_args.config_overrides)

    # Initialize the tokenizer
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name, 
            use_fast=model_args.use_fast_tokenizer
        )
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path, 
            use_fast=model_args.use_fast_tokenizer
        )
    else:
        raise ValueError(
            'You are instantiating a new tokenizer from scratch. This is not supported by this script.'
            'You can do it from another script, save it, and load it from here, using --tokenizer_name.'
        )

    # Initialize the model
    if model_args.model_name_or_path:
        model = SSMLMHeadModel.from_pretrained(
            model_args.model_name_or_path,
            config=config
        )
    else:
        logger.info('Training model from scratch!')
        model = SSMLMHeadModel.from_config(config)

    no_decay = ['bias', 'layer_norm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [
                p 
                for n, p in model.named_parameters() 
                if not any(nd in n for nd in no_decay)
            ],
            'weight_decay': training_args.weight_decay,
        },
        {
            'params': [
                p 
                for n, p in model.named_parameters() 
                if any(nd in n for nd in no_decay)
            ],
            'weight_decay': 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters, 
        lr=training_args.learning_rate
    )

    (
        train_dataset, 
        eval_dataset, 
        train_dataloader, 
        eval_dataloader
    ) = get_dataloaders(accelerator, training_args, model_args, data_args)

    overrode_max_train_steps  = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / training_args.gradient_accumulation_steps)
    if training_args.max_steps in (None, -1):
        training_args.max_steps = training_args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps  = True

    lr_scheduler = get_scheduler(
        name=training_args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=training_args.warmup_steps * training_args.gradient_accumulation_steps,
        num_training_steps=training_args.max_steps * training_args.gradient_accumulation_steps,
    )

    (
        model, 
        optimizer, 
        train_dataloader, 
        eval_dataloader, 
        lr_scheduler
    ) = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / training_args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        training_args.max_steps = int(training_args.num_train_epochs * num_update_steps_per_epoch)
    # Afterwards we recalculate our number of training epochs
    training_args.num_train_epochs = math.ceil(training_args.max_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = training_args.save_steps
    if checkpointing_steps is not None and isinstance(checkpointing_steps, str) and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    total_batch_size = training_args.per_device_train_batch_size * accelerator.num_processes * training_args.gradient_accumulation_steps

    # Initialize the loggers
    experiment_config = {**vars(training_args), **vars(data_args), **vars(model_args), **vars(model.config)}
    # TensorBoard cannot log Enums, need the raw value
    experiment_config['lr_scheduler_type'] = experiment_config['lr_scheduler_type'].value
    experiment_config = {k: v for k, v in experiment_config.items() if not k.startswith('_')}
    experiment_config = flatten_dict.flatten(experiment_config, reducer='path')
    cast = lambda a: a if isinstance(a, (int, float, str, bool, torch.Tensor)) else str(a)
    experiment_config = {k: cast(v) for k, v in experiment_config.items() if not k.startswith('_')}
    accelerator.init_trackers('feed_h3_clm', experiment_config)

    # Train!
    print('***** Running training *****')
    print(f'  Num examples = {len(train_dataset)}')
    print(f'  Num Epochs = {training_args.num_train_epochs}')
    print(f'  Instantaneous batch size per device = {training_args.per_device_train_batch_size}')
    print(f'  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}')
    print(f'  Gradient Accumulation steps = {training_args.gradient_accumulation_steps}')
    print(f'  Total optimization steps = {training_args.max_steps}')
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(training_args.max_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0

    for epoch in range(starting_epoch, training_args.num_train_epochs):
        model.train()
        total_loss = 0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                output = model(**batch)
                loss = output.loss
                total_loss += loss.detach().float()
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1
            
            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0:
                    output_dir = f'step_{completed_steps }'
                    if training_args.output_dir is not None:
                        output_dir = os.path.join(training_args.output_dir, output_dir)
                    accelerator.save_state(output_dir)
            if completed_steps >= training_args.max_steps:
                break

        model.eval()
        losses = []
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                output = model(**batch)
            loss = output.loss
            losses.append(
                accelerator.gather_for_metrics(
                    loss.repeat(training_args.per_device_eval_batch_size)
                )
            )
        losses = torch.cat(losses)
        try:
            eval_loss = torch.mean(losses)
            perplexity = math.exp(eval_loss)
        except OverflowError:
            perplexity = float('inf')

        print(f'epoch {epoch}: perplexity: {perplexity} eval_loss: {eval_loss}')

        accelerator.log(
            {
                'perplexity': perplexity,
                'eval_loss': eval_loss,
                'train_loss': total_loss.item() / len(train_dataloader),
                'epoch': epoch,
                'step': completed_steps,
            },
            step=completed_steps,
        )

    if training_args.output_dir is not None:
        output_dir = f'step_{completed_steps}'
        if training_args.output_dir is not None:
            output_dir = os.path.join(training_args.output_dir, output_dir)
        accelerator.save_state(output_dir)
        accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            tokenizer.save_pretrained(training_args.output_dir)

            with open(os.path.join(training_args.output_dir, 'all_results.json'), 'w') as f:
                json.dump({'perplexity': perplexity}, f)

def main():
    parser = HfArgumentParser((
        TrainingArguments,
        ModelArguments,
        DataTrainingArguments
    ))
    training_args, model_args, data_args = parser.parse_args_into_dataclasses()

    training_fn(training_args, model_args, data_args)





if __name__ == '__main__':
    main()