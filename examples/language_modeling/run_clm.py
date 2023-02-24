from typing import Optional

import sys
from dataclasses import (
    dataclass,
    field
)

import datasets
from datasets import load_dataset

import evaluate

import transformers
from transformers import (
    HfArgumentParser,
    TrainingArguments,
    AutoTokenizer
)

import accelerate
from accelerate import logging


logger = logging.get_logger(__name__)


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
    model_config: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Set the model config when training from scratch."
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
        help={
            "help": "State if you want to keep linebreaks during training and validation."
        }
    )
    max_train_samples: Optional[int] = field(
        default=None,
        help={
            "help": "Maximum number of training examples (mainly for testing)."
        }
    )
    max_validation_samples: Optional[int] = field(
        default=None,
        help={
            "help": "Maximum number of validation examples (mainly for testing)."
        }
    )
    overwrite_cache: Optional[bool] = field(
        default=False,
        help={
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


def main():
    parser = HfArgumentParser((
        TrainingArguments,
        ModelArguments,
        DataTrainingArguments
    ))
    training_args, model_args, data_args = parser.parse_args_into_dataclasses()

    accelerate.logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler('./logs/run_clm.logs')],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)





if __name__ == '__main__':
    main()