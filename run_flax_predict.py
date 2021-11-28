import os
import sys
from dataclasses import dataclass, field

import jax

import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers.hf_argparser import HfArgumentParser
from flax.training.common_utils import shard
from modeling_flax import FlaxBertForTokenClassification


@dataclass
class InferenceArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to predict from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    test_file: str = field(
        default=None,
        metadata={"help": "An input test data file to predict on (a csv or JSON file)."},
    )
    output_file: str = field(
        default=None,
        metadata={"help": "An output file to save predictions to."},
    )
    max_seq_length: int = field(
        default=None,
        metadata={
            "help": "The maximum total input sequence length after tokenization. If set, sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    per_device_eval_batch_size: int = field(
        default=8, metadata={"help": "Batch size per GPU/TPU core/CPU for evaluation."}
    )

    def __post_init__(self):
        if self.test_file is not None:
            extension = self.test_file.split(".")[-1]
            assert extension in ["csv", "json"], "`test_file` should be a csv or a json file."


def main():
    parser = HfArgumentParser(InferenceArguments)
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        inference_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        inference_args = parser.parse_args_into_dataclasses()[0]

    # Loading the dataset from local csv or json file.
    data_files = {"test": inference_args.test_file}
    extenstion = data_files["test"].split(".")[-1]
    raw_data = load_dataset(extenstion, data_files=data_files)

    column_names = raw_data["test"].column_names
    text_column_name = column_names[0]

    # Load pretrained model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(inference_args.model_name_or_path)
    model = FlaxBertForTokenClassification.from_pretrained(inference_args.model_name_or_path)
    config = model.config

    def preprocess_dataset(dataset):
        """
        Preprocess a given dataset.
        """
        tokenized_inputs = tokenizer(dataset["tokens"], is_split_into_words=True)

        pad = lambda x: x + [0] * (128 - len(x))
        tagged_positions = []
        for i in range(len(dataset["tokens"])):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            tagged_positions_for_word = []
            for current_input_word_idx, word_idx in enumerate(word_ids):
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    pass
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    tagged_positions_for_word.append(current_input_word_idx)

                previous_word_idx = word_idx
            tagged_positions_for_word = pad(tagged_positions_for_word)
            tagged_positions.append(tagged_positions_for_word)

        tokenized_inputs = tokenizer.pad(
            tokenized_inputs, padding="max_length", max_length=inference_args.max_seq_length
        )
        tokenized_inputs["labeled_positions"] = tagged_positions

        return tokenized_inputs

    # Preprocess the dataset
    tokenized_dataset = raw_data.map(
        preprocess_dataset,
        batched=True,
        remove_columns=raw_data["test"].column_names,
        desc="Running tokenizer on dataset",
    )
    inference_dataset = tokenized_dataset["test"]

    p_model = jax.pmap(model)

    def eval_data_collator(dataset, batch_size):
        """Returns batches of size `batch_size` from `eval dataset`, sharded over all local devices."""
        for i in range(len(dataset) // batch_size):
            batch = dataset[i * batch_size : (i + 1) * batch_size]
            batch = {k: np.array(v) for k, v in batch.items()}
            batch = shard(batch)

            yield batch

    batch_size = inference_args.per_device_eval_batch_size

    predictions = []
    for batch in tqdm(
        eval_data_collator(inference_dataset, batch_size),
        total=len(inference_dataset) // batch_size,
        desc="Running inference",
    ):
        outputs = p_model(**batch)
        logits = outputs.logits.squeeze().argmax(-1)
        predictions.extend(logits.tolist())

    # evaluate also on leftover examples (not divisible by batch_size)
    num_leftover_samples = len(inference_dataset) % batch_size
    # make sure leftover batch is evaluated on one device
    if num_leftover_samples > 0 and jax.process_index() == 0:
        batch = inference_dataset[-num_leftover_samples:]
        batch = {k: np.array(v) for k, v in batch.items()}
        logits = outputs.logits.argmax(-1)
        predictions.extend(logits.tolist())

    # write predictions to file
    with open(inference_args.output_file, "w") as f:
        for pred, data in tqdm(zip(predictions, raw_data["test"]), total=len(raw_data["test"]), desc="Formatting results"):
            preds = [config.id2label[logit] for logit in pred[: len(data[text_column_name])]]
            tokens = data[text_column_name]
            assert len(tokens) == len(preds)
            for token, pred in zip(tokens, preds):
                f.write(f"{pred}\n")
            f.write("\n")


if __name__ == "__main__":
    main()
