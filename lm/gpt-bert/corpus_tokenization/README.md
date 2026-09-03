# Corpus pre-tokenization

In this folder you will find the corpus pre-tokenization script. This is required to work with our dataset.

## Pre-tokenization script

The pre-tokenization script allows the tokenization of a train and validation set at the same time. Here is how to use it:

```bash
python tokenize_corpus.py \
    # Default value: "../data"
    --data_folder="FOLDER_CONTAINING_THE_DATA_TO_TOKENIZE" \
    # Default value: "train_100M.jsonl"
    --train_file="NAME_OF_TRAINING_DATA_FILE" \
    # Default value: None
    --valid_file="NAME_OF_VALIDATION_DATA_FILE" \
    # Default value: the committed "../gpt-bert-babylm-small" (resolved
    # relative to this script, so it works from any working directory)
    --tokenizer_folder="PATH_TO_TOKENIZER_FOLDER" \
    # Default value: "tokenizer.json"
    --tokenizer_file="NAME_OF_TOKENIZER_FILE" \ 
    # Default value: None
    --name="ADDITIONAL_SUFFIX_FOR_TOKENIZED_FILE_NAME"
```

Output files are written next to each input as
`<input stem>[_<name>]_tokenized.bin` — e.g. `--train_file=train.jsonl
--name=10M` produces `train_10M_tokenized.bin`.