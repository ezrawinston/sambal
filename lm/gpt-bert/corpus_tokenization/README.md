# Corpus pre-tokenization

In this folder you will find the corpus pre-tokenization script. This is required to work with our dataset.

## Pre-tokenization script

The pre-tokenization script allows the tokenization of a train and validation set at the same time. Here is how to use it:

```bash
python tokenize_corpus.py \
    # Default value: "../data"
    --data_folder="FOLDER_CONTAINING_THE_DATA_TO_TOKENIZE" \
    # Default value: "train_100M.jsonl"
    --train_file="PATH_TO_VALIDATION_DATA" \
    # Default value: None
    --valid_file="NAME_OF_VALIDATION_DATA" \
    # Default value: "../tokenizers"
    --tokenizer_folder="PATH_TO_TOKENIZER_FOLDER" \
    # Default value: "tokenizer_100M.json"
    --tokenizer_file="NAME_OF_TOKENIZER_FILE" \ 
    # Default value: "100M"
    --name="ADDITIONAL_SUFFIX_FOR_TOKENIZED_FILE_NAME"
```