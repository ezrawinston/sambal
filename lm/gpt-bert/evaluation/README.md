# Evaluation

## BLiMP


## EWoK


## (Super)GLUE

### Tokenizer changes
So that your tokanizer automatically adds the special tokens at the right places the following lines need to be added to the tokenizer.json file:

```json
"post_processor": {
    "type": "TemplateProcessing",
    "single": [
        {
            "SpecialToken": {
                "id": "START_TOKEN_TEXT_VALUE",
                "type_id": 0
            }
        },
        {
            "Sequence": {
                "id": "A",
                "type_id": 0
            }
        }
    ],
    "pair": [
        {
            "SpecialToken": {
                "id": "START_TOKEN_TEXT_VALUE",
                "type_id": 0
            }
        },
        {
            "Sequence": {
                "id": "A",
                "type_id": 0
            }
        },
        {
            "SpecialToken": {
                "id": "SEPERATOR_TOKEN_TEXT_VALUE",
                "type_id": 0
            }
        },
        {
            "Sequence": {
                "id": "B",
                "type_id": 0
            }
        }
    ],
    "special_tokens": {
        "START_TOKEN_TEXT_VALUE": {
            "id": "START_TOKEN_TEXT_VALUE",
            "ids": [
                START_TOKEN_ID
            ],
            "tokens": [
                "START_TOKEN_TEXT_VALUE"
            ]
        },
        "SEPERATOR_TOKEN_TEXT_VALUE": {
            "id": "SEPERATOR_TOKEN_TEXT_VALUE",
            "ids": [
                SEPERATOR_TOKEN_ID
            ],
            "tokens": [
                "SEPERATOR_TOKEN_TEXT_VALUE"
            ]
        }
    }
},
```

## LAMBADA