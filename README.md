# Ernie Implemented by Transformers

I implement Ernie model with `transformers` for study and work.

## Usage
```python
from transformers_ernie.ernie import ErnieConfig, ErnieForPreTraining, ErnieTokenizer, ErnieTokenizerFast
model = ErnieForPreTraining.from_pretrained("./ernie-1.0")
tokenizer = ErnieTokenizerFast.from_pretrained("./ernie-1.0")

texts = ["吉林省会是哪里", "哈尔滨是黑龙江的省会城市"]
args = tokenizer(texts, padding=True, return_tensors="pt")
model_output = model(**args)
print(model_output)
```

```python`
from ernie_ctm import ErnieCtmConfig, ErnieCtmModel, ErnieCtmForWordtag, ErnieCtmForNptag, ErnieCtmTokenizer, ErnieCtmTokenizerFast
model = ErnieCtmForWordtag.from_pretrained("./wordtag", num_cls_label=4, num_tag=265)
tokenizer = ErnieTokenizerFast.from_pretrained("./wordtag")

texts = ["吉林省会是哪里", "哈尔滨是黑龙江的省会城市"]
args = tokenizer(texts, padding=True, return_tensors="pt", return_length=True)
model_output = model(**args)
print(model_output)
```

## Convert Model for yourself

First, you need to install `paddlenlp` with `pip install paddlenlp`

Then you can run script `convert_original_paddle_checkpoint_to_pytorch.py` in `src/transformers_ernie`. This script will download model then convert to pytorch model.

- ernie-1.0
```Shell
python src/transformers_ernie/convert_original_paddle_checkpoint_to_pytorch.py --paddle_checkpoint_path ernie-1.0 --pytorch_dump_path ./ernie-1.0
```

- ernie-ctm
```Shell
python src/transformers_ernie/convert_original_paddle_checkpoint_to_pytorch.py --paddle_checkpoint_path ernie-ctm --pytorch_dump_path ./ernie-ctm
```

- wordtag
```Shell
python src/transformers_ernie/convert_original_paddle_checkpoint_to_pytorch.py --paddle_checkpoint_path wordtag --pytorch_dump_path ./wordtag
```

