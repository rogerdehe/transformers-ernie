# Ernie Implemented by Transformers

I implement Ernie model with `transformers` for study and work.
If you want know more about `Ernie`, please refer [offical url](https://github.com/PaddlePaddle/ERNIE)

## Usage
```python
from transformers_ernie import ErnieConfig, ErnieForPreTraining, ErnieTokenizer, ErnieTokenizerFast
model = ErnieForPreTraining.from_pretrained("./ernie-1.0")
tokenizer = ErnieTokenizerFast.from_pretrained("./ernie-1.0")

texts = ["吉林省会是哪里", "哈尔滨是黑龙江的省会城市"]
args = tokenizer(texts, padding=True, return_tensors="pt")
model_output = model(**args)
print(model_output)
```

```python
from transformers_ernie import ErnieCtmConfig, ErnieCtmModel, ErnieCtmForWordtag, ErnieCtmForNptag, ErnieCtmTokenizer, ErnieCtmTokenizerFast
model = ErnieCtmForWordtag.from_pretrained("./wordtag", num_cls_label=4, num_tag=265)
tokenizer = ErnieCtmTokenizerFast.from_pretrained("./wordtag")

texts = ["吉林省会是哪里", "哈尔滨是黑龙江的省会城市"]
args = tokenizer(texts, padding=True, return_tensors="pt", return_length=True)
model_output = model(**args)
print(model_output)
```

## Convert Model for yourself

1. First, you need to install `paddlenlp` with `pip install paddlenlp`

2. Then you can run script `convert_original_paddle_checkpoint_to_pytorch.py` in `src/transformers_ernie`. 

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
This script will: 
- download model from paddle repository if model name given.
- load model into paddle
- convert paddle model to PyTorch model.

## TODO
I will convert more model next
