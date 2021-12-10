# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Fast Tokenization classes for Ernie."""

import json
from typing import List, Optional, Tuple, Dict, Any

from tokenizers import Tokenizer, decoders, normalizers, pre_tokenizers, processors, Encoding as EncodingFast
from tokenizers.models import WordPiece
from transformers import PreTrainedTokenizerFast
from transformers.convert_slow_tokenizer import Converter
from transformers.utils import logging
from .tokenization_ernie_ctm import ErnieCtmTokenizer


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt", "tokenizer_file": "tokenizer.json"}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        # "bert-base-uncased": "https://huggingface.co/bert-base-uncased/resolve/main/vocab.txt",
    },
    "tokenizer_file": {
        # "bert-base-uncased": "https://huggingface.co/bert-base-uncased/resolve/main/tokenizer.json",
    },
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "ernie-ctm": 128,
}

PRETRAINED_INIT_CONFIGURATION = {
    # "bert-base-uncased": {"do_lower_case": True},
}


class ErnieCtmConverter(Converter):
    def converted(self) -> Tokenizer:
        vocab = self.original_tokenizer.vocab
        tokenizer = Tokenizer(WordPiece(vocab, unk_token=str(self.original_tokenizer.unk_token)))

        tokenize_chinese_chars = False
        strip_accents = False
        do_lower_case = False
        if hasattr(self.original_tokenizer, "basic_tokenizer"):
            tokenize_chinese_chars = self.original_tokenizer.basic_tokenizer.tokenize_chinese_chars
            strip_accents = self.original_tokenizer.basic_tokenizer.strip_accents
            do_lower_case = self.original_tokenizer.basic_tokenizer.do_lower_case

        tokenizer.normalizer = normalizers.BertNormalizer(
            clean_text=True,
            handle_chinese_chars=tokenize_chinese_chars,
            strip_accents=strip_accents,
            lowercase=do_lower_case,
        )
        tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()

        cls = [f"{self.original_tokenizer.cls_token_template.format(sid)}" for sid in range(self.original_tokenizer.cls_num)]
        sep = str(self.original_tokenizer.sep_token)
        cls_token_id = self.original_tokenizer.convert_tokens_to_ids(cls)
        sep_token_id = self.original_tokenizer.sep_token_id

        tokenizer.post_processor = processors.TemplateProcessing(
            single=" ".join([f"{v}:0" for v in cls]) + f" $A:0 {sep}:0",
            pair=" ".join([f"{v}:0" for v in cls]) + f" $A:0 {sep}:0 $B:1 {sep}:1",
            special_tokens=list(zip(cls, cls_token_id)) + [(sep, sep_token_id),],
        )
        tokenizer.decoder = decoders.WordPiece(prefix="##")

        return tokenizer


class ErnieCtmTokenizerFast(PreTrainedTokenizerFast):
    r"""
    Construct a "fast" ERNIE tokenizer (backed by HuggingFace's `tokenizers` library). Based on WordPiece.

    This tokenizer inherits from :class:`~transformers.PreTrainedTokenizerFast` which contains most of the main
    methods. Users should refer to this superclass for more information regarding those methods.

    Args:
        vocab_file (:obj:`str`):
            File containing the vocabulary.
        do_lower_case (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not to lowercase the input when tokenizing.
        unk_token (:obj:`str`, `optional`, defaults to :obj:`"[UNK]"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        sep_token (:obj:`str`, `optional`, defaults to :obj:`"[SEP]"`):
            The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
            sequence classification or for a text and a question for question answering. It is also used as the last
            token of a sequence built with special tokens.
        pad_token (:obj:`str`, `optional`, defaults to :obj:`"[PAD]"`):
            The token used for padding, for example when batching sequences of different lengths.
        cls_token (:obj:`str`, `optional`, defaults to :obj:`"[CLS]"`):
            The classifier token which is used when doing sequence classification (classification of the whole sequence
            instead of per-token classification). It is the first token of the sequence when built with special tokens.
        mask_token (:obj:`str`, `optional`, defaults to :obj:`"[MASK]"`):
            The token used for masking values. This is the token used when training this model with masked language
            modeling. This is the token which the model will try to predict.
        clean_text (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not to clean the text before tokenization by removing any control characters and replacing all
            whitespaces by the classic one.
        tokenize_chinese_chars (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not to tokenize Chinese characters. This should likely be deactivated for Japanese (see `this
            issue <https://github.com/huggingface/transformers/issues/328>`__).
        strip_accents: (:obj:`bool`, `optional`):
            Whether or not to strip all accents. If this option is not specified, then it will be determined by the
            value for :obj:`lowercase` (as in the original ERNIE).
        wordpieces_prefix: (:obj:`str`, `optional`, defaults to :obj:`"##"`):
            The prefix for subwords.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    slow_tokenizer_class = ErnieCtmTokenizer

    def __init__(self,
                 vocab_file,
                 do_lower_case=True,
                 do_basic_tokenize=True,
                 unk_token="[UNK]",
                 sep_token="[SEP]",
                 pad_token="[PAD]",
                 cls_token_template="[CLS{}]",
                 cls_num=1,
                 mask_token="[MASK]",
                 **kwargs):

        slow_tokenizer = self.slow_tokenizer_class(vocab_file=vocab_file,
                                                   do_lower_case=do_lower_case,
                                                   do_basic_tokenize=do_basic_tokenize,
                                                   unk_token=unk_token,
                                                   sep_token=sep_token,
                                                   pad_token=pad_token,
                                                   cls_token_template=cls_token_template,
                                                   cls_num=cls_num,
                                                   mask_token=mask_token,
                                                   **kwargs)
        fast_tokenizer = ErnieCtmConverter(slow_tokenizer).converted()

        super().__init__(vocab_file=vocab_file,
                         do_lower_case=do_lower_case,
                         do_basic_tokenize=do_basic_tokenize,
                         unk_token=unk_token,
                         sep_token=sep_token,
                         pad_token=pad_token,
                         mask_token=mask_token,
                         tokenizer_object=fast_tokenizer,
                         **kwargs)
        self.cls_token_template = cls_token_template
        self.cls_num = cls_num

    def _convert_encoding(
        self,
        encoding: EncodingFast,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
    ) -> Tuple[Dict[str, Any], List[EncodingFast]]:
        encoding_dict, encodings = super()._convert_encoding(encoding=encoding,
                                                             return_token_type_ids=return_token_type_ids,
                                                             return_attention_mask=return_attention_mask,
                                                             return_overflowing_tokens=return_overflowing_tokens,
                                                             return_special_tokens_mask=return_special_tokens_mask,
                                                             return_offsets_mapping=return_offsets_mapping,
                                                             return_length=return_length,
                                                             verbose=verbose)
        encoding_dict["attention_mask"] = [[0 if m == 1 else -1e9 for m in v] for v in encoding_dict["attention_mask"]]
        if return_length:
            encoding_dict["length"] = [len([tid for tid in v if tid != self.pad_token_id]) for v in encoding_dict["input_ids"]]
        return encoding_dict, encodings
