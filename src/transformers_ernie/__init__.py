# -*- coding: utf-8 -*-
from transformers.file_utils import _LazyModule, is_flax_available, is_tf_available, is_tokenizers_available, is_torch_available

from .ernie import ERNIE_PRETRAINED_CONFIG_ARCHIVE_MAP, ErnieConfig, ErnieOnnxConfig
from .ernie import ErnieTokenizer
from .ernie_ctm import ERNIE_CTM_PRETRAINED_CONFIG_ARCHIVE_MAP, ErnieCtmConfig, ErnieCtmOnnxConfig
from .ernie_ctm import ErnieCtmTokenizer

if is_tokenizers_available():
    from .ernie import ErnieTokenizerFast
    from .ernie_ctm import ErnieCtmTokenizerFast

if is_torch_available():
    from .ernie import (
        ERNIE_PRETRAINED_MODEL_ARCHIVE_LIST,
        ErnieForMaskedLM,
        ErnieForMultipleChoice,
        ErnieForNextSentencePrediction,
        ErnieForPreTraining,
        ErnieForQuestionAnswering,
        ErnieForSequenceClassification,
        ErnieForTokenClassification,
        ErnieLayer,
        ErnieLMHeadModel,
        ErnieModel,
        ErniePreTrainedModel,
    )

    from .ernie_ctm import (
        ERNIE_CTM_PRETRAINED_MODEL_ARCHIVE_LIST,
        ErnieCtmForTokenClassification,
        ErnieCtmModel,
        ErnieCtmPreTrainedModel,
        ErnieCtmForWordtag,
        ErnieCtmForNptag,
    )
