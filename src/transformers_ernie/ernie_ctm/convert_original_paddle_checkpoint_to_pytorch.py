# coding=utf-8
# Copyright 2018 The HuggingFace Inc. team.
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
"""Convert BERT checkpoint."""


import argparse
import os
import hashlib
from transformers.utils import logging


logging.set_verbosity_info()


def load_labels(tag_path):
    tags_to_idx = {}
    i = 0
    with open(tag_path, encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            tags_to_idx[line] = i
            i += 1
    idx_to_tags = dict(zip(*(tags_to_idx.values(), tags_to_idx.keys())))
    return tags_to_idx, idx_to_tags


def download_extra_files(save_dir):
    from paddlenlp.taskflow.utils import download_file
    URLS = {
        "TermTree.V1.0": [
            "https://kg-concept.bj.bcebos.com/TermTree/TermTree.V1.0.tar.gz",
            "3514221be5017b3b4349daa6435f7b5e"
        ],
        "termtree_type.csv": [
            "https://paddlenlp.bj.bcebos.com/models/transformers/ernie_ctm/termtree_type.csv",
            "062cb9ac24f4135bf836e2a2fc5a1209"
        ],
        "termtree_tags_pos.txt": [
            "https://paddlenlp.bj.bcebos.com/models/transformers/ernie_ctm/termtree_tags_pos.txt",
            "87db06ae6ca42565157045ab3e9a996f"
        ],
    }

    paths = list()
    for save_name, (url, md5) in URLS.items():
        path = download_file(save_dir, save_name, url, md5)
        paths.append(path)

    return paths


def convert_paddle_checkpoint_to_pytorch(paddle_checkpoint_path, pytorch_dump_path):
    import torch
    from paddlenlp.transformers import ErnieCtmModel as PErnieCtmModel, ErnieCtmWordtagModel as PErnieCtmForWordtag, ErnieCtmTokenizer as PErnieCtmTokenizer
    from transformers_ernie.ernie_ctm import ErnieCtmConfig, ErnieCtmModel, ErnieCtmForWordtag, ErnieCtmForNptag


    # Initialise PyTorch model
    model_type = "ernie-ctm"
    if "wordtag" in paddle_checkpoint_path:
        model_type = "wordtag"
    src_cls_map = {
        "wordtag": PErnieCtmForWordtag,
        "ernie-ctm": PErnieCtmModel,
    }
    tgt_cls_map = {
        "wordtag": ErnieCtmForWordtag,
        "ernie-ctm": ErnieCtmModel,
    }
    src_cls = src_cls_map[model_type]
    tgt_cls = tgt_cls_map[model_type]

    extra_args = {}
    if model_type == "wordtag":
        term_schema_path, term_data_path, tag_path = download_extra_files(pytorch_dump_path)
        tag2idx, idx2tag = load_labels(tag_path)

        extra_args["num_cls_label"] = 4
        extra_args["num_tag"] = len(tag2idx)

    src_model = src_cls.from_pretrained(paddle_checkpoint_path, **extra_args)
    config = src_model.config if "ernie_ctm" not in src_model.config else src_model.config["ernie_ctm"].config
    config = {k: v for k, v in config.items() if k != "self"}
    config = ErnieCtmConfig(**config)

    tokenizer = PErnieCtmTokenizer.from_pretrained(paddle_checkpoint_path)
    print(f"Save vocab to {pytorch_dump_path}")
    tokenizer.save_pretrained(pytorch_dump_path)

    print(f"Building PyTorch model from configuration: {config}")
    tgt_model = tgt_cls(config, **extra_args)

    key_map = {
        "transform": "transform.dense",
        "predictions.layer_norm": "predictions.transform.LayerNorm",
        "decoder_weight": "decoder.weight",
        "decoder_bias": "decoder.bias",
        "encoder.layers": "encoder.layer",
        "self_attn.q_proj": "attention.self.query",
        "self_attn.k_proj": "attention.self.key",
        "self_attn.v_proj": "attention.self.value",
        "self_attn.dropout": "attention.self.dropout",
        "self_attn.out_proj": "attention.output.dense",
        "norm1": "attention.output.LayerNorm",
        "linear1": "intermediate.dense",
        "activation": "intermediate.intermediate_act_fn",
        "linear2": "output.dense",
        "norm2": "output.LayerNorm",
        "embeddings.layer_norm": "embeddings.LayerNorm",
    }

    def do_tranpose(name):
        parts = name.split(".")
        if parts[-1] == "weight":
            if parts[-2] in {"transform", "seq_relationship", "embedding_hidden_mapping_in", "feature_fuse", "feature_output", "tag_classifier", "sent_classifier"}:
                return True
            if parts[-2].endswith("_proj") or parts[-2].startswith("linear") or parts[-2].startswith("dense"):
                return True
        return False

    src_state_dict = src_model.state_dict()
    tgt_state_dict = tgt_model.state_dict()
    for name, src_data in src_state_dict.items():
        tgt_name = name
        for k, v in key_map.items():
            tgt_name = tgt_name.replace(k, v)
        src_data = src_data.numpy()
        if do_tranpose(name):
            src_data = src_data.T

        tgt_data = tgt_state_dict[tgt_name].numpy()
        assert src_data.shape == tgt_data.shape

        tgt_state_dict[tgt_name] = torch.from_numpy(src_data)

    tgt_model.load_state_dict(tgt_state_dict)

    # Save pytorch-model
    print(f"Save PyTorch model to {pytorch_dump_path}")
    tgt_model.save_pretrained(pytorch_dump_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--paddle_checkpoint_path", default=None, type=str, required=True, help="Path to the TensorFlow checkpoint path."
    )
    parser.add_argument(
        "--pytorch_dump_path", default=None, type=str, required=True, help="Path to the output PyTorch model."
    )
    args = parser.parse_args()
    convert_paddle_checkpoint_to_pytorch(args.paddle_checkpoint_path, args.pytorch_dump_path)
