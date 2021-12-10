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
from transformers.utils import logging


logging.set_verbosity_info()


def convert_paddle_checkpoint_to_pytorch(paddle_checkpoint_path, pytorch_dump_path):
    import torch
    from paddlenlp.transformers import ErnieForPretraining as PErnieForPreTraining, ErnieTokenizer as PErnieTokenizer
    from transformers_ernie import ErnieConfig, ErnieForPreTraining

    # Initialise PyTorch model
    src_model = PErnieForPreTraining.from_pretrained(paddle_checkpoint_path)
    config = {k: v for k, v in src_model.config["ernie"].config.items() if k != "self"}
    config = ErnieConfig(**config)

    tokenizer = PErnieTokenizer.from_pretrained(paddle_checkpoint_path)
    print(f"Save vocab to {pytorch_dump_path}")
    tokenizer.save_pretrained(pytorch_dump_path)

    print(f"Building PyTorch model from configuration: {config}")
    tgt_model = ErnieForPreTraining(config)

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
            if parts[-2].endswith("_proj") or parts[-2].startswith("linear") or parts[-2].startswith("dense") or parts[-2] in {"transform", "seq_relationship"}:
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
