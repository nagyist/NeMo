# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
"""Pruning example for Llama model.

Usage:
    torchrun --nproc_per_node 2 prune.py --devices 2 --pp_size 2
"""

import argparse
import os
from pathlib import Path

# isort: off
import modelopt.torch.prune as mtp

# isort: on
import torch
from megatron.core import dist_checkpointing

from nemo import lightning as nl
from nemo.collections import llm
from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer
from nemo.collections.llm.gpt.data.mock import MockDataModule
from nemo.collections.nlp.models.language_modeling.megatron.gpt_layer_modelopt_spec import (
    get_gpt_layer_modelopt_spec,
)
from nemo.lightning.ckpt_utils import ckpt_to_context_subdir
from nemo.lightning.io.pl import TrainerContext, ckpt_to_weights_subdir
from nemo.utils import logging

os.environ["TOKENIZERS_PARALLELISM"] = "true"


def main(args):
    """Main function for pruning Llama model."""
    # pylint: disable=C0115,C0116

    # Load model (with modelopt spec) and tokenizer
    tokenizer = AutoTokenizer(args.tokenizer_name_or_path)
    llm_config = llm.Llama32Config1B()
    llm_config.transformer_layer_spec = get_gpt_layer_modelopt_spec()
    model = llm.LlamaModel(llm_config, tokenizer=tokenizer)

    # Training strategy setup
    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=1,  # Pruning restricted to TP=1
        pipeline_model_parallel_size=args.pp_size,
        pipeline_dtype=torch.bfloat16,
        sequence_parallel=False,
        ckpt_load_optimizer=False,
        ckpt_parallel_save_optim=False,
        setup_optimizers=False,
        ddp="pytorch",
    )

    # Trainer setup
    trainer = nl.Trainer(
        num_nodes=args.num_nodes,
        devices=args.devices,
        max_steps=0,
        accelerator="gpu",
        strategy=strategy,
        plugins=nl.MegatronMixedPrecision(
            precision="bf16-mixed", params_dtype=torch.bfloat16, autocast_enabled=True
        ),
        # limit_val_batches=args.num_val_samples // args.gbs,
        num_sanity_val_steps=0,
    )

    # Initialize distributed environment and model
    strategy.connect(model)
    strategy.setup_environment()
    model.configure_model()
    logging.info(f"Loaded model: {model}")

    # nemo_checkpoint_path = "/home/scratch.omniml_data_1/models/nemo/llama3.1-8b-nemo2.nemo"
    # model_path = Path(nemo_checkpoint_path)
    # model = nl.io.load_context(path=ckpt_to_context_subdir(model_path), subpath="model")
    # model.config = quantizable_model_config(model.config)
    # del model.optim
    # _setup_trainer_and_restore_model(nemo_checkpoint_path, trainer, model)

    def forward_loop(model):
        data_module = MockDataModule(
            seq_length=args.seq_length,
            micro_batch_size=args.mbs,
            global_batch_size=args.gbs,
            num_val_samples=args.num_val_samples,
        )
        llm.validate(model, data_module, trainer)

    logging.info("Pruning model...")
    model, _ = mtp.prune(
        model,
        mode="mcore_gpt_minitron",
        constraints={
            "export_config": {
                "ffn_hidden_size": 512,
            },
        },
        dummy_input=None,  # Not used
        config={"forward_loop": forward_loop},
    )

    logging.info("Saving pruned model...")
    output_path = "results_pruned/"
    weight_path = ckpt_to_weights_subdir(output_path, is_saving=True)
    Path(weight_path).mkdir(parents=True, exist_ok=True)
    dist_checkpointing.save(
        model.module.sharded_state_dict(), str(ckpt_to_weights_subdir(output_path, is_saving=True))
    )
    if hasattr(model.tokenizer, "save_pretrained"):
        model.tokenizer.save_pretrained("/tmp/nemo_tokenizer")
        model.tokenizer = AutoTokenizer("/tmp/nemo_tokenizer")
    if hasattr(trainer.model, "__io__") and hasattr(trainer.model.tokenizer, "__io__"):
        trainer.model.__io__.tokenizer = trainer.model.tokenizer.__io__
    TrainerContext.from_trainer(trainer).io_dump(
        ckpt_to_context_subdir(output_path), yaml_attrs=["model"]
    )

    logging.info("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Llama Pruning Script")
    parser.add_argument(
        "--restore_path",
        type=str,
        required=False,
        default=None,
        help="Path to restore model checkpoint from",
    )
    parser.add_argument(
        "--tokenizer_name_or_path",
        type=str,
        default="meta-llama/Llama-3.2-1B",
        help="Path to restore tokenizer from",
    )
    parser.add_argument("--seq_length", type=int, default=256)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--pp_size", type=int, default=1, help="Pipeline parallel size")
    parser.add_argument("--mbs", type=int, default=8, help="Micro batch size")
    parser.add_argument("--gbs", type=int, default=32, help="Global batch size")
    parser.add_argument(
        "--num_val_samples", type=int, default=128, help="Number of validation samples"
    )

    args = parser.parse_args()
    main(args)
