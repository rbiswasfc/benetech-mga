import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# import torch.utils.checkpoint
from transformers import Pix2StructConfig, Pix2StructForConditionalGeneration


class MGAModel(nn.Module):
    """
    The MGA model
    """

    def __init__(self, cfg):
        print("initializing the MGA model...")

        super(MGAModel, self).__init__()
        self.cfg = cfg

        backbone_config = Pix2StructConfig.from_pretrained(cfg.model.backbone_path)
        backbone_config.text_config.max_length = cfg.model.max_length
        backbone_config.text_config.is_decoder = True

        backbone_config.text_config.pad_token_id = cfg.model.pad_token_id
        backbone_config.text_config.decoder_start_token_id = cfg.model.decoder_start_token_id
        backbone_config.text_config.bos_token_id = cfg.model.bos_token_id

        # backbone_config.decoder.max_length = cfg.model.max_length

        self.backbone = Pix2StructForConditionalGeneration.from_pretrained(
            cfg.model.backbone_path,
            config=backbone_config,
        )

        # resize model embeddings
        print("resizing model embeddings...")
        print(f"tokenizer length = {cfg.model.len_tokenizer}")
        self.backbone.decoder.resize_token_embeddings(cfg.model.len_tokenizer)

        self.loss_fn = nn.CrossEntropyLoss(
            ignore_index=-100,
            reduction="mean",
        )

        # self.backbone.encoder.gradient_checkpointing_enable()
        # self.backbone.decoder.gradient_checkpointing_enable()

    def forward(
            self,
            flattened_patches,
            attention_mask,
            labels,
    ):

        outputs = self.backbone(
            flattened_patches=flattened_patches,
            attention_mask=attention_mask,
            labels=labels,
        )

        loss_main = outputs.loss

        # logits = outputs.logits  # [:, 0]  # (bs, vocab)
        # labels_cls = labels[:, 0]  # (bs, )
        # loss = self.loss_fn(logits, labels)

        # logits_num = outputs.logits[:, 1]  # (bs, vocab)
        # labels_num = labels[:, 1]  # (bs, )
        # loss_num = self.loss_fn(logits_num, labels_num)

        loss = loss_main  # + 0.10 * loss_cls

        loss_dict = {
            "loss_main": loss_main,
            "loss_cls": loss,
        }

        return loss, loss_dict
