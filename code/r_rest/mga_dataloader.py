import pdb
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import torch
from transformers import DataCollatorWithPadding


@dataclass
class MGACollator(DataCollatorWithPadding):
    """
    data collector for mga task
    """

    tokenizer = None
    padding = True
    max_length = None
    pad_to_multiple_of = None
    return_tensors = "pt"

    def __call__(self, features):
        batch = dict()

        # graph ids
        batch["id"] = [feature["id"] for feature in features]
        batch["chart_type"] = [feature["chart_type"] for feature in features]

        batch["texts"] = [feature["text"] for feature in features]
        batch["images"] = [feature["image"] for feature in features]

        # image features ---
        flattened_patches = [feature["flattened_patches"] for feature in features]
        attention_mask = [feature["attention_mask"] for feature in features]

        flattened_patches = np.concatenate(flattened_patches, axis=0)
        attention_mask = np.concatenate(attention_mask, axis=0)

        batch["flattened_patches"] = flattened_patches
        batch["attention_mask"] = attention_mask

        # text features ----
        decoder_features = [
            {
                "input_ids": feature["decoder_input_ids"],
                "attention_mask": feature["decoder_attention_mask"]
            } for feature in features
        ]

        decoder_batch = self.tokenizer.pad(
            decoder_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=None,
        )

        batch["decoder_input_ids"] = decoder_batch["input_ids"]
        batch["decoder_attention_mask"] = decoder_batch["attention_mask"]

        # -100 -> ignored in loss computations
        pad_token_id = self.tokenizer.pad_token_id
        labels = []
        for ex_labels in batch["decoder_input_ids"]:
            tmp = [l if l != pad_token_id else -100 for l in ex_labels]
            labels.append(tmp)
        batch["labels"] = labels

        # casting ---
        tensor_keys = ["flattened_patches", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"]
        for key in tensor_keys:
            if key != "flattened_patches":
                batch[key] = torch.tensor(batch[key], dtype=torch.int64)
            else:
                batch[key] = torch.tensor(batch[key], dtype=torch.float32)
        # pdb.set_trace()

        return batch
