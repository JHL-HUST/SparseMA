import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lavis.common.registry import registry
from lavis.models import load_model_and_preprocess, load_preprocess
from omegaconf import OmegaConf
import os

import warnings
from copy import deepcopy

import torch
import torch.nn.functional as F
from lavis.common.registry import registry
from lavis.models.albef_models import AlbefBase
from lavis.models.albef_models.albef_outputs import (
    AlbefIntermediateOutput,
    AlbefOutputWithLogits,
)
from lavis.models.base_model import MomentumDistilationMixin
from lavis.models.med import XBertEncoder
from lavis.models.vit import VisionTransformerEncoder
from torch import nn
from transformers import BertTokenizer

# @registry.register_model("albef_classification")
class AlbefClassification(AlbefBase, MomentumDistilationMixin):
    PRETRAINED_MODEL_CONFIG_DICT = {
        "ve": "configs/models/albef_classification_ve.yaml",
    }

    def __init__(
        self,
        image_encoder,
        text_encoder,
        num_classes,
        momentum=0.995,
        alpha=0.4,
        use_distill=True,
        max_txt_len=40,
    ):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.max_txt_len = max_txt_len

        self.use_distill = use_distill

        self.visual_encoder = image_encoder
        self.text_encoder = text_encoder

        hidden_size = text_encoder.config.hidden_size

        if num_classes > 0:
            self.cls_head = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, num_classes),
            )
        else:
            warnings.warn(
                f"Found num_classes=0, initializing {type(self)} without classifier."
            )

        if self.use_distill:
            self.visual_encoder_m = deepcopy(self.visual_encoder)
            self.text_encoder_m = deepcopy(self.text_encoder)
            self.cls_head_m = deepcopy(self.cls_head)

            self.momentum = momentum
            self.alpha = alpha

            self.model_pairs = [
                [self.visual_encoder, self.visual_encoder_m],
                [self.text_encoder, self.text_encoder_m],
                [self.cls_head, self.cls_head_m],
            ]

            self.copy_params()

    def _rampup_factor(self, epoch, iters, num_iters_per_epoch):
        return min(1, (epoch * num_iters_per_epoch + iters) / num_iters_per_epoch)

    def forward(self, samples, is_train=True):
        sentences = samples["text_input"]
        sentences = self.tokenizer(
            sentences,
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(self.device)
        samples.update({"tokenized_text": sentences})

        targets = samples["label"]

        image_embeds = self.visual_encoder.forward_features(samples["image"])
        encoder_output = self.text_encoder.forward_automask(
            samples["tokenized_text"], image_embeds
        )

        prediction = self.cls_head(encoder_output.last_hidden_state[:, 0, :])

        if is_train:
            if self.use_distill:
                with torch.no_grad():
                    self._momentum_update()

                    image_embeds_m = self.visual_encoder_m(samples["image"])
                    encoder_output_m = self.text_encoder_m.forward_automask(
                        samples["tokenized_text"], image_embeds_m
                    )

                    prediction_m = self.cls_head_m(
                        encoder_output_m.last_hidden_state[:, 0, :]
                    )

                alpha = self.alpha * self._rampup_factor(
                    epoch=samples["epoch"],
                    iters=samples["iters"],
                    num_iters_per_epoch=samples["num_iters_per_epoch"],
                )

                loss = (1 - alpha) * F.cross_entropy(
                    prediction, targets
                ) - alpha * torch.sum(
                    F.log_softmax(prediction, dim=1) * F.softmax(prediction_m, dim=1),
                    dim=1,
                ).mean()
            else:
                loss = F.cross_entropy(prediction, targets)

                image_embeds_m, encoder_output_m, prediction_m = None, None, None

            # return {"loss": loss}
            return AlbefOutputWithLogits(
                loss=loss,
                intermediate_output=AlbefIntermediateOutput(
                    image_embeds=image_embeds,
                    image_embeds_m=image_embeds_m,
                    encoder_output=encoder_output,
                    encoder_output_m=encoder_output_m,
                ),
                logits=prediction,
                logits_m=prediction_m,
            )
        else:
            return {"predictions": prediction, "targets": targets}

    def predict(self, samples):
        output = self.forward(samples, is_train=False)
        return output

    @classmethod
    def from_config(cls, cfg=None):
        image_encoder = VisionTransformerEncoder.from_config(cfg)

        # text encoder + multimodal encoder
        text_encoder = XBertEncoder.from_config(cfg)

        alpha = cfg.get("alpha", 0.4)
        momentum = cfg.get("momentum", 0.995)
        use_distill = cfg.get("use_distill", True)
        num_classes = cfg.get("num_classes", -1)
        max_txt_len = cfg.get("max_txt_len", 40)

        assert num_classes > 1, "Invalid number of classes provided, found {}".format(
            num_classes
        )

        model = cls(
            image_encoder=image_encoder,
            text_encoder=text_encoder,
            use_distill=use_distill,
            alpha=alpha,
            num_classes=num_classes,
            momentum=momentum,
            max_txt_len=max_txt_len,
        )

        model.load_checkpoint_from_config(cfg)

        return model
    def forward_features(self, samples):
        sentences = samples["text_input"]
        sentences = self.tokenizer(
            sentences,
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(self.device)
        samples.update({"tokenized_text": sentences})

        image_embeds = self.visual_encoder.forward_features(samples["image"])
        encoder_output = self.text_encoder.forward_automask(
            samples["tokenized_text"], image_embeds
        )
        
        feature = self.cls_head[0](encoder_output.last_hidden_state[:, 0, :])
        return feature

def load_ALBEF(model_path,num_classes,is_distributed_model=False):
    if not '.pt' in model_path:
        new_model_path = model_path+'_n.pt'
        if not os.path.isfile(new_model_path):
            checkpoint = torch.load(model_path, map_location='cpu')
            checkpoint = {k.replace('module.',''):v for k,v in checkpoint.items()}
            torch.save(checkpoint,new_model_path)
        model_path = new_model_path
    model_cfg = dict(
        arch= 'albef_classification', 
        finetuned = model_path,
        load_finetuned= True, 
        num_classes= num_classes,
        use_distill= False,
        momentum= 0.995,
        alpha= 0.4,
        vit_type= 'base',
        vit_grad_ckpt= False,
        vit_ckpt_layer= 0,
        vit_layer_norm_epsilon= 1e-06,
        image_size= 384, 
        med_config_path= 'configs/models/med_config_albef.json',
    )
    model = AlbefClassification.from_config(model_cfg)
    model.eval()
    return model


if __name__ == '__main__':
    
    model_path = 'path_to_models'
    model = load_ALBEF(model_path,3)

