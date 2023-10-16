import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lavis.common.registry import registry
import os
from lavis.models import load_model_and_preprocess, load_preprocess
from lavis.models.clip_models.model import CLIP,load_openai_model
from omegaconf import OmegaConf


class CLIPFusion(nn.Module):
    def __init__(self, model_path, num_classes=2,output_dim=512):
        super(CLIPFusion, self).__init__()
        self.clip_model = load_openai_model(model_path, device=torch.device('cpu'), jit=False)

        self.mlp1 = nn.Sequential(
            nn.BatchNorm1d(2 * output_dim),
            nn.Linear(2 * output_dim, 128),
            nn.ReLU(True),
            nn.Dropout(),
        )
        self.mlp2 = nn.Linear(in_features=128, out_features=num_classes, bias=True)
    
    def forward(self, sample):
        clip_features = self.clip_model.extract_features(sample)
        image_feature = clip_features.image_embeds_proj
        text_feature = clip_features.text_embeds_proj
        # print(image_feature.shape)
        # print(text_feature.shape)
        feature = torch.cat([image_feature, text_feature], dim = -1)
        # print(feature.shape)
        feature = self.mlp1(feature)
        output = self.mlp2(feature)
        return output

    def forward_features(self, sample):
        clip_features = self.clip_model.extract_features(sample)
        image_feature = clip_features.image_embeds_proj
        text_feature = clip_features.text_embeds_proj
        feature = torch.cat([image_feature, text_feature], dim = -1)
        feature = self.mlp1(feature)
        return feature
    
    # def forward_features(self, image, input_ids):
    #     image_feature = self.image_encoder(image)
    #     text_feature = self.text_encoder(input_ids)
    #     feature = torch.cat([image_feature, text_feature], dim = -1)
    #     feature = self.mlp1(feature)
    #     return feature
    
    # def forward_image_features(self, image):
    #     image_feature = self.image_encoder(image)
    #     return image_feature
    
    # def forward_text_features(self, input_ids):
    #     text_feature = self.text_encoder(input_ids)
    #     return text_feature
    
    # def forward_through_features(self, image_feature, text_feature):
    #     feature = torch.cat([image_feature, text_feature], dim = -1)
    #     feature = self.mlp1(feature)
    #     output = self.mlp2(feature)
    #     return output
    
def load_clip_fusion(model_path,pretrain_path,num_classes,output_dim=512,is_distributed_model=True):
    '''
        model_path: pretuned_path
        pretrian_path: clip_path
    '''
    model = CLIPFusion(model_path=pretrain_path, num_classes=num_classes,output_dim=output_dim)
    if os.path.isfile(model_path):
        print('load')
        checkpoint = torch.load(model_path, map_location='cpu')
        checkpoint = {k.replace('module.',''):v for k,v in checkpoint.items()}
        real_checkpoint = {} 
        old_state = model.state_dict()
        for key,value in old_state.items():
            if 'mlp' in key and 'clip_model' not in key:
                real_checkpoint[key] = checkpoint[key]
            else :
                real_checkpoint[key] = old_state[key]

        model.load_state_dict(real_checkpoint)
    model.eval()
    return model


if __name__ == '__main__':
    
    clip = load_clip_fusion('path_to_clip',3,is_distributed_model=True,output_dim=1024)
