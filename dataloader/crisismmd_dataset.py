import numpy as np
import pandas as pd
import pickle
import os
import utils
# from model_loader.bert_package import BertTokenizer
from transformers import RobertaTokenizer, BertTokenizer
from torch.utils.data import Dataset
import torch
from dataloader import dataloader_utils
import re
import string
import cv2
from PIL import Image
import torchvision.transforms as transforms
import json
from lavis.models import load_preprocess
from torchvision.transforms.functional import InterpolationMode
from omegaconf import OmegaConf

def clean_text(txt):
    at_pattern = re.compile('@[a-zA-Z0-9]+')
    http_pattern = re.compile("((http|ftp|https)://)(([a-zA-Z0-9\._-]+\.[a-zA-Z]{2,6})|([0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}))(:[0-9]{1,4})*(/[a-zA-Z0-9\&%_\./-~-]*)?")
    punc_pattern = re.compile('[%s]' % re.escape(string.punctuation))
    txt = re.sub('#', '', txt)
    txt = re.sub(at_pattern, 'user', txt)
    txt = re.sub(http_pattern, 'link', txt)
#     txt = re.sub(punc_pattern, '', txt) # ONLY REMOVE punc for word2vec not BERT
    # as the data was crawled using Twitter API, it marked retweet data with RT <user> tag which has no meaning considering it in training
    if txt.startswith('RT user'):
        txt = ''.join(txt.split(':')[1:])    
    txt = txt.strip().lower()
    return txt

def clean_str(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip() if TREC else string.strip().lower()


class CrisisMMD_Dataset(Dataset):
    def __init__(self, data_dir = './dataset/CrisisMMD_v2.0/', mode = None, is_bert=False, max_seq_length= None, bert_dirname = None, word2id=None, week_process=False, is_clip=False):
        self.data_dir = data_dir
        self.is_bert = is_bert
        self.max_seq_length = max_seq_length
        self.week_process = week_process
        
        self.my_transform = transforms.Compose(
            [transforms.Resize((224, 224)), transforms.ToTensor()]
        )
        if mode == 'train':
            self.pd_data = pd.read_csv(os.path.join(self.data_dir, 'train_data.csv'), index_col = 0)
        else:
            self.pd_data = pd.read_csv(os.path.join(self.data_dir, 'test_data.csv'), index_col = 0)

        if self.is_bert:
            self.tokenizer = BertTokenizer.from_pretrained(bert_dirname, do_lower_case=True)
        
        if self.week_process:
            if is_clip:
                preprocess_cfg = OmegaConf.create({'vis_processor':{'train':{'name':'blip_image_train','image_size':224},'eval':{'name':'blip_image_eval','image_size':224}},'txt_processor':{'train':{'name':'blip_caption'},'eval':{'name':'blip_caption'}}})
            else:
                preprocess_cfg = OmegaConf.create({'vis_processor':{'train':{'name':'blip_image_train'},'eval':{'name':'blip_image_eval'}},'txt_processor':{'train':{'name':'blip_caption'},'eval':{'name':'blip_caption'}}})
            vis_processors, txt_processors = load_preprocess(preprocess_cfg)
            if mode == 'train':
                self.vis_processors = vis_processors['train']
                self.txt_processors = txt_processors['train']
            else :
                self.vis_processors = vis_processors['eval']
                self.txt_processors = txt_processors['eval']

        self.word2id = word2id

    def __len__(self):
        return len(self.pd_data)
    
    def __getitem__(self, index):
        image_path, text_data, label = self.pd_data.iloc[index]


        # preprocess image data
        image_data = Image.open(os.path.join(self.data_dir, image_path))
        image_data = image_data.convert('RGB')
        if self.week_process:
            image_data = self.vis_processors(image_data)
            text_data = self.txt_processors(text_data)
            return image_data, text_data, torch.tensor(label)

        image_data = self.my_transform(image_data)

        # preprocess text data
        text_data = clean_str(text_data.strip()).split()

        if self.is_bert:
            input_ids, input_mask, segment_ids = utils.convert_example_to_feature_for_bert(' '.join(text_data), self.max_seq_length, self.tokenizer)
            return image_data, torch.tensor(input_ids), torch.tensor(input_mask), torch.tensor(segment_ids), torch.tensor(label)
        else:
            input_ids, input_mask = utils.convert_example_to_feature_for_cnn(text_data, self.word2id, self.max_seq_length)
            return image_data, torch.tensor(input_ids), torch.tensor(label)


class CrisisMMD_AdvDataset(object):
    def __init__(self, data_dir = './dataset/CrisisMMD_v2.0/', mode = 'train', is_bert=False, max_seq_length= None, bert_dirname = None, word2id=None, week_process=False, is_clip=False):
        self.data_dir = data_dir
        self.is_bert = is_bert
        self.max_seq_length = max_seq_length
        self.week_process = week_process
        
        self.my_transform = transforms.Compose(
            [transforms.Resize((224, 224)), transforms.ToTensor()]
        )

        self.pd_data = pd.read_csv(os.path.join(self.data_dir, 'test_data.csv'), index_col = 0)

        if self.is_bert:
            self.tokenizer = BertTokenizer.from_pretrained(bert_dirname, do_lower_case=True)
        
        if self.week_process:
            if is_clip:
                preprocess_cfg = OmegaConf.create({'vis_processor':{'train':{'name':'blip_image_train','image_size':224},'eval':{'name':'blip_image_eval','image_size':224}},'txt_processor':{'train':{'name':'blip_caption'},'eval':{'name':'blip_caption'}}})
            else:
                preprocess_cfg = OmegaConf.create({'vis_processor':{'train':{'name':'blip_image_train'},'eval':{'name':'blip_image_eval'}},'txt_processor':{'train':{'name':'blip_caption'},'eval':{'name':'blip_caption'}}})
                self.my_transform = transforms.Compose(
                    [transforms.Resize((384, 384), interpolation=InterpolationMode.BICUBIC), 
                    transforms.ToTensor()]
                )
            vis_processors, txt_processors = load_preprocess(preprocess_cfg)
            if mode == 'train':
                self.vis_processors = vis_processors['train']
                self.txt_processors = txt_processors['train']
            else :
                self.vis_processors = vis_processors['eval']
                self.txt_processors = txt_processors['eval']

        self.word2id = word2id

    def __len__(self):
        return len(self.pd_data)
    
    def __getitem__(self, index):
        image_path, text_data, label = self.pd_data.iloc[index]

        # preprocess image data
        image_data = Image.open(os.path.join(self.data_dir, image_path))
        image_data = image_data.convert('RGB')

        if self.week_process:
            image_data = self.my_transform(image_data)
            text_data = self.txt_processors(text_data)
            text_data = text_data.split()
            return text_data, image_data, label

        image_data = self.my_transform(image_data)

        # preprocess text data
        text_data = clean_str(text_data.strip()).split()

        return text_data, image_data, label