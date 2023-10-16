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
from PIL import Image, ImageEnhance
import torchvision.transforms as transforms
from lavis.models import load_model_and_preprocess, load_preprocess
from lavis.models.albef_models.albef_classification import AlbefClassification
from omegaconf import OmegaConf
from torchvision.transforms.functional import InterpolationMode

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

def read_text_file(path, multi_line=False):
    c_text = clean_text(clean_text(open(path, 'r', encoding='latin-1').read().rstrip('\n')))
    return clean_str(c_text.strip()).split()

def read_image_file(path):
    # import pdb; pdb.set_trace()
    image = cv2.imread(path)[:, :, ::-1] #, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224), interpolation = cv2.INTER_AREA)
    image = image.transpose(2, 0, 1)
    return image

def label_to_num(label):
    labeldict = {"positive": 0, "negative": 1, "neutral": 2}
    return labeldict[label]


class MVSA_Dataset(Dataset):
    def __init__(self, data_dir = './data/dataset/MVSA/', mode = None, is_bert=False, max_seq_length= None, bert_dirname = None, word2id=None, week_process=False, is_clip=False):
        self.data_dir = data_dir
        self.text_image_dir = os.path.join(self.data_dir, 'data')
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
        # import pdb; pdb.set_trace()
        text_path, image_path, multimodal_label, text_label, image_label = self.pd_data.iloc[index]
        image_data = Image.open(os.path.join(self.text_image_dir, image_path))
        image_data = image_data.convert('RGB')
        multimodal_label = label_to_num(multimodal_label)
        if self.week_process:
            text_data = open(os.path.join(self.text_image_dir, text_path), 'r', encoding='latin-1').read().rstrip('\n')
            image_data = self.vis_processors(image_data)
            text_data = self.txt_processors(text_data)
            return image_data, text_data, torch.tensor(multimodal_label)
        
        text_data = read_text_file(os.path.join(self.text_image_dir, text_path))
        image_data = self.my_transform(image_data)
        text_label = label_to_num(text_label)
        image_label = label_to_num(image_label)

        if self.is_bert:
            input_ids, input_mask, segment_ids = utils.convert_example_to_feature_for_bert(' '.join(text_data), self.max_seq_length, self.tokenizer)
            return image_data, torch.tensor(input_ids), torch.tensor(input_mask), torch.tensor(segment_ids), torch.tensor(multimodal_label)
        else:
            input_ids, input_mask = utils.convert_example_to_feature_for_cnn(text_data, self.word2id, self.max_seq_length)
            return image_data, torch.tensor(input_ids), torch.tensor(multimodal_label)


class MVSA_AdvDataset(object):
    def __init__(self, data_dir = './data/dataset/MVSA/', mode = 'train', is_bert=False, max_seq_length= None, bert_dirname = None, word2id=None, week_process=False, is_clip=False):
        self.data_dir = data_dir
        self.text_image_dir = os.path.join(self.data_dir, 'data')
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
        text_path, image_path, multimodal_label, text_label, image_label = self.pd_data.iloc[index]
        image_data = Image.open(os.path.join(self.text_image_dir, image_path))
        image_data = image_data.convert('RGB')
        multimodal_label = label_to_num(multimodal_label)
        if self.week_process:
            text_data = open(os.path.join(self.text_image_dir, text_path), 'r', encoding='latin-1').read().rstrip('\n')
            image_data = self.my_transform(image_data)
            text_data = self.txt_processors(text_data)
            text_data = text_data.split()
            return text_data,image_data, multimodal_label
        
        text_data = read_text_file(os.path.join(self.text_image_dir, text_path))
        image_data = self.my_transform(image_data)
        text_label = label_to_num(text_label)
        image_label = label_to_num(image_label)

        # if self.is_bert:
        #     input_ids, input_mask, segment_ids = utils.convert_example_to_feature_for_bert(' '.join(text_data), self.max_seq_length, self.tokenizer)
        #     return torch.FloatTensor(image_data), torch.tensor(input_ids), torch.tensor(input_mask), torch.tensor(segment_ids), torch.tensor(multimodal_label), torch.tensor(text_label), torch.tensor(image_label)
        # else:
        #     input_ids, input_mask = utils.convert_example_to_feature_for_cnn(text_data, self.word2id, self.max_seq_length)
        #     return torch.FloatTensor(image_data), torch.tensor(input_ids), torch.tensor(multimodal_label), torch.tensor(text_label), torch.tensor(image_label)


class MVSA_AdvTrainDataset(Dataset):
    def __init__(self, train_data_dir = './data/dataset/MVSA_Single_AdvTrain/', test_data_dir = './dataset/MVSA_Sinlge/', mode = None, is_bert=False, max_seq_length= None, bert_dirname = None, word2id=None):
        self.is_bert = is_bert
        self.max_seq_length = max_seq_length
        
        self.my_transform = transforms.Compose(
            [transforms.Resize((224, 224)), transforms.ToTensor()]
        )
        self.mode = mode

        if mode == 'train':
            self.pd_data = pd.read_csv(os.path.join(train_data_dir, 'adv_train_data.csv'), index_col = 0)
            self.text_image_dir = os.path.join(train_data_dir, 'data')
        else:
            self.pd_data = pd.read_csv(os.path.join(test_data_dir, 'test_data.csv'), index_col = 0)
            self.text_image_dir = os.path.join(test_data_dir, 'data')
        

        if self.is_bert:
            self.tokenizer = BertTokenizer.from_pretrained(bert_dirname, do_lower_case=True)
        
        self.word2id = word2id

    def __len__(self):
        return len(self.pd_data)
    
    def __getitem__(self, index):
        # import pdb; pdb.set_trace()
        if self.mode == 'train':
            index, image_path, text, multimodal_label = self.pd_data.iloc[index]
            # import pdb;pdb.set_trace()
            text_data = str(text).split(' ')
            image_data = Image.open(os.path.join(self.text_image_dir, image_path))
            image_data = image_data.convert('RGB')
            image_data = self.my_transform(image_data)
        else:
            text_path, image_path, multimodal_label, text_label, image_label = self.pd_data.iloc[index]
            text_data = read_text_file(os.path.join(self.text_image_dir, text_path))
            image_data = Image.open(os.path.join(self.text_image_dir, image_path))
            image_data = image_data.convert('RGB')
            image_data = self.my_transform(image_data)
            multimodal_label = label_to_num(multimodal_label)
            text_label = label_to_num(text_label)
            image_label = label_to_num(image_label)

        if self.is_bert:
            input_ids, input_mask, segment_ids = utils.convert_example_to_feature_for_bert(' '.join(text_data), self.max_seq_length, self.tokenizer)
            return image_data, torch.tensor(input_ids), torch.tensor(input_mask), torch.tensor(segment_ids), torch.tensor(multimodal_label)
        else:
            input_ids, input_mask = utils.convert_example_to_feature_for_cnn(text_data, self.word2id, self.max_seq_length)
            return image_data, torch.tensor(input_ids), torch.tensor(multimodal_label)


class MVSA_Contrastive_AdvTrainDataset(Dataset):
    def __init__(self, train_data_dir = './dataset/MVSA_Single_AdvTrain/', test_data_dir = './dataset/MVSA_Sinlge/', mode = None, is_bert=False, max_seq_length= None, bert_dirname = None, word2id=None):
        self.is_bert = is_bert
        self.max_seq_length = max_seq_length
        
        self.my_transform = transforms.Compose(
            [transforms.Resize((224, 224)), transforms.ToTensor()]
        )
        self.mode = mode

        if mode == 'train':
            self.pd_data = pd.read_csv(os.path.join(train_data_dir, 'adv_train_data_contrastive.csv'), index_col = 0)
            self.text_image_dir = os.path.join(train_data_dir, 'data')
        else:
            self.pd_data = pd.read_csv(os.path.join(test_data_dir, 'test_data.csv'), index_col = 0)
            self.text_image_dir = os.path.join(test_data_dir, 'data')
        

        if self.is_bert:
            self.tokenizer = BertTokenizer.from_pretrained(bert_dirname, do_lower_case=True)
        
        self.word2id = word2id

    def __len__(self):
        return len(self.pd_data)
    
    def __getitem__(self, index):
        # import pdb; pdb.set_trace()
        if self.mode == 'train':
            index, ori_image_path, adv_image_path, ori_text, adv_text, multimodal_label = self.pd_data.iloc[index]
            ori_text_data = str(ori_text).split(' ')
            ori_image_data = Image.open(os.path.join(self.text_image_dir, ori_image_path))
            ori_image_data = ori_image_data.convert('RGB')
            ori_image_data = self.my_transform(ori_image_data)

            adv_text_data = str(adv_text).split(' ')
            adv_image_data = Image.open(os.path.join(self.text_image_dir, adv_image_path))
            adv_image_data = adv_image_data.convert('RGB')
            adv_image_data = self.my_transform(adv_image_data)

            if self.is_bert:
                ori_input_ids, ori_input_mask,ori_segment_ids = utils.convert_example_to_feature_for_bert(' '.join(ori_text_data), self.max_seq_length, self.tokenizer)
                adv_input_ids, adv_input_mask, adv_segment_ids = utils.convert_example_to_feature_for_bert(' '.join(adv_text_data), self.max_seq_length, self.tokenizer)
                return ori_image_data, adv_image_data, torch.tensor(ori_input_ids), torch.tensor(ori_input_mask), torch.tensor(ori_segment_ids), torch.tensor(adv_input_ids), torch.tensor(adv_input_mask), torch.tensor(adv_segment_ids), torch.tensor(multimodal_label)
            else:
                ori_input_ids, ori_input_mask = utils.convert_example_to_feature_for_cnn(ori_text_data, self.word2id, self.max_seq_length)
                adv_input_ids, adv_input_mask = utils.convert_example_to_feature_for_cnn(adv_text_data, self.word2id, self.max_seq_length)
                return ori_image_data, adv_image_data, torch.tensor(ori_input_ids), torch.tensor(adv_input_ids), torch.tensor(multimodal_label)

        else:
            text_path, image_path, multimodal_label, text_label, image_label = self.pd_data.iloc[index]
            text_data = read_text_file(os.path.join(self.text_image_dir, text_path))
            image_data = Image.open(os.path.join(self.text_image_dir, image_path))
            image_data = image_data.convert('RGB')
            image_data = self.my_transform(image_data)
            multimodal_label = label_to_num(multimodal_label)
            text_label = label_to_num(text_label)
            image_label = label_to_num(image_label)

            if self.is_bert:
                input_ids, input_mask, segment_ids = utils.convert_example_to_feature_for_bert(' '.join(text_data), self.max_seq_length, self.tokenizer)
                return image_data, torch.tensor(input_ids), torch.tensor(input_mask), torch.tensor(segment_ids), torch.tensor(multimodal_label)
            else:
                input_ids, input_mask = utils.convert_example_to_feature_for_cnn(text_data, self.word2id, self.max_seq_length)
                return image_data, torch.tensor(input_ids), torch.tensor(multimodal_label)


class MVSA_Physical_AdvExamplesDataset(Dataset):
    def __init__(self, data_dir = './dataset/pgd_adv_examples/', mode = None, is_bert=False, max_seq_length= None, bert_dirname = None, word2id=None, enhance=None, week_process=False, is_clip=False):
        self.is_bert = is_bert
        self.max_seq_length = max_seq_length
        self.week_process = week_process
        
        self.my_transform = transforms.Compose(
            [transforms.Resize((224, 224)), transforms.ToTensor()]
        )
        self.mode = mode

        self.pd_data = pd.read_csv(os.path.join(data_dir, 'adv_examples.csv'), index_col = 0)
        self.text_image_dir = os.path.join(data_dir, 'data')
        

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
        self.enhance = enhance


    def __len__(self):
        return len(self.pd_data)
    
    def __getitem__(self, index):
        # import pdb; pdb.set_trace()
        index, ori_image_path, adv_image_path, ori_text, adv_text, multimodal_label = self.pd_data.iloc[index]
        text_data = str(adv_text).split(' ')
        image_data = Image.open(os.path.join(self.text_image_dir, adv_image_path))
        image_data = image_data.convert('RGB')

        Enhancer = None
        # enhance picutre
        if self.enhance == 'Brightness':
            Enhancer = ImageEnhance.Brightness(image_data)
        elif self.enhance == 'Contrast':
            Enhancer = ImageEnhance.Contrast(image_data)
        elif self.enhance == 'Color':
            Enhancer = ImageEnhance.Color(image_data)
        elif self.enhance == 'Sharpness':
            Enhancer = ImageEnhance.Sharpness(image_data)
        
        if Enhancer is not None:
            degree = np.random.uniform(low=0.5, high=1.5)
            image_data = Enhancer.enhance(degree)

        if self.week_process:
            image_data = self.vis_processors(image_data)
            text_data = self.txt_processors(text_data)
            return image_data, text_data, torch.tensor(multimodal_label)

        image_data = self.my_transform(image_data)

        if self.is_bert:
            input_ids, input_mask, segment_ids = utils.convert_example_to_feature_for_bert(' '.join(text_data), self.max_seq_length, self.tokenizer)
            return image_data, torch.tensor(input_ids), torch.tensor(input_mask), torch.tensor(segment_ids), torch.tensor(multimodal_label)
        else:
            input_ids, input_mask = utils.convert_example_to_feature_for_cnn(text_data, self.word2id, self.max_seq_length)
            return image_data, torch.tensor(input_ids), torch.tensor(multimodal_label)