import os
import gc
import cv2
import h5py
import random
import numpy as np
import pandas as pd
# import tensorflow as tf

from tqdm import tqdm
from  matplotlib import pyplot as plt
import argparse
import sys
from PIL import Image

def get_data_paths(path, extension):
    ''' Get list of data paths with input extension and sort by its filename (ID)
    path: Folder path
    extension: File extension wants to get
    '''
    paths = os.listdir(path)
    paths = list(filter(lambda x: x.endswith(extension), paths))
    paths.sort(key = lambda x : int(x.split('.')[0]))
    paths = [x for x in paths]
    return paths

def read_image_file(path):
    try:
        image = cv2.imread(path)[:, :, ::-1] #, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224), interpolation = cv2.INTER_AREA)
        image_data = Image.open(path)
        image_data = image_data.convert('RGB')
#         image = tf.keras.utils.load_img(path, target_size=IMAGE_SIZE)
#         image = tf.keras.preprocessing.image.img_to_array(image)
        invalid_ID = -1
    except:
        image = np.zeros((224, 224, 3))
        invalid_ID = int(os.path.split(path)[1].split('.')[0])
    return image, invalid_ID


# there are 3 annotators labelling each modality labels in the MVSA-Multiple dataset
# merge those 3 label pairs into 1 pair by taking majority vote on each modality label
# since there are only 3 different labels, if 1 modality receives 3 different labels from 3 annotators
# => the data pair contains it is considered invalid
def merge_multi_label(dataframe):
    anno_1 = list(dataframe.loc[:, ['text', 'image']].itertuples(index=False, name=None))
    anno_2 = list(dataframe.loc[:, ['text.1', 'image.1']].itertuples(index=False, name=None))
    anno_3 = list(dataframe.loc[:, ['text.2', 'image.2']].itertuples(index=False, name=None))
    IDs = list(dataframe.iloc[:, 0])
    
    valid_pairs = []
    
    for i in range(len(anno_1)):
        pairs = [anno_1[i], anno_2[i], anno_3[i]]
        ID = IDs[i]
        
        text_labels = [pair[0] for pair in pairs]
        image_labels = [pair[1] for pair in pairs]
        
        max_occur_text_label = max(text_labels, key=text_labels.count)
        max_occur_image_label = max(image_labels, key=image_labels.count)

        if text_labels.count(max_occur_text_label) > 1 and image_labels.count(max_occur_image_label) > 1:
            valid_pair = (ID, max_occur_text_label, max_occur_image_label)
        else:
            valid_pair = (ID, 'invalid', 'invalid')
        valid_pairs.append(valid_pair)
    valid_dataframe = pd.DataFrame(valid_pairs, columns=['ID', 'text', 'image'])
    return valid_dataframe

def multimodal_label(text_label, image_label):
    if text_label == image_label:
        label = text_label
    elif (text_label == 'positive' and image_label == 'negative') or (text_label == 'negative' and image_label == 'positive'):
        label = 'invalid'
    elif (text_label == 'neutral' and image_label != 'neutral') or (text_label != 'neutral' or image_label == 'neutral'):
        label = image_label if text_label == 'neutral' else text_label
    return label

def read_labels_file(path):
    dataframe = pd.read_csv(path, sep="\s+|,", engine="python")
    return dataframe

def create_multimodal_labels(path, multiple=False, mappings=False):
    dataframe = read_labels_file(path)
    
    if multiple == True:
        dataframe = merge_multi_label(dataframe)

    labels = []
    for label_pair in dataframe.loc[:, ['text', 'image']].values:
        label = multimodal_label(label_pair[0], label_pair[1])
        labels.append(label)
        
    if mappings == True:
        label_map = {}
        for i in range(len(labels)):
            ID = dataframe.iloc[i, 0]
            label_map[ID] = labels[i]            
        return label_map
    
    return np.array(labels, dtype='object')

def create_original_labels(path, multiple=False):
    dataframe = read_labels_file(path)
    
    if multiple == True:
        dataframe = merge_multi_label(dataframe)
        
    text_labels = dataframe['text'].to_numpy()
    image_labels = dataframe['image'].to_numpy()
    return text_labels, image_labels


def preprocess_mvsa_data(data_path, multiple = False):
    invalid_indices = []

    text_paths = get_data_paths(os.path.join(data_path, 'data'), '.txt')
    image_paths = get_data_paths(os.path.join(data_path, 'data'), '.jpg')

    for i, (text_path, image_path) in enumerate(zip(text_paths, image_paths)):
        image, invalid_ID = read_image_file(os.path.join(data_path, 'data', image_path))

        if invalid_ID != -1:
            invalid_indices.append(i)
    mvsa_single_multimodal_labels = create_multimodal_labels(os.path.join(data_path, 'labelResultAll.txt'), multiple = multiple)
    mvsa_single_text_labels, mvsa_single_image_labels = create_original_labels(os.path.join(data_path, 'labelResultAll.txt'), multiple = multiple)

    # Get invalid label indices
    mvsa_single_multimodal_labels_invalid_indices = [i for i in range(len(text_paths)) if mvsa_single_multimodal_labels[i] == 'invalid']
    invalid_indices.extend(mvsa_single_multimodal_labels_invalid_indices)
    invalid_indices = list(set(invalid_indices))

    valid_indices = []
    for i in range(len(text_paths)):
        if i not in invalid_indices:
            valid_indices.append(i)
    
    text_paths = np.array(text_paths)[valid_indices]
    image_paths = np.array(image_paths)[valid_indices]
    mvsa_single_multimodal_labels = np.array(mvsa_single_multimodal_labels)[valid_indices]
    mvsa_single_text_labels = np.array(mvsa_single_text_labels)[valid_indices]
    mvsa_single_image_labels = np.array(mvsa_single_image_labels)[valid_indices]

    dataframe_dict = {'text_path': text_paths, 'image_path': image_paths, 'multimodal_label': mvsa_single_multimodal_labels, 'text_label': mvsa_single_text_labels, 'image_label':mvsa_single_image_labels}
    dataframe = pd.DataFrame(dataframe_dict)
    # import pdb; pdb.set_trace()
    dataframe.to_csv(os.path.join(data_path, 'all_data.csv'))

    # split the dataset into trainset and testset
    np.random.seed(0)
    shuffle_indices = list(range(len(text_paths)))
    np.random.shuffle(shuffle_indices)
    text_paths, image_paths, mvsa_single_multimodal_labels, mvsa_single_text_labels, mvsa_single_image_labels = text_paths[shuffle_indices], image_paths[shuffle_indices], mvsa_single_multimodal_labels[shuffle_indices], mvsa_single_text_labels[shuffle_indices], mvsa_single_image_labels[shuffle_indices]
    # import pdb; pdb.set_trace()
    Num = len(text_paths)
    train_num = Num - 1000
    train_dataframe_dict = {'text_path': text_paths[:train_num], 'image_path': image_paths[:train_num], 'multimodal_label': mvsa_single_multimodal_labels[:train_num], 'text_label': mvsa_single_text_labels[:train_num], 'image_label':mvsa_single_image_labels[:train_num]}
    train_dataframe = pd.DataFrame(train_dataframe_dict)
    train_dataframe.to_csv(os.path.join(data_path, 'train_data.csv'))

    test_dataframe_dict = {'text_path': text_paths[train_num:], 'image_path': image_paths[train_num:], 'multimodal_label': mvsa_single_multimodal_labels[train_num:], 'text_label': mvsa_single_text_labels[train_num:], 'image_label':mvsa_single_image_labels[train_num:]}
    test_dataframe = pd.DataFrame(test_dataframe_dict)
    test_dataframe.to_csv(os.path.join(data_path, 'test_data.csv'))

    print('Preprocess the data in {} folder done'.format(data_path))



if __name__ == '__main__':
    argparser = argparse.ArgumentParser(sys.argv[0], conflict_handler='resolve')
    argparser.add_argument("--mvsa_single_data_path", type=str, default="MVSA_Single", help="The dirname to fold MVSA_Single dataset")
    argparser.add_argument("--mvsa_multiple_data_path", type=str, default="MVSA", help="The dirname to fold MVSA dataset")

    args = argparser.parse_args()

    # preprocess the data on MVSA_Single
    preprocess_mvsa_data(args.mvsa_single_data_path)

    # preprocess the data on MVSA
    preprocess_mvsa_data(args.mvsa_multiple_data_path)