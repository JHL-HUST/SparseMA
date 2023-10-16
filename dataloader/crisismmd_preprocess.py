import numpy as np
import pandas as pd
import re
import string
import os
import argparse
import sys


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(sys.argv[0], conflict_handler='resolve')
    argparser.add_argument("--dataset_filename", type=str, default="CrisisMMD_v2", help="The dirname to fold CrisisMMD_v2 dataset")
    args = argparser.parse_args()

    labeldict = {"california_wildfires": 0, "hurricane_harvey": 1, "hurricane_irma": 2, "hurricane_maria": 3, "iraq_iran_earthquake": 4, "mexico_earthquake":5, "srilanka_floods": 6}

    annotations_dir = os.path.join(args.dataset_filename, 'annotations')
    annotations_filenames = os.listdir(annotations_dir)

    all_image_paths = []
    all_text_data = []
    all_labels = []
    for annotations_filename in annotations_filenames:
        pd_data = pd.read_csv(os.path.join(annotations_dir, annotations_filename), sep='\t')
        image_paths = list(pd_data['image_path'])
        all_image_paths.extend(image_paths)
        texts = list(pd_data['tweet_text'])
        all_text_data.extend(texts)
        for key, value in labeldict.items():
            if key in annotations_filename:
                labels = [value] * len(pd_data)
                all_labels.extend(labels)
                break
    
    # split the dataset into trainset and testset
    np.random.seed(0)
    shuffle_indices = list(range(len(all_image_paths)))
    np.random.shuffle(shuffle_indices)
    all_image_paths, all_text_data, all_labels = np.array(all_image_paths)[shuffle_indices], np.array(all_text_data)[shuffle_indices], np.array(all_labels)[shuffle_indices]

    Num = len(all_image_paths)
    train_num = Num - 1000
    train_dataframe_dict = {'image_path': all_image_paths[:train_num], 'text_data': all_text_data[:train_num], 'label': all_labels[:train_num]}
    train_dataframe = pd.DataFrame(train_dataframe_dict)
    train_dataframe.to_csv(os.path.join(args.dataset_filename, 'train_data.csv'))

    test_dataframe_dict = {'image_path': all_image_paths[train_num:], 'text_data': all_text_data[train_num:], 'label': all_labels[train_num:]}
    test_dataframe = pd.DataFrame(test_dataframe_dict)
    test_dataframe.to_csv(os.path.join(args.dataset_filename, 'test_data.csv'))

    print('Preprocess the data in {} folder done'.format(args.dataset_filename))
