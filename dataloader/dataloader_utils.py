import gzip
import os
import sys
import io
import re
import random
import csv
import numpy as np
import torch
import string

def helper_name(x):
    name = x.split('/')[-1]
    return int(name.split('_')[0])

csv.field_size_limit(sys.maxsize)

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

def read_corpus(path, csvf=False , clean=True, MR=True, encoding='utf8', shuffle=False, lower=True):
    data = []
    labels = []
    if not csvf:
        with open(path, encoding=encoding) as fin:
            for line in fin:
                if MR:
                    label, sep, text = line.partition(' ')
                    label = int(label)
                else:
                    label, sep, text = line.partition(',')
                    label = int(label) - 1
                if clean:
                    text = clean_str(text.strip()) if clean else text.strip()
                if lower:
                    text = text.lower()
                labels.append(label)
                data.append(text.split())
    else:
        with open(path, "r") as f:
            reader = csv.reader(f, delimiter=",")
            for line in reader:
                text = line[0]
                label = int(line[1])
                if clean:
                    text = clean_str(text.strip()) if clean else text.strip()
                if lower:
                    text = text.lower()
                labels.append(label)
                data.append(text.split())

    if shuffle:
        perm = list(range(len(data)))
        random.shuffle(perm)
        data = [data[i] for i in perm]
        labels = [labels[i] for i in perm]

    return data, labels


def read_nli_data(filepath, target_model='infersent', lowercase=False, ignore_punctuation=False, stopwords=[]):
    """
    Read the premises, hypotheses and labels from some NLI dataset's
    file and return them in a dictionary. The file should be in the same
    form as SNLI's .txt files.

    Args:
        filepath: The path to a file containing some premises, hypotheses
            and labels that must be read. The file should be formatted in
            the same way as the SNLI (and MultiNLI) dataset.

    Returns:
        A dictionary containing three lists, one for the premises, one for
        the hypotheses, and one for the labels in the input data.
    """
    if target_model == 'Bert_NLI':
        labeldict = {"contradiction": 0,
                      "entailment": 1,
                      "neutral": 2}
    else:
        labeldict = {"entailment": 0,
                     "neutral": 1,
                     "contradiction": 2}
    with open(filepath, 'r', encoding='utf8') as input_data:
        premises, hypotheses, labels = [], [], []

        # Translation tables to remove punctuation from strings.
        punct_table = str.maketrans({key: ' ' for key in string.punctuation})

        for idx, line in enumerate(input_data):
            line = line.strip().split('\t')

            # Ignore sentences that have no gold label.
            if line[0] == '-':
                continue
            
            premise = line[1]
            hypothesis = line[2]

            if lowercase:
                premise = premise.lower()
                hypothesis = hypothesis.lower()

            if ignore_punctuation:
                premise = premise.translate(punct_table)
                hypothesis = hypothesis.translate(punct_table)

            # Each premise and hypothesis is split into a list of words.
            premises.append([w for w in premise.rstrip().split()
                             if w not in stopwords])
            hypotheses.append([w for w in hypothesis.rstrip().split()
                               if w not in stopwords])
            labels.append(labeldict[line[0]])

        return premises, hypotheses, labels


def read_train_text(dataset, data_dir="./data/train_dataset", shuffle = False):
    print("Reading the train dataset: %s" % (dataset))
    label_list = []
    clean_text_list = []
    if dataset == 'ag':
        with open(os.path.join(data_dir, dataset, "train.csv"), "r", encoding="utf-8") as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=",")
            count = 0
            for i, row in enumerate(csv_reader):
                if i == 0:
                    continue
                count += 1
                label_list.append(int(row[0]) - 1)
                text = " . ".join(row[1:]).lower()
                clean_text_list.append(clean_str(text.strip()).split())
    elif dataset == 'imdb':
        pos_list = []
        neg_list = []
        
        pos_path = os.path.join(data_dir, dataset + '/train/pos')
        neg_path = os.path.join(data_dir, dataset + '/train/neg')
        pos_files = [pos_path + '/' + x for x in os.listdir(pos_path) if x.endswith('.txt')]
        neg_files = [neg_path + '/' + x for x in os.listdir(neg_path) if x.endswith('.txt')]

        pos_files = sorted(pos_files, key=lambda x : helper_name(x))
        neg_files = sorted(neg_files, key=lambda x : helper_name(x))

        pos_list = [open(x, 'r', encoding='utf-8').read().lower().strip().replace('<br />', ' ') for x in pos_files]
        neg_list = [open(x, 'r', encoding='utf-8').read().lower().strip().replace('<br />', ' ') for x in neg_files]
        text_list = pos_list + neg_list
        # clean the texts
        clean_text_list = [clean_str(text.strip()).split() for text in text_list]
        label_list = [1]*len(pos_list) + [0]*len(neg_list)

    elif dataset == 'mr':
        if not os.path.exists(os.path.join(data_dir, dataset, 'mr_train')):
            text_set = []
            label_set = []
            with open(os.path.join(data_dir, dataset, 'rt-polarity.neg'), 'r', encoding='utf-8', errors='ignore') as file:
                lines = file.readlines()
                for line in lines:
                    text = line.strip(' ').strip('\n')
                    text_set.append(text)
                    label_set.append(0)

            with open(os.path.join(data_dir, dataset, 'rt-polarity.pos'), 'r', encoding='utf-8', errors='ignore') as file:
                lines = file.readlines()
                for line in lines:
                    text = line.strip(' ').strip('\n')
                    text_set.append(text)
                    label_set.append(1)
            random.seed(0)
            index =list(range(len(text_set)))
            random.shuffle(index)
            text_set = np.array(text_set)[index].tolist()
            label_set = np.array(label_set)[index].tolist()

            with open(os.path.join(data_dir, dataset, 'mr_train'), 'w') as f:
                for text, label in zip(text_set[:-1000], label_set[:-1000]):
                    f.write(str(label) + ' '+ text)
                    f.write('\n')
            
            with open(os.path.join(data_dir, dataset, 'mr'), 'w') as f:
                for text, label in zip(text_set[-1000:], label_set[-1000:]):
                    f.write(str(label) + ' '+ text)
                    f.write('\n')

        with open(os.path.join(data_dir, dataset, 'mr_train'), 'r') as f:
            for line in f.readlines():
                label, sep, text = line.partition(' ')
                label_list.append(int(label))
                clean_text_list.append(clean_str(text.strip(' ')).split(' '))  
    elif dataset == 'sst':
        pos_list = []
        neg_list = []
        
        with open(os.path.join(data_dir, dataset, 'train.tsv'), "r", encoding="utf-8") as fp:
            lines = fp.readlines()
            for line in lines[1:]:
                line = line.split('\t')
                clean_text_list.append(clean_str(line[0].lower().strip(' ')).split(' '))
                label_list.append(int(line[1]))
        if not os.path.exists(os.path.join(data_dir, dataset, 'sst')):
            test_text_list = []
            test_label_list = []
            with open(os.path.join(data_dir, dataset, 'dev.tsv'), "r", encoding="utf-8") as fp:
                lines = fp.readlines()
                for line in lines[1:]:
                    line = line.split('\t')
                    test_text_list.append(clean_str(line[0].lower().strip(' ')).split(' '))
                    test_label_list.append(int(line[1]))

            index =list(range(len(test_text_list)))
            random.shuffle(index)
            test_text_list = np.array(test_text_list)[index].tolist()
            test_label_list = np.array(test_label_list)[index].tolist()

            with open(os.path.join(data_dir, dataset, 'sst'), 'w') as f:
                for text, label in zip(test_text_list[:1000], test_label_list[:1000]):
                    f.write(str(label) + ' '+ ' '.join(text))
                    f.write('\n')
    else:
        raise NotImplementedError
    
    if shuffle:
        index =list(range(len(clean_text_list)))
        random.shuffle(index)
        clean_text_list = np.array(clean_text_list)[index].tolist()
        label_list = np.array(label_list)[index].tolist()
    
    return clean_text_list, label_list



def read_train_nli_text(dataset, data_dir="./data/train_dataset", shuffle = False):
    label_set = {'contradiction': 0, 'entailment': 1, 'neutral': 2}
    if dataset == 'snli':
        file_name = os.path.join(data_dir, dataset, 'snli_1.0_train.txt')
        with open(file_name, 'r') as f:
            rows = [row.split('\t') for row in f.readlines()[1:]]

        premises = [row[5].rstrip().split() for row in rows if row[0] in label_set]
        hypotheses = [row[6].rstrip().split() for row in rows if row[0] in label_set]
        labels = [label_set[row[0]] for row in rows if row[0] in label_set]

        # snli_data = pd.read_json(os.path.join(data_dir, dataset, 'snli_1.0_dev.jsonl'), lines=True)
        # premises = snli_data["sentence1"].apply(lambda premise: premise.rstrip().split())
        # hypotheses = snli_data["sentence2"].apply(lambda hypothese: hypothese.rstrip().split())
        # labels = snli_data["gold_label"].apply(lambda label: label_set[label])

    elif dataset == 'mnli':
        mnli_data = pd.read_json(os.path.join(data_dir, dataset, 'multinli_1.0_train.jsonl'), lines=True)
        premises = mnli_data["sentence1"].apply(lambda premise: premise.rstrip().split())
        hypotheses = mnli_data["sentence2"].apply(lambda hypothese: hypothese.rstrip().split())
        labels = mnli_data["gold_label"].apply(lambda label: label_set[label])
    
    if shuffle:
        index =list(range(len(premises)))
        random.shuffle(index)
        premises = np.array(premises)[index].tolist()
        hypotheses = np.array(hypotheses)[index].tolist()
        labels = np.array(labels)[index].tolist()

    return premises, hypotheses, labels



def read_test_text(dataset, data_dir="./data/train_dataset", shuffle = False):
    print("Reading the test dataset: %s" % (dataset))
    label_list = []
    clean_text_list = []
    if dataset == 'ag':
        with open(os.path.join(data_dir, dataset, "test.csv"), "r", encoding="utf-8") as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=",")
            count = 0
            for i, row in enumerate(csv_reader):
                if i == 0:
                    continue
                count += 1
                label_list.append(int(row[0]) - 1)
                text = " . ".join(row[1:]).lower()
                clean_text_list.append(clean_str(text.strip()).split())
    elif dataset == 'imdb':
        pos_list = []
        neg_list = []
        
        pos_path = os.path.join(data_dir, dataset + '/test/pos')
        neg_path = os.path.join(data_dir, dataset + '/test/neg')
        pos_files = [pos_path + '/' + x for x in os.listdir(pos_path) if x.endswith('.txt')]
        neg_files = [neg_path + '/' + x for x in os.listdir(neg_path) if x.endswith('.txt')]

        pos_files = sorted(pos_files, key=lambda x : helper_name(x))
        neg_files = sorted(neg_files, key=lambda x : helper_name(x))

        pos_list = [open(x, 'r', encoding='utf-8').read().lower().strip().replace('<br />', ' ') for x in pos_files]
        neg_list = [open(x, 'r', encoding='utf-8').read().lower().strip().replace('<br />', ' ') for x in neg_files]
        text_list = pos_list + neg_list
        # clean the texts
        clean_text_list = [clean_str(text.strip()).split() for text in text_list]
        label_list = [1]*len(pos_list) + [0]*len(neg_list)

    elif dataset == 'mr':
        if not os.path.exists(os.path.join(data_dir, dataset, 'mr')):
            text_set = []
            label_set = []
            with open(os.path.join(data_dir, dataset, 'rt-polarity.neg'), 'r', encoding='utf-8', errors='ignore') as file:
                lines = file.readlines()
                for line in lines:
                    text = line.strip(' ').strip('\n')
                    text_set.append(text)
                    label_set.append(0)

            with open(os.path.join(data_dir, dataset, 'rt-polarity.pos'), 'r', encoding='utf-8', errors='ignore') as file:
                lines = file.readlines()
                for line in lines:
                    text = line.strip(' ').strip('\n')
                    text_set.append(text)
                    label_set.append(1)
            random.seed(0)
            index =list(range(len(text_set)))
            random.shuffle(index)
            text_set = np.array(text_set)[index].tolist()
            label_set = np.array(label_set)[index].tolist()

            with open(os.path.join(data_dir, dataset, 'mr_train'), 'w') as f:
                for text, label in zip(text_set[:-1000], label_set[:-1000]):
                    f.write(str(label) + ' '+ text)
                    f.write('\n')
            
            with open(os.path.join(data_dir, dataset, 'mr'), 'w') as f:
                for text, label in zip(text_set[-1000:], label_set[-1000:]):
                    f.write(str(label) + ' '+ text)
                    f.write('\n')

        with open(os.path.join(data_dir, dataset, 'mr'), 'r') as f:
            for line in f.readlines():
                label, sep, text = line.partition(' ')
                label_list.append(int(label))
                clean_text_list.append(clean_str(text.strip(' ')).split(' '))  
    elif dataset == 'sst':
        pos_list = []
        neg_list = []
        
        with open(os.path.join(data_dir, dataset, 'dev.tsv'), "r", encoding="utf-8") as fp:
            lines = fp.readlines()
            for line in lines[1:]:
                line = line.split('\t')
                clean_text_list.append(clean_str(line[0].lower().strip(' ')).split(' '))
                label_list.append(int(line[1]))
    else:
        raise NotImplementedError
    
    if shuffle:
        index =list(range(len(clean_text_list)))
        random.shuffle(index)
        clean_text_list = np.array(clean_text_list)[index].tolist()
        label_list = np.array(label_list)[index].tolist()
    
    return clean_text_list, label_list


def read_test_nli_text(dataset, data_dir="./data/train_dataset", shuffle = False):
    label_set = {'contradiction': 0, 'entailment': 1, 'neutral': 2}
    if dataset == 'snli':
        file_name = os.path.join(data_dir, dataset, 'snli_1.0_dev.txt')
        with open(file_name, 'r') as f:
            rows = [row.split('\t') for row in f.readlines()[1:]]

        premises = [row[5].rstrip().split() for row in rows if row[0] in label_set]
        hypotheses = [row[6].rstrip().split() for row in rows if row[0] in label_set]
        labels = [label_set[row[0]] for row in rows if row[0] in label_set]

        # snli_data = pd.read_json(os.path.join(data_dir, dataset, 'snli_1.0_dev.jsonl'), lines=True)
        # premises = snli_data["sentence1"].apply(lambda premise: premise.rstrip().split())
        # hypotheses = snli_data["sentence2"].apply(lambda hypothese: hypothese.rstrip().split())
        # labels = snli_data["gold_label"].apply(lambda label: label_set[label])

    elif dataset == 'mnli':
        mnli_data = pd.read_json(os.path.join(data_dir, dataset, 'multinli_1.0_dev_matched.jsonl'), lines=True)
        premises = mnli_data["sentence1"].apply(lambda premise: premise.rstrip().split())
        hypotheses = mnli_data["sentence2"].apply(lambda hypothese: hypothese.rstrip().split())
        labels = mnli_data["gold_label"].apply(lambda label: label_set[label])
    
    if shuffle:
        index =list(range(len(premises)))
        random.shuffle(index)
        premises = np.array(premises)[index].tolist()
        hypotheses = np.array(hypotheses)[index].tolist()
        labels = np.array(labels)[index].tolist()

    return premises, hypotheses, labels
