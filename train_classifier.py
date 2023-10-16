import os
import sys
import argparse
import time
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import dataloader
import utils
import model_loader
import config as config_module
from tqdm import tqdm
from torch.utils.data import DataLoader, BatchSampler
from torch.utils.data.distributed import DistributedSampler
from model_loader.bert_package.optimization import BertAdam

import torch.distributed as dist
from datetime import timedelta
import torch.nn.functional as F
from torch.autograd import Variable
    
def train_model(epoch, model, optimizer, criterion,
        train_loader, test_loader,
        num_classes, best_test, save_path, device, is_bert,
        is_lavis=False,is_clip=False):

    model.train()
    niter = epoch*len(train_loader)
    loss_list = []
    correct = 0.0
    cnt = 0
    
    each_class_correct = [0 for i in range(num_classes)]
    each_class_sum = [0 for i in range(num_classes)]
    if is_bert:
        for step, (image, input_ids, input_mask, segment_ids, multimodal_label) in enumerate(tqdm(train_loader, desc="Iteration")):
            niter += 1
            cnt += 1
            model.zero_grad()
            image = Variable(image.to(device))
            input_ids, input_mask, segment_ids, multimodal_label = Variable(input_ids.to(device)), Variable(input_mask.to(device)), Variable(segment_ids.to(device)), Variable(multimodal_label.to(device))
            output = model(image, input_ids, input_mask, segment_ids)
            loss = criterion(output, multimodal_label)
            loss.backward()
            optimizer.step()

            loss_list.append(loss.item())
            
            pred = output.data.max(1)[1]
            correct += pred.eq(multimodal_label.data).cpu().sum()
            cnt += multimodal_label.numel()
            
            multimodal_label = multimodal_label.cpu().data
            pred = pred.cpu().data
            for each_label in range(num_classes):
                each_label_index = multimodal_label == each_label
                each_class_sum[each_label] += each_label_index.sum()
                each_class_correct[each_label] += (pred[each_label_index] == each_label).sum()
    elif is_lavis:
        for step, (image, input_ids, multimodal_label) in enumerate(tqdm(train_loader, desc="Iteration")):
            niter += 1
            model.zero_grad()
            image, multimodal_label = Variable(image.to(device)), Variable(multimodal_label.to(device))
            samples = {"image": image, "text_input": input_ids,'label':multimodal_label}
            output = model(samples)

            if is_clip:
                pred = output.data.max(1)[1]
                loss = criterion(output, multimodal_label)
            else:
                pred = output.logits.data.max(1)[1]
                loss = output.loss
            loss.backward()
            optimizer.step()

            loss_list.append(loss.item())
            
            correct += pred.eq(multimodal_label.data).cpu().sum()
            cnt += multimodal_label.numel()
            
            multimodal_label = multimodal_label.cpu().data
            pred = pred.cpu().data
            for each_label in range(num_classes):
                each_label_index = multimodal_label == each_label
                each_class_sum[each_label] += each_label_index.sum()
                each_class_correct[each_label] += (pred[each_label_index] == each_label).sum()
             
    else:
        for step, (image, input_ids, multimodal_label) in enumerate(tqdm(train_loader, desc="Iteration")):
            niter += 1
            model.zero_grad()
            image, input_ids, multimodal_label = Variable(image.to(device)), Variable(input_ids.to(device)), Variable(multimodal_label.to(device))
            output = model(image, input_ids)
            loss = criterion(output, multimodal_label)
            loss.backward()
            optimizer.step()

            loss_list.append(loss.item())
            
            pred = output.data.max(1)[1]
            correct += pred.eq(multimodal_label.data).cpu().sum()
            cnt += multimodal_label.numel()
            
            multimodal_label = multimodal_label.cpu().data
            pred = pred.cpu().data
            for each_label in range(num_classes):
                each_label_index = multimodal_label == each_label
                each_class_sum[each_label] += each_label_index.sum()
                each_class_correct[each_label] += (pred[each_label_index] == each_label).sum()
            
    
    train_acc = correct/cnt
    print("Epoch={} Loss={} train_acc={:.6f}".format(epoch, np.mean(loss_list), train_acc))
    for i in range(num_classes):
        print("The acc of class {} is {}".format(i, each_class_correct[i]/each_class_sum[i]))

    test_acc, each_class_correct, each_class_sum = eval_model(niter, model, test_loader, num_classes, device, is_bert,is_lavis=is_lavis,is_clip=is_clip)

    sys.stdout.write("Epoch={} iter={} lr={:.6f} train_loss={:.6f} train_acc = {:.6f} test_acc={:.6f}\n".format(
        epoch, niter,
        optimizer.param_groups[0]['lr'],
        loss.item(), train_acc,
        test_acc
    ))
    for i in range(num_classes):
        print("The acc of class {} is {}".format(i, each_class_correct[i]/each_class_sum[i]))

    if test_acc > best_test:
        best_test = test_acc
        if save_path:
            torch.save(model.state_dict(), save_path)
    sys.stdout.write("\n")
    return best_test

def eval_model(niter, model, test_loader, num_classes, device, is_bert,is_lavis=False,is_clip=False):
    model.eval()
    # N = len(valid_x)
    # criterion = nn.CrossEntropyLoss()
    correct = 0.0
    cnt = 0.
    # total_loss = 0.0
    
    each_class_correct = [0 for i in range(num_classes)]
    each_class_sum = [0 for i in range(num_classes)]
    with torch.no_grad():
        if is_bert:
            for step, (image, input_ids, input_mask, segment_ids, multimodal_label) in enumerate(tqdm(test_loader, desc="Iteration")):
                image = Variable(image.to(device))
                input_ids, input_mask, segment_ids, multimodal_label = Variable(input_ids.to(device)), Variable(input_mask.to(device)), Variable(segment_ids.to(device)), Variable(multimodal_label.to(device))
                output = model(image, input_ids, input_mask, segment_ids)
                pred = output.data.max(1)[1]
                correct += pred.eq(multimodal_label.data).cpu().sum()
                cnt += multimodal_label.numel()
                
                multimodal_label = multimodal_label.cpu().data
                pred = pred.cpu().data
                for each_label in range(num_classes):
                    each_label_index = multimodal_label == each_label
                    each_class_sum[each_label] += each_label_index.sum()
                    each_class_correct[each_label] += (pred[each_label_index] == each_label).sum()
        elif is_lavis:
            for step, (image, input_ids, multimodal_label) in enumerate(tqdm(test_loader, desc="Iteration")):
                image, multimodal_label = Variable(image.to(device)), Variable(multimodal_label.to(device))
                samples = {"image": image, "text_input": input_ids,'label':multimodal_label}
                output = model(samples)

                if is_clip:
                    pred = output.data.max(1)[1]
                else:
                    pred = output.logits.data.max(1)[1]
                correct += pred.eq(multimodal_label.data).cpu().sum()
                cnt += multimodal_label.numel()
                
                multimodal_label = multimodal_label.cpu().data
                pred = pred.cpu().data
                for each_label in range(num_classes):
                    each_label_index = multimodal_label == each_label
                    each_class_sum[each_label] += each_label_index.sum()
                    each_class_correct[each_label] += (pred[each_label_index] == each_label).sum()
        
        else:
            for step, (image, input_ids, multimodal_label) in enumerate(tqdm(test_loader, desc="Iteration")):
                image, input_ids, multimodal_label = Variable(image.to(device)), Variable(input_ids.to(device)), Variable(multimodal_label.to(device))
                output = model(image, input_ids)
                pred = output.data.max(1)[1]
                # print(pred)
                correct += pred.eq(multimodal_label.data).cpu().sum()
                cnt += multimodal_label.numel()
                
                multimodal_label = multimodal_label.cpu().data
                pred = pred.cpu().data
                for each_label in range(num_classes):
                    each_label_index = multimodal_label == each_label
                    each_class_sum[each_label] += each_label_index.sum()
                    each_class_correct[each_label] += (pred[each_label_index] == each_label).sum()
    
    correct= correct.item()

    model.train()
    return correct/cnt, each_class_correct, each_class_sum

def save_data(data, labels, path, type='train'):
    with open(os.path.join(path, type+'.txt'), 'w') as ofile:
        for text, label in zip(data, labels):
            ofile.write('{} {}\n'.format(label, ' '.join(text)))

def main(args):

    utils.setup_seed(2022)

    if not os.path.exists(os.path.dirname(args.save_path)):
        os.makedirs((os.path.dirname(args.save_path)))

    ## Configure the GPU
    device = torch.device('cuda', args.device_id)

    # load_embedding_dict_info
    if args.embedding_path != '':
        idx2word, word2idx = utils.load_embedding_dict_info(args.embedding_path)
    else:
        word2idx = None

    # load model
    if args.model == 'albef':
        model = model_loader.load_ALBEF(model_path=args.image_encoder_pretrained_dir,num_classes = args.nclasses)
        is_bert = False
        week_process = True
        is_lavis = True
    elif args.model == 'clip_fusion':
        model = model_loader.load_clip_fusion(model_path='',pretrain_path=args.image_encoder_pretrained_dir,num_classes = args.nclasses,output_dim=args.output_dim)
        is_bert = False
        week_process = True
        is_lavis = True
        is_clip = True
    
        
    # load dataset
    if args.dataset == 'MVSA' or args.dataset == 'MVSA_Single':
        train_dataset = dataloader.MVSA_Dataset(args.data_dir, mode='train', is_bert=is_bert, max_seq_length=args.max_seq_length, bert_dirname = args.bert_model_path, word2id=word2idx)
        test_dataset = dataloader.MVSA_Dataset(args.data_dir, mode='test', is_bert=is_bert, max_seq_length=args.max_seq_length, bert_dirname = args.bert_model_path, word2id=word2idx)
    elif args.dataset == 'Hateful_Memes_Challenge':
        train_dataset = dataloader.Hateful_Memes_Dataset(args.data_dir, mode='train', is_bert=is_bert, max_seq_length=args.max_seq_length, bert_dirname = args.bert_model_path, word2id=word2idx)
        test_dataset = dataloader.Hateful_Memes_Dataset(args.data_dir, mode='test', is_bert=is_bert, max_seq_length=args.max_seq_length, bert_dirname = args.bert_model_path, word2id=word2idx)
    elif args.dataset == 'CrisisMMD_v2.0':
        train_dataset = dataloader.CrisisMMD_Dataset(args.data_dir, mode='train', is_bert=is_bert, max_seq_length=args.max_seq_length, bert_dirname = args.bert_model_path, word2id=word2idx)
        test_dataset = dataloader.CrisisMMD_Dataset(args.data_dir, mode='test', is_bert=is_bert, max_seq_length=args.max_seq_length, bert_dirname = args.bert_model_path, word2id=word2idx)
    elif args.dataset == 'MVSA_Single_AdvTrain':
        train_dataset = dataloader.MVSA_Dataset(args.train_data_dir, args.test_data_dir,mode='train', is_bert=is_bert, max_seq_length=args.max_seq_length, bert_dirname = args.bert_model_path, word2id=word2idx)
        test_dataset = dataloader.MVSA_Dataset(args.train_data_dir, args.test_data_dir, mode='test', is_bert=is_bert, max_seq_length=args.max_seq_length, bert_dirname = args.bert_model_path, word2id=word2idx)

    train_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle=False)

    # load optimizer
    need_grad = lambda x: x.requires_grad
    optimizer = optim.Adam(
        filter(need_grad, model.parameters()),
        lr = args.lr
    )
    
    criterion = nn.CrossEntropyLoss()
    # criterion = FocalLoss(args.nclasses)

    best_test = args.best_test
    for epoch in range(args.max_epoch):
        best_test = train_model(epoch, model, optimizer, criterion,
            train_loader,
            test_loader,
            args.nclasses, best_test, args.save_path, device, is_bert)
        if args.lr_decay>0:
            optimizer.param_groups[0]['lr'] *= args.lr_decay

    sys.stdout.write("test_acc: {:.6f}\n".format(
        best_test
    ))

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(sys.argv[0], conflict_handler='resolve')
    argparser.add_argument("--model", type=str, default="cnn", help="which model")
    argparser.add_argument("--dataset", type=str, default="sst", help="which dataset")
    argparser.add_argument("--data_dir", type=str, default="./data/train_dataset", help="The dirname to fold dataset")
    argparser.add_argument("--embedding_path", type=str, default="", help="word vectors")
    argparser.add_argument("--nclasses", type=int, default=2, help='The number of class')
    argparser.add_argument("--batch_size", "--batch", type=int, default=32)
    argparser.add_argument("--max_epoch", type=int, default=70)
    argparser.add_argument("--d", type=int, default=150)
    argparser.add_argument("--dropout", type=float, default=0.3)
    argparser.add_argument("--depth", type=int, default=1)
    argparser.add_argument("--lr", type=float, default=0.001)
    argparser.add_argument("--lr_decay", type=float, default=0)
    argparser.add_argument("--cv", type=int, default=0)
    argparser.add_argument("--save_path", type=str, default='./checkpoint')
    argparser.add_argument("--save_data_split", action='store_true', help="whether to save train/test split")
    argparser.add_argument("--max_seq_length", type=int, default=256)
    argparser.add_argument("--image_encoder_pretrained_dir", type=str, default='', help="The dirname to save the pretrained vgg or resnet model")
    argparser.add_argument("--best_test", type=float, default=0.)
    argparser.add_argument("--output_dim", type=int, default=768)
    argparser.add_argument("--bert_model_name", type=str, default='bert-base-uncased', help="The name of bert model")
    argparser.add_argument("--bert_model_path", type=str, default='', help="The dirname where the pretrained bert model is located")
    argparser.add_argument("--device_id", type=str, default='0')

    argparser.add_argument('--log_save_dir',
                        default='',
                        type=str,
                        help="The oss dirname to save log")
    

    args = argparser.parse_args()
    # args.save_path = os.path.join(args.save_path, args.dataset)
    print (args)
    # torch.cuda.set_device(args.gpu_id)
    # os.environ["CUDA_VISIBLE_DEVICES"]= str(args.gpu_id)
    main(args)
    