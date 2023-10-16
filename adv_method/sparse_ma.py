import numpy as np

import torch
import os
import pickle
import utils
from transformers import BertTokenizer
import cv2
from torch.autograd import Variable
import random

class Content_to_Replace(object):
    def __init__(self, modify_flag = 0,text_pos = None, image_pos = None, data = None, score = 0):
        '''
            modify_flag: [0 : synonym substitution, 1: mosaic]
        '''
        self.modify_flag = modify_flag
        self.text_pos = text_pos
        self.image_pos = image_pos
        self.data = data
        self.score = score
    
    def modify_benign_sample(self, adv_image, adv_text):
        if self.modify_flag == 0:
            adv_text[self.text_pos] = self.data
        elif self.modify_flag == 1:
            adv_image[:, self.image_pos[0]: self.image_pos[0] + self.data.shape[1], self.image_pos[1]: self.image_pos[1] + self.data.shape[2]] = self.data
        return adv_image, adv_text
    
    def judge_is_modify(self, ori_text):
        if self.modify_flag == 0 and ori_text[self.text_pos] == self.data:
            return False
        else:
            return True
    
    def change_back(self, ori_image, ori_text, adv_image, adv_text):
        adv_image, adv_text = adv_image.clone(), adv_text.copy()
        if self.modify_flag == 0:
            adv_text[self.text_pos] = ori_text[self.text_pos]
        elif self.modify_flag == 1:
            adv_image[:, self.image_pos[0]: self.image_pos[0] + self.data.shape[1], self.image_pos[1]: self.image_pos[1] + self.data.shape[2]] = ori_image[:, self.image_pos[0]: self.image_pos[0] + self.data.shape[1], self.image_pos[1]: self.image_pos[1] + self.data.shape[2]]
        return adv_image, adv_text


class SparseMA(object):
    def __init__(self, model, device, synonym_pick_way = 'embedding', synonym_num = 50, synonym_embedding_path='', synonym_cos_path = '', embedding_path = '', bert_dirname = '', max_seq_length = 64, batch_size = 32, is_bert = False, patch_side = 20, use_path = './aux_files',model_type=0):
        self.model = model
        self.device = device
        self.model_type = model_type
 
        self.synonym_dict = {}
        self.synonym_pick_way = synonym_pick_way
        self.synonym_num = synonym_num
        self.synonym_embedding_path = synonym_embedding_path
        self.synonym_cos_path = synonym_cos_path
        
        self.embedding_path = embedding_path
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.is_bert = is_bert

        self.patch_side = patch_side
        
        
        # Prepare the synonym selection
        self.synonym_idx2word, self.synonym_word2idx = utils.load_embedding_dict_info(self.synonym_embedding_path)
        self.cos_sim = utils.load_cos_sim_matrix(self.synonym_cos_path)
        
        # Prepare the embedding for classifier
        self.idx2word, self.word2idx = utils.load_embedding_dict_info(self.embedding_path)

        if is_bert:
            self.tokenizer = BertTokenizer.from_pretrained(bert_dirname, do_lower_case=True)
        
        self.use = utils.USE(use_path)
    
    def attack(self, ori_image, ori_text, ori_label):
        utils.setup_seed(2022)
        log ={}
        query_number = 0
        ori_probs, prediction = self.query_model(ori_image, ori_text)
        if prediction != ori_label:
            log['classification'] = False
            return log
        
        ori_prob = ori_probs[ori_label]
        log['classification'] = True

        adv_image = ori_image.clone()
        adv_text = ori_text.copy()
        all_content_candidates, new_query = self.position_order_decision(ori_image, ori_text, ori_label)

        success = False
        adv_label = ori_label
        query_number += new_query
        replace_content_candidates = []
        image_perturb_unit = (self.patch_side * self.patch_side) / (ori_image.shape[1] * ori_image.shape[2])
        for content in all_content_candidates:
            probs = None
            if content.modify_flag == 0:
                pos = content.text_pos
                w = content.data
                if w in self.synonym_dict.keys():
                    candidates = self.synonym_dict[w]
                else:
                    candidates = self.replace_with_synonym_function(w)
                    self.synonym_dict[w] = candidates

                candidate_word, max_score, new_query,probs = self.candidate_word_saliency_decision(adv_image, adv_text, pos, candidates, ori_label, ori_prob)
                query_number += new_query
                adv_text[pos] = candidate_word
            elif content.modify_flag == 1:
                height, width = content.image_pos
                all_image_list = []
                # white patch
                white_patch_img = self.color_mosaic(adv_image, height, width, self.patch_side, self.patch_side, 1)
                # black patch
                black_patch_img = self.color_mosaic(adv_image, height, width, self.patch_side, self.patch_side, 0)
                # white black patch
                white_black_mosaic_img = self.white_black_mosaic(adv_image, height, width, self.patch_side, self.patch_side)
                # all_image_list.append(mosaic_img)
                all_image_list.append(white_patch_img)
                all_image_list.append(black_patch_img)
                all_image_list.append(white_black_mosaic_img)
                logits_new, labels = self.query_model_group(all_image_list, [adv_text for _ in range(len(all_image_list))])
                query_number += len(logits_new)
                probs = logits_new.copy()
                logits_new = logits_new[:, ori_label]
                
                # consider the model output
                score = ori_prob - logits_new
                
                # consider the perturb unit
                score = score / image_perturb_unit
                
                max_score = np.max(score)
                candidate_image = all_image_list[np.argmax(score)]
                adv_image[:, height : height + self.patch_side, width: width + self.patch_side] = candidate_image[:, height : height + self.patch_side, width: width + self.patch_side]
                probs = probs[np.argmax(score)]
            
            replace_content_candidates.append(content)

            if probs is not None:
                prediction = np.argmax(probs,axis=-1)
            else :
                continue
            if int(prediction) != int(ori_label):
                # prediction alone
                probs, prediction = self.query_model(adv_image, adv_text)
                query_number += 1
                if int(prediction) != int(ori_label):
                    success = True
                    adv_label = prediction
                    break

        log['status'] = success
        log['query_number'] = query_number
        if not success:
            return log
        
        # import pdb; pdb.set_trace()
        adv_image, adv_text, new_query = self.exchange_back(ori_image, ori_text, adv_image, adv_text, replace_content_candidates, ori_label)
        query_number += new_query
        
        log['adv_image'] = adv_image
        log['image_perturbation_rate'] = ((adv_image != ori_image).sum()/(ori_image.shape[0] *ori_image.shape[1] * ori_image.shape[2])).item()
        log['adv_text'] = ' '.join(adv_text)
        if len(adv_text) == 0:
            log['text_perturbation_rate'] = 0.0
        else:
            log['text_perturbation_rate'] = self.check_diff(ori_text, adv_text)/len(adv_text)
        log['adv_label'] = adv_label
        
        adv_image_sim = utils.ssim(ori_image, adv_image)
        adv_text_sim = self.use.semantic_sim([' '.join(ori_text)], [' '.join(adv_text)])[0][0]
        log['adv_image_sim'] = adv_image_sim
        log['adv_text_sim'] = adv_text_sim
        
        return log

    def position_order_decision(self, ori_image, ori_text, ori_label):
        query_number = 0
        probs, label = self.query_model(ori_image, ori_text)
        ori_prob = probs[ori_label]
        all_content_candidates = []
        all_scores = []
        
        image_perturb_unit = (self.patch_side * self.patch_side) / (ori_image.shape[1] * ori_image.shape[2])
        
        # decide the importance of each region in image
        candidates_images = []
        _, image_height, image_width = ori_image.shape
        for h in range(0, image_height, self.patch_side):
            for w in range(0, image_width, self.patch_side):
                unk_image = self.color_mosaic(ori_image, h, w, self.patch_side, self.patch_side, 0)
                candidates_images.append(unk_image)
                
                content = Content_to_Replace(modify_flag=1, image_pos=(h,w), data=unk_image[:, h:h+self.patch_side, w:w+self.patch_side])
                all_content_candidates.append(content)

        logits_news, labels = self.query_model_group(candidates_images, [ori_text for _ in range(len(candidates_images))])
        logits_news = logits_news[:, ori_label]
        query_number += len(labels)

        # consider the model output
        scores = ori_prob - logits_news

        # consider the perturb unit
        scores = scores / image_perturb_unit   
        all_scores.extend(list(scores))
        
        
        if len(ori_text) == 0:
            indexes = np.argsort(np.array(all_scores))[::-1]
            all_content_candidates = np.array(all_content_candidates)[indexes]
            return all_content_candidates, query_number
        

        # decide the importance of each token in text
        candidates_texts = []
        for i, w in enumerate(ori_text):
            unk_text = ori_text.copy()
            if self.model_type >0:
                unk_text[i] = '[UNK]'
            elif self.is_bert:
                unk_text[i] = '[UNK]'
            else:
                unk_text[i] = '<oov>'
            
            candidates_texts.append(unk_text)

            # all_scores.append(score)
            content = Content_to_Replace(modify_flag=0, text_pos=i, data=ori_text[i])
            all_content_candidates.append(content)
        
        logits_news, labels = self.query_model_group([ori_image for _ in range(len(candidates_texts))], candidates_texts)
        logits_news = logits_news[:, ori_label]
        query_number += len(labels)

        # consider the model output
        scores = ori_prob - logits_news

        # consider the perturb unit
        scores = scores / (1/len(ori_text))
        
        all_scores.extend(list(scores))
        
        indexes = np.argsort(np.array(all_scores))[::-1]
        all_content_candidates = np.array(all_content_candidates)[indexes]

        return all_content_candidates, query_number



    def candidate_word_saliency_decision(self, ori_image, ori_text, idx, candidates, ori_label, ori_prob):
        text_perturb_unit = 1/len(ori_text)
            
        query_number = 0
        max_score = -100
        # min_prob = 1
        candidate_word = ori_text[idx]
        query_number += 1

        sentence_new_list = []
        for c in candidates:
            clean_tokens_new = list(ori_text)
            clean_tokens_new[idx] = c
            sentence_new_list.append(clean_tokens_new.copy())
        
        new_logits = None
        if len(sentence_new_list) != 0:
            logits_new, labels = self.query_model_group([ori_image for _ in range(len(sentence_new_list))], sentence_new_list)
            query_number += len(logits_new)
            new_logits = logits_new
            logits_new = logits_new[:, ori_label]
            
            # consider the model output
            score = ori_prob - logits_new

            # consider the perturb unit
            score = score / text_perturb_unit
            
            max_score = np.max(score)
            # max_diff = np.max(diff)
            candidate_word = candidates[np.argmax(score)]
            new_logits = new_logits[np.argmax(score)]
            # candidate_word = candidates[np.argmax(diff)]
        return candidate_word, max_score, query_number,new_logits
    

    def exchange_back(self, ori_image, ori_text, adv_image, adv_text, replace_content_candidates, ori_label):
        query_number = 0
        change_flag = list(range(len(replace_content_candidates)))
        # import pdb; pdb.set_trace()
        for i in range(4 *len(replace_content_candidates)):
            perturb_len = len(change_flag)
            if perturb_len == 1:
                break
            
            extract_index = random.choice(change_flag)
            content = replace_content_candidates[extract_index]
            change_adv_image, change_adv_text = content.change_back(ori_image, ori_text, adv_image, adv_text)
            
            probs, prediction = self.query_model(change_adv_image, change_adv_text)
            query_number += 1
            if prediction != ori_label:
                adv_image = change_adv_image
                adv_text = change_adv_text
                
                change_flag.remove(extract_index)

        return adv_image, adv_text, query_number


    def replace_with_synonym_function(self, word):
        # candidate_word_list,_ = utils.replace_with_synonym(word, 'nltk')
        candidate_word_list,_ = utils.replace_with_synonym(word, self.synonym_pick_way, idx2word= self.synonym_idx2word, word2idx=self.synonym_word2idx, cos_sim= self.cos_sim)
        candidate_word_list = candidate_word_list[:self.synonym_num+1]
        return candidate_word_list


    def query_model(self, image, text):
        if self.model_type > 0:
            input_ids = ' '.join(text)
            image = utils.norm_transforms(image)
            image = Variable(image.unsqueeze(0).to(self.device))
            samples = {"image": image, "text_input": input_ids,'label':0}
            if self.model_type==1: # albef
                output = self.model.predict(samples)
                probs = output['predictions'].data.cpu().numpy()
                
            elif self.model_type == 2: # clip_fusion
                output = self.model(samples)
                probs = output.data.cpu().numpy()
            probs = self._softmax(probs)
            label = np.argmax(probs, axis=-1)[0]
        elif self.is_bert:
            input_ids, input_mask, segment_ids = utils.convert_example_to_feature_for_bert(' '.join(text), self.max_seq_length, self.tokenizer)
            input_ids, input_mask, segment_ids = torch.tensor(input_ids), torch.tensor(input_mask), torch.tensor(segment_ids)
            input_ids, input_mask, segment_ids = Variable(input_ids.unsqueeze(0).to(self.device)), Variable(input_mask.unsqueeze(0).to(self.device)), Variable(segment_ids.unsqueeze(0).to(self.device))
            image = Variable(image.unsqueeze(0).to(self.device))
            output = self.model(image, input_ids, input_mask, segment_ids).data.cpu().numpy()
            # probs = self._softmax(output)
            probs = output
            label = np.argmax(probs, axis=-1)[0]
        else:
            input_ids, input_mask = utils.convert_example_to_feature_for_cnn(text, self.word2idx, self.max_seq_length)
            input_ids = torch.tensor(input_ids)
            image, input_ids = Variable(image.unsqueeze(0).to(self.device)), Variable(input_ids.unsqueeze(0).to(self.device))
            output = self.model(image, input_ids).data.cpu().numpy()
            # probs = self._softmax(output)
            probs = self._softmax(output)
            label = np.argmax(probs, axis=-1)[0]
        return probs[0], label

    def query_model_group(self, image_list, text_list):
        if self.model_type > 0:
            input_ids_list = []
            for text in text_list:
                input_ids = ' '.join(text)
                input_ids_list.append(input_ids)
            
            outs = []
            nbatch = (len(text_list) -1)// self.batch_size + 1
            for i in range(nbatch):
                image = image_list[i*self.batch_size:(i+1)*self.batch_size]
                input_ids= input_ids_list[i*self.batch_size:(i+1)*self.batch_size]
                image = torch.stack(image)
                image = utils.norm_transforms(image)
                image = Variable(image.to(self.device))
                samples = {"image": image, "text_input": input_ids,'label':0}
                if self.model_type==1: # albef
                    output = self.model.predict(samples)
                    output = output['predictions'].data.cpu().numpy()
                    
                elif self.model_type == 2: # clip_fusion
                    output = self.model(samples)
                    output = output.data.cpu().numpy()
                probs = self._softmax(output)
                # probs = output
                outs.append(probs)
                
            probs = np.vstack(outs)
            label = np.argmax(probs, axis=-1)
        elif self.is_bert:
            input_ids_list, input_mask_list, segment_ids_list  = [], [], []
            for text in text_list:
                input_ids, input_mask, segment_ids = utils.convert_example_to_feature_for_bert(' '.join(text), self.max_seq_length, self.tokenizer)
                input_ids_list.append(input_ids)
                input_mask_list.append(input_mask)
                segment_ids_list.append(segment_ids)
            outs = []
            nbatch = (len(text_list) -1)// self.batch_size + 1
            for i in range(nbatch):
                image = image_list[i*self.batch_size:(i+1)*self.batch_size]
                input_ids= input_ids_list[i*self.batch_size:(i+1)*self.batch_size]
                input_mask= input_mask_list[i*self.batch_size:(i+1)*self.batch_size]
                segment_ids= segment_ids_list[i*self.batch_size:(i+1)*self.batch_size]

                input_ids, input_mask, segment_ids = torch.tensor(input_ids), torch.tensor(input_mask), torch.tensor(segment_ids)
                image = torch.stack(image)
                image = Variable(image.to(self.device))
                input_ids, input_mask, segment_ids = Variable(input_ids.to(self.device)), Variable(input_mask.to(self.device)), Variable(segment_ids.to(self.device))
                
                output = self.model(image, input_ids, input_mask, segment_ids).data.cpu().numpy()
                # probs = self._softmax(output)
                probs = output
                outs.append(probs)

            probs = np.vstack(outs)
            label = np.argmax(probs, axis=-1)
        else:
            input_ids_list = []
            for text in text_list:
                input_ids, input_mask = utils.convert_example_to_feature_for_cnn(text, self.word2idx, self.max_seq_length)
                input_ids_list.append(input_ids)

            outs = []
            nbatch = (len(text_list) -1)// self.batch_size + 1
            for i in range(nbatch):
                image = image_list[i*self.batch_size:(i+1)*self.batch_size]
                input_ids= input_ids_list[i*self.batch_size:(i+1)*self.batch_size]
                input_ids = torch.tensor(input_ids)
                image = torch.stack(image)
                image = Variable(image.to(self.device))
                input_ids = Variable(input_ids.to(self.device))
                
                output = self.model(image, input_ids).data.cpu().numpy()
                # probs = self._softmax(output)
                probs = self._softmax(output)
                outs.append(probs)
                
            probs = np.vstack(outs)
            label = np.argmax(probs, axis=-1)

        return probs, label


    def color_mosaic(self, img, start_x, start_y, mask_w, mask_h, color):  
        image = img.clone()
        for i in range(0, min(mask_h, 224 - start_x), 1):
            for j in range(0, min(mask_w, 224 - start_y), 1):
                image[:, start_x +i : start_x +i + 1, start_y+j:start_y + j + 1] = color

        return image
    

    def white_black_mosaic(self, img, start_x, start_y, mask_w, mask_h):  
        image = img.clone()
        pixel = 1
        fill_pixel = [torch.tensor([216/255, 216/255, 216/255]), torch.tensor([165/255, 165/255, 165/255])]
        for i in range(0, min(mask_h, 224 - start_x)):
            row_pixel = pixel
            for j in range(0, min(mask_w, 224 - start_y)):
                image[:, start_x +i : start_x +i + 1, start_y+j:start_y + j + 1] = max(row_pixel, 0)
                row_pixel *= -1
            pixel *= -1

        return image

    def _softmax(self, x):
        orig_shape = x.shape
        if len(x.shape) > 1:
            _c_matrix = np.max(x, axis=1)
            _c_matrix = np.reshape(_c_matrix, [_c_matrix.shape[0], 1])
            _diff = np.exp(x - _c_matrix)
            x = _diff / np.reshape(np.sum(_diff, axis=1), [_c_matrix.shape[0], 1])
        else:
            _c = np.max(x)
            _diff = np.exp(x - _c)
            x = _diff / np.sum(_diff)
        assert x.shape == orig_shape
        return x


    def check_diff(self, sentence, perturbed_sentence):
        words = sentence
        perturbed_words = perturbed_sentence
        diff_count = 0
        if len(words) != len(perturbed_words):
            raise RuntimeError("Length changed after attack.")
        for i in range(len(words)):
            if words[i] != perturbed_words[i]:
                diff_count += 1
        return diff_count
    

    