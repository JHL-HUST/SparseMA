import os
import json
import datetime
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
plt.switch_backend('agg')

import torch
from torch.utils.data import DataLoader

from config import *
import utils as util
import pickle
import datetime
import time
import language_tool_python
import torchvision.transforms as transforms

class Attacker(object):

    def __init__(self, model, config, attack_method):
        self.model = model
        self.config = config
        self.attack_method = attack_method

        self.attack_name = self.config.CONFIG['attack_name']
        # self.tool = language_tool_python.LanguageTool('en-US')
        
        
        self.save_sample = False
        self.sample_log_path = config.Common['sample_log_path']
        if self.sample_log_path != '':
            if not os.path.exists(self.sample_log_path):
                os.makedirs(self.sample_log_path)
            else:
                os.system('rm -rf {}'.format(self.sample_log_path))
                os.makedirs(self.sample_log_path)
            self.save_sample = True


    def start_attack(self, dataloader):
        attack_switch = self.config.Switch_Method['method']
        log = {}
        if attack_switch == 'One_Sample_Attack':
            index = getattr(self.config, attack_switch)['index']
            for i,(image, text, label) in enumerate(dataloader):
                if (i == index):
                    log = self.one_sample_attack(image, text, label)
                    break
        elif attack_switch == 'Batch_Sample_Attack':
            log = self.batch_sample_attack(dataloader, **getattr(self.config, attack_switch))

        return log



    def one_sample_attack(self, image, text, label):
        log = {}
        attack_log = self.attack_method.attack(image, text, label)
        log['pre_image_data'] = image
        log['pre_text_data'] = ' '.join(text)
        log['pre_label'] = label
        log.update(attack_log)
        return log

    def batch_sample_attack(self, data_loader, batch):  
        log = {}
        # use = util.USE('./data/aux_files')

        ## Record the attack performance
        success = 0                             # The sample number of successful attacks
        classify_true = 0                       # The sample number of successfully classified after attacks
        sample_num = 0                          # The total number of samples

        query_number_list = []                  # The query number of each attack for target model
        image_perturbation_rate_list = []             # The perturbation rate of the adversarial example after each attack
        text_perturbation_rate_list = []             # The perturbation rate of the adversarial example after each attack
        process_time_list = []                  # The processing time for each attack
        
        perturb_image_only = 0                   # The number of adversarial examples which only perturb image
        perturb_text_only = 0                   # The number of adversarial examples which only perturb text
        perturb_image_and_text = 0                # The number of adversarial examples which only perturb image and text
        
        image_sim_list = []
        text_sim_list = []


        for i in range(len(data_loader)):
            text, image, label = data_loader.__getitem__(i)
            # if i == batch:
            #     break
            
            starttime = datetime.datetime.now()
            one_log = self.one_sample_attack(image, text, label)
            endtime = datetime.datetime.now()
            process_time =  (endtime - starttime).seconds
            process_time_list.append(process_time)


            if not one_log['classification']:
                message = 'The {:3}-th sample is not correctly classified'.format(i)
                log['print_{}'.format(i)] = message
                print(message)
                
                continue

            sample_num += 1

            query_number = one_log['query_number']
            query_number_list.append(query_number)
            


            if(one_log['status']):
                success += 1
                
                ## Record the perturbation rate
                image_perturbation_rate = one_log['image_perturbation_rate']
                image_perturbation_rate_list.append(image_perturbation_rate)

                text_perturbation_rate = one_log['text_perturbation_rate']
                text_perturbation_rate_list.append(text_perturbation_rate)
                
                if image_perturbation_rate != 0 and text_perturbation_rate == 0:
                    perturb_image_only += 1
                elif image_perturbation_rate == 0 and text_perturbation_rate != 0:
                    perturb_text_only += 1
                else:
                    perturb_image_and_text += 1
                
                adv_image_sim = one_log['adv_image_sim']
                adv_text_sim = one_log['adv_text_sim']
                image_sim_list.append(adv_image_sim)
                text_sim_list.append(adv_text_sim)
                
                
                if self.save_sample:
                    adv_image = transforms.ToPILImage()(one_log['adv_image'])
                    ori_image = transforms.ToPILImage()(image)
                    adv_image.save(os.path.join(self.sample_log_path, '{}_adv.png'.format(i)))
                    ori_image.save(os.path.join(self.sample_log_path, '{}_ori.png'.format(i)))
                    adv_text = one_log['adv_text']
                    ori_text = ' '.join(text)
                    adv_label = one_log['adv_label']
                    with open(os.path.join(self.sample_log_path, '{}_text.txt'.format(i)), 'w') as f:
                        f.write('ori_text:\t{}\n'.format(ori_text))
                        f.write('adv_text:\t{}\n'.format(adv_text))
                        f.write('ori_label:\t{}\n'.format(str(label)))
                        f.write('adv_label:\t{}\n'.format(str(adv_label)))
                        f.write('\n')
                        f.write('image_perturbation_rate:\t{}\n'.format(str(image_perturbation_rate)))
                        f.write('text_perturbation_rate:\t{}\n'.format(str(text_perturbation_rate)))
                        f.write('\n')
                        f.write('adv_image_sim:\t{}\n'.format(str(adv_image_sim)))
                        f.write('adv_text_sim:\t{}\n'.format(str(adv_text_sim)))
                

                message = 'The {:3}-th sample takes {:3}s, with the image perturbation rate: {:.5}, text perturbation rate {:.5}, semantic similarity: {:.5}, query number: {:4}. Attack succeeds.'.format(i, process_time, image_perturbation_rate, text_perturbation_rate, 0.0, query_number)
                print(message)
            else:
                classify_true += 1
                message = 'The {:3}-th sample takes {:3}s, Attack fails'.format(i, process_time)
                print(message)
            
            log['print_{}'.format(i)] = message
        
        message = '\nA total of {:4} samples were selected, {:3} samples were correctly classified, {:3} samples were attacked successfully and {:4} samples failed'.format(batch, sample_num, success, sample_num - success)
        print(message)
        log['print_last'] = message


        acc = sample_num/batch                                                      # The classification accuracy of target model
        attack_acc = classify_true/batch              # The classification accuracy of target model after attack
        success_rate = success/sample_num                                           # The attack success rate of attack method
        average_image_perturbation_rate = np.mean(image_perturbation_rate_list).item()          # The average perturbation rate of the adversarial example
        average_text_perturbation_rate = np.mean(text_perturbation_rate_list).item()          # The average perturbation rate of the adversarial example
        # average_sim = np.mean(sim_list).item()                                      # The average semantic similarity of the adversarial example
        average_image_sim = np.mean(image_sim_list).item()
        average_text_sim = np.mean(text_sim_list).item()

        average_query_number = np.mean(query_number_list).item()                    # The average query number of each attack
        average_process_time = np.mean(process_time_list).item()                    # The average process time of each attack
        

        log['acc'] = acc
        log['after_attack_acc'] = attack_acc
        log['attack_success_rate'] = success_rate
        log['average_image_perturbation_rate'] = average_image_perturbation_rate
        log['average_text_perturbation_rate'] = average_text_perturbation_rate
        
        log['perturb_image_only'] = perturb_image_only/max(1, success)
        log['perturb_text_only'] = perturb_text_only/max(1, success)
        log['perturb_image_and_text'] = perturb_image_and_text/max(1, success)
        
        log['adv_image_sim'] = average_image_sim
        log['adv_text_sim'] = average_text_sim

        log['average_query_number'] = average_query_number
        log['average_process_time'] = average_process_time
        
        return log

        