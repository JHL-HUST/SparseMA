import time
class Config(object):
    def __init__(self):

        ## Main parameters
        self.CONFIG = dict(
            dataset_name = 'MVSA_AdvDataset',
            model_name = 'clip_fusion',
            attack_name = 'SparseMA',
        )

        ## Model params
        self.ALBEF = dict(
            model_path = './data/model/albef/CrisisMMD_v2.0',
            num_classes = 7,
            # output_dim: 256,            
        ),

        self.clip_fusion = dict(
            model_path= './data/model/clip_fusion/CrisisMMD_v2.0_rn50_3',
            pretrain_path= './data/pretrain/RN50.pt',
            num_classes= 7,
            output_dim= 1024,            
        ),
        

        ## Dataset params
        self.MVSA_AdvDataset = dict(
            data_dir = './data/train_dataset/MVSA_Single',
            mode = 'test',
            is_bert = False,
            max_seq_length = 64,
            bert_dirname = '',
        )

        self.CrisisMMD_AdvDataset = dict(
            data_dir = './data/dataset/CrisisMMD_v2.0',
            mode = 'test',
            is_bert = True,
            max_seq_length = 256,
            bert_dirname = './data/pretrain/bert-base',
        )

        ## Hyper-parameters of attack method params
        self.SparseMA = dict(
            synonym_pick_way = 'embedding',
            synonym_num = 4,                            # Synonym number
            synonym_embedding_path = './data/aux_files/counter-fitted-vectors.txt',       # The path to the counter-fitting embeddings we used to find synonyms
            synonym_cos_path =  './data/aux_files/mat.txt',            # The pre-compute the cosine similarity scores based on the counter-fitting embeddings
            embedding_path = './data/embedding/glove.6B.200d.txt',
            bert_dirname = "./data/pretrain/bert-base",
            max_seq_length = 64,
            batch_size = 32,
            is_bert = True,
            patch_side = 16,
            use_path = './data/aux_files',
        )

        ## Common params
        self.Common = dict(
            embedding_path = './data/embedding/glove.6B.200d.txt',
            sample_log_path = './data/adv_data/sparse_ma/vgg_cnn_mvsa_single_adv_train_cont',
        )

        ## Log params
        self.Checkpoint = dict(
            log_dir = './log',                          # The log directory where the attack results will be written
            log_filename = '{}_{}_{}.log'.format(self.CONFIG['model_name'], self.CONFIG['attack_name'], time.strftime("%m-%d_%H-%M")),              # The output log filename
        )

        ## Attack setting params
        self.Switch_Method = dict(
            method = 'Batch_Sample_Attack',             # The attack mode: batch attack multiple samples or attack only one sample (choices: 'Batch_Sample_Attack', 'One_Sample_Attack')
        )
        self.Batch_Sample_Attack = dict(
            batch = 1000,                               # The sample number of batch attack
        )
        self.One_Sample_Attack = dict(
            index = 66,                                 # The sample index of attack
        )

    def log_output(self):
        log = {}

        log['CONFIG'] = self.CONFIG
        for name,value in self.CONFIG.items():
            if type(value) is str and hasattr(self,value):
                log[value] = getattr(self,value)
            else:
                log[name] = value
        log['Switch_Method'] = self.Switch_Method['method']
        log['Switch_Method_Value'] = getattr(self, self.Switch_Method['method'])

        return log

    
    def load_parameter(self, parameter):
        for key, value in parameter.items():
            if hasattr(self, key):
                if type(value) is dict:
                    orig_config = getattr(self, key)
                    if orig_config.keys() == value.keys():
                        setattr(self, key, value)
                    else:
                        redundant_key = value.keys() - orig_config.keys()
                        if redundant_key:
                            msg = "there are many redundant keys in config file, e.g.:  " + str(redundant_key)
                            assert None, msg
                        
                        lack_key = orig_config.keys() - value.keys()
                        if lack_key:
                            msg = "there are many lack keys in config file, e.g.:  " + str(lack_key)
                            assert None, msg
                else:
                    setattr(self, key, value)
            else:
                setattr(self, key, value)

        self.Checkpoint['log_filename'] =  '{}_{}_{}.log'.format(self.CONFIG['model_name'], self.CONFIG['attack_name'], time.strftime("%m-%d_%H-%M"))              #log文件名称

        return None
    
    