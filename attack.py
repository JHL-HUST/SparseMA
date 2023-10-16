import adv_method
import model_loader
import user as user_module
import config as config_module
import utils as utils_module
import dataloader as dataloader_module
import os
import argparse
import yaml
import torch

def main():
    utils_module.setup_seed(2022)

    ## Required parameters for target model and hyper-parameter
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', 
                        default=None,
                        help="The config parameter yaml file contains the parameters of the dataset, target model and attack method",
                        type=str)
    parser.add_argument("--device_id", type=int, default=0)

    args = parser.parse_args()


    ## Universal parameters
    config = config_module.Config()

    if args.config:
        assert os.path.exists(args.config), "There's no '" + args.config + "' file."
        with open(args.config, "r") as load_f:
            config_parameter = yaml.load(load_f)
            config.load_parameter(config_parameter)

    ## Save the parameters into log
    Log = config.log_output()

    ## Configure the GPU
    device = torch.device('cuda', args.device_id)

    ## Prepare the dataset
    idx2word, word2idx = utils_module.load_embedding_dict_info(config.Common['embedding_path'])
    adv_dataset = getattr(dataloader_module, config.CONFIG['dataset_name'])(**getattr(config, config.CONFIG['dataset_name']), word2id = word2idx)

    ## Prepare the target model
    model = getattr(model_loader, 'load_' + config.CONFIG['model_name'])(**getattr(config, config.CONFIG['model_name']))
    model.to(device)
    model.eval()

    ## Prepare the attack method
    attack_parameter = getattr(config, config.CONFIG['attack_name'])
    attack_name = config.CONFIG['attack_name']
    attack_method = getattr(adv_method, attack_name)(model, device, **attack_parameter)

    ## Prepare the attacker
    attacker = user_module.Attacker(model, config, attack_method)
    

    ## Start the attack
    log = attacker.start_attack(adv_dataset)
    Log.update(log)

    
    ## Save and print the Log
    print(config.Checkpoint['log_dir'], config.Checkpoint['log_filename'])
    utils_module.ensure_dir(config.Checkpoint['log_dir'])
    filename = os.path.join(config.Checkpoint['log_dir'], config.Checkpoint['log_filename'])
    f = open(filename,'w')

    for key, value in Log.items():
        if 'print' not in key:
            print('    {:15s}: {}'.format(str(key), value))
        log = {}
        log[key] = value
        utils_module.log_write(f, log)



if __name__ == "__main__":
    main()
