import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import time
import tqdm
import sys
sys.path.append('/home/pzc/subspace_fusion/')
import torch
from eval import eval_single_dataset
from args import parse_arguments


def create_log_dir(path, filename='log.txt'):
    import logging
    if not os.path.exists(path):
        os.makedirs(path)
    logger = logging.getLogger(path)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(path+'/'+filename)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

# exam_datasets = ['SUN397', 'Cars', 'RESISC45', 'EuroSAT', 'SVHN', 'GTSRB', 'MNIST', 'DTD'] # SUN397 | Cars | RESISC45 | EuroSAT | SVHN | GTSRB | MNIST | DTD
args = parse_arguments()
#获取task_num列表
name_list=[[1], [2], [3], [4], [5], [6], [7], [8]]
model_name = 'ViT-L-14'
source_root_path = '/home/pzc/subspace_fusion/'
args = parse_arguments()
args.data_location = source_root_path+'data/'
args.model_name = model_name
args.model = model_name
args.save = source_root_path+'cache/checkpoints/task_vectors_checkpoints/'+model_name
args.logs_path = source_root_path + 'logs/' + model_name
pretrained_checkpoint = source_root_path +'cache/checkpoints/task_vectors_checkpoints/'+ model_name + '/zeroshot.pt'

str_time_ = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
log = create_log_dir(args.logs_path, 'log_{}_ties_merging.txt'.format(str_time_))
args.log = log

def get_datasets(name_list):
    datasets_dic = {'SUN397': 1, 'Cars': 2, 'RESISC45': 3, 'EuroSAT': 4, 
                     'SVHN': 5, 'GTSRB': 6, 'MNIST': 7, 'DTD': 8}
    datasets = []
    for sublist in name_list:
        converted_sublist = [key for index in sublist 
                             for key, value in datasets_dic.items() if value == index]
        datasets.append(converted_sublist)
    return datasets
exam_datasets=get_datasets(name_list)

from ties_merging_utils import *

#load pretrain model
ptm_check = torch.load(pretrained_checkpoint).state_dict()
ptm_check = torch.load(pretrained_checkpoint).state_dict()
best_tv_state_dict=[]
#sequence merging
for step in range(len(exam_datasets)):
    #load current step ft model
    ft_checks = [torch.load(source_root_path + 'cache/checkpoints/task_vectors_checkpoints/'+ model_name + '/' + dataset_name + '/finetuned.pt').state_dict() for dataset_name in exam_datasets[step]]
    if best_tv_state_dict:
        ft_checks.append(best_tv_state_dict[-1])
    check_parameterNamesMatch(ft_checks + [ptm_check])

    remove_keys = []
    print(f"Flattening out Checkpoints")
    flat_ft = torch.vstack([state_dict_to_vector(check, remove_keys) for check in ft_checks])
    flat_ptm = state_dict_to_vector(ptm_check, remove_keys)

    tv_flat_checks = flat_ft - flat_ptm
    assert check_state_dicts_equal(vector_to_state_dict(flat_ptm, ptm_check, remove_keys), ptm_check)
    assert all([check_state_dicts_equal(vector_to_state_dict(flat_ft[i], ptm_check, remove_keys), ft_checks[i])for i in range(len(ft_checks))])


    K = 20
    merge_func = "dis-sum"
    scaling_coef_ = 0.3

    merged_tv = ties_merging(tv_flat_checks, reset_thresh=K, merge_func=merge_func,)
    merged_check = flat_ptm + scaling_coef_ * merged_tv
    merged_state_dict = vector_to_state_dict(merged_check, ptm_check, remove_keys=remove_keys)

    #add the last tv's state_dict
    best_tv_state_dict.append(merged_state_dict)

    image_encoder = torch.load(pretrained_checkpoint)
    image_encoder.load_state_dict(merged_state_dict, strict=False)

    Total_ACC = 0.
    for dataset in exam_datasets:
        metrics = eval_single_dataset(image_encoder, dataset[0], args)
        Total_ACC += metrics['top1']
        log.info(str(dataset) + ':' + str(metrics))

    log.info('Final: ' + 'Avg ACC:' + str(Total_ACC / len(exam_datasets)))
