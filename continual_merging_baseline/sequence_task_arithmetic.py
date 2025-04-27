import os

import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import time
import tqdm
import sys
sys.path.append('/home/pzc/subspace_fusion/')
import torch
from task_vectors import TaskVector
from eval import eval_single_dataset, eval_single_dataset_head, eval_single_dataset_preprocess_head
from args import parse_arguments
from src.datasets.registry import get_dataset
from src.datasets.common import get_dataloader, maybe_dictionarize, get_dataloader_shuffle


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

args = parse_arguments()
#name_list=args.name_list
#获取task_num列表
name_list=[[1], [2], [3], [4], [5], [6], [7], [8]]
model_name = 'ViT-B-32'
source_root_path = '/home/pzc/subspace_fusion/'
args = parse_arguments()
args.data_location = source_root_path+'data/'
args.model_name = model_name
args.save = source_root_path+'cache/checkpoints/task_vectors_checkpoints/'+model_name
args.logs_path = source_root_path + 'logs/' + model_name
pretrained_checkpoint = source_root_path +'cache/checkpoints/task_vectors_checkpoints/'+ model_name + '/zeroshot.pt'

str_time_ = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
log = create_log_dir(args.logs_path, 'log_{}_Task_Arithmetic.txt'.format(str_time_))
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
exam_datasets = get_datasets(name_list)
best_vector = []

for step in range(len(exam_datasets)):
    # 创建当前任务向量列表
    task_vectors = [
        TaskVector(pretrained_checkpoint, source_root_path + 'cache/checkpoints/task_vectors_checkpoints/' + model_name + '/' + dataset_name + '/finetuned.pt')
        for dataset_name in exam_datasets[step]
    ]
    if best_vector:
        task_vectors.append(best_vector[-1])  # 添加上一轮的结果

    task_vector_sum = sum(task_vectors)
    print(f"Type of task_vector_sum: {type(task_vector_sum)}")  # 添加这一行

    scaling_coef_ = 0.3

    # 应用任务向量到预训练模型，得到当前轮的 image_encoder
    image_encoder = task_vector_sum.apply_to(pretrained_checkpoint, scaling_coef=scaling_coef_)

    log.info('*' * 20 + ' scaling_coef: ' + str(scaling_coef_) + ' ' + '*' * 20)

    accs = []
    for dataset in exam_datasets:
        metrics = eval_single_dataset(image_encoder, dataset[0], args)
        log.info(str(dataset) + ': ' + str(metrics.get('top1') * 100) + '%')
        accs.append(metrics.get('top1') * 100)

    avg_acc = np.mean(accs)
    log.info('Avg ACC: ' + str(avg_acc) + '%')
    
    # 将当前 image_encoder 添加到 best_vector 以供下一轮使用
    best_vector.append(image_encoder)
# 在结束后，best_vector 将包含所有轮次的最佳模型
