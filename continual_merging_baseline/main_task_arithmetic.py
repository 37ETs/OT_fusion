import os

import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import time
import sys
sys.path.append('/Users/dahuahua/PycharmProjects/fusion/AdaMerging/')

from task_vectors import TaskVector
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

exam_datasets = ['SUN397', 'Cars', 'RESISC45', 'EuroSAT', 'SVHN', 'GTSRB', 'MNIST', 'DTD'] # SUN397 | Cars | RESISC45 | EuroSAT | SVHN | GTSRB | MNIST | DTD
args = parse_arguments()
name_list=args.name_list
#获取task_num列表
# name_list=[[1, 2, 3, 4, 5, 6], [7], [8]]
model_name = 'ViT-B-32'
source_root_path = '/Users/dahuahua/PycharmProjects/fusion/AdaMerging/'
args = parse_arguments()
args.data_location = source_root_path+'data/'
args.model = model_name
args.save = source_root_path+'checkpoints/'+ model_name
args.logs_path =  '/Users/dahuahua/PycharmProjects/fusion/AdaMerging/logs/' + model_name
pretrained_checkpoint = source_root_path + 'checkpoints/'+ model_name + '/zeroshot.pt'

str_time_ = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
log = create_log_dir(args.logs_path, 'log_{}_layer_wise_AdaMerging.txt'.format(str_time_))
args.log = log
task_vectors = [
    TaskVector(pretrained_checkpoint, source_root_path + 'checkpoints/' + model_name + '/' + dataset_name + '/finetuned.pt') for dataset_name in exam_datasets
]

task_vector_sum = sum(task_vectors)
print(f"Type of task_vector_sum: {type(task_vector_sum)}")  # 添加这一行

scaling_coef_ = 0.3

image_encoder = task_vector_sum.apply_to(pretrained_checkpoint, scaling_coef=scaling_coef_)
log.info('*'*20 + 'scaling_coef:' + str(scaling_coef_) + '*'*20)

accs = []
for dataset in exam_datasets:
    metrics = eval_single_dataset(image_encoder, dataset, args)
    log.info(str(dataset) + ':' + str(metrics.get('top1')*100)+'%')
    accs.append(metrics.get('top1')*100)
log.info('Avg ACC:' + str(np.mean(accs)) + '%')
