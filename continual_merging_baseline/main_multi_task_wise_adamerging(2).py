import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import time
import sys
import tqdm
sys.path.append('/root/taskarithmetic')

import torch
from task_vectors import TaskVector
from eval import eval_single_dataset, eval_single_dataset_head, eval_single_dataset_preprocess_head
from args import parse_arguments
from datasets.registry import get_dataset
from datasets.common import get_dataloader, maybe_dictionarize, get_dataloader_shuffle
#建立日志
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
#获取task_num列表
name_list=args.name_list
model = 'ViT-B-32'
source_root_path = '/root'
args.data_location = source_root_path+'/dataset'
args.model = model
args.save = source_root_path+'/checkpoint/' + model
args.logs_path = '/root/taskarithmetic/src/logs/' + model
pretrained_checkpoint = source_root_path+'/checkpoint/'+model+'/zeroshot.pt'
str_time_ = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
log = create_log_dir(args.logs_path, 'log_{}_Task_wise_AdaMerging.txt'.format(str_time_))
args.log = log

# exam_datasets = ['SUN397', 'Cars', 'RESISC45', 'EuroSAT', 'SVHN', 'GTSRB', 'MNIST', 'DTD'] # SUN397 | Cars | RESISC45 | EuroSAT | SVHN | GTSRB | MNIST | DTD
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

def del_attr(obj, names):
    if len(names) == 1:
        delattr(obj, names[0])
    else:
        del_attr(getattr(obj, names[0]), names[1:])

def set_attr(obj, names, val):
    if len(names) == 1:
        setattr(obj, names[0], val)
    else:
        set_attr(getattr(obj, names[0]), names[1:], val)

def make_functional(mod):
    orig_params = tuple(mod.parameters())
    names = []
    for name, p in list(mod.named_parameters()):
        del_attr(mod, name.split("."))
        names.append(name)
    return orig_params, names

def load_weights(mod, names, params):
    for name, p in zip(names, params):
        set_attr(mod, name.split("."), p)

class ModelWrapper(torch.nn.Module):
    def __init__(self, model, initial_weights=None):
        super(ModelWrapper, self).__init__()
        self.model = model

        if hasattr(self.model, 'transformer'):
            delattr(self.model, 'transformer')

    def forward(self, images):
        features = self.model(images)
        return features

from heads import get_classification_head
class AdaMerging(torch.nn.Module):
    def __init__(self, paramslist, model, names, exam_datasets):
        super(AdaMerging, self).__init__()
        self.paramslist = paramslist
        self.model = model
        self.names = names
        self.pretrain_lambdas = torch.ones(1, 1)
        prior = 0.3
        rlambdas = torch.ones(1, len(paramslist)-1) * prior  # (1 * tasks)
        self.lambdas_raw = torch.nn.Parameter(rlambdas)

        self.classifier = []
        for dataset_name in exam_datasets:
            classification_head = get_classification_head(args, dataset_name)
            layer_name = 'classifier_{}'.format(dataset_name)
            self.add_module(layer_name, classification_head.to(args.device))
            self.classifier.append(layer_name)

    def lambdas(self):
        task_lambdas = torch.clamp(self.lambdas_raw, min=0.0, max=1.0)
        lambdass = torch.cat((self.pretrain_lambdas, task_lambdas), 1)
        return lambdass

    def collect_trainable_params(self):
        return [self.lambdas_raw]

    def get_classification_head(self, dataset_name):
        layer_name = 'classifier_{}'.format(dataset_name)
        classification_head = getattr(self, layer_name)
        return classification_head

    def get_image_encoder(self):
        alph = self.lambdas()
        params = tuple(sum(tuple(pi * lambdasi for pi, lambdasi in zip(p, alph[0].cpu()))) for j, p in enumerate(zip(*self.paramslist)))
        params = tuple(p.cuda(0) for p in params)
        load_weights(self.model, self.names, params)
        return self.model

    def forward(self, inp, dataset_name):
        alph = self.lambdas()
        params = tuple(sum(tuple(pi * lambdasi for pi, lambdasi in zip(p, alph[0].cpu()))) for j, p in enumerate(zip(*self.paramslist)))

        params = tuple(p.cuda(0) for p in params)
        load_weights(self.model, self.names, params)
        feature = self.model(inp)

        layer_name = 'classifier_{}'.format(dataset_name)
        classification_head = getattr(self, layer_name)
        out = classification_head(feature)

        return out
    
def softmax_entropy(x):
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

pretrained_model = torch.load(pretrained_checkpoint)
pretrained_model_dic = pretrained_model.state_dict()

model = ModelWrapper(pretrained_model, exam_datasets[0])
model = model.to(args.device)
_, names = make_functional(model)

best_task_vectors = []
for step in range(len(exam_datasets)):
    if step > 0:
        #保留上一轮次的best_tv
        task_vectors = [best_task_vectors[-1]]
    else:
        task_vectors = []
    #添加当前轮次的tv
    task_vectors += [
        TaskVector(pretrained_checkpoint, source_root_path + '/checkpoint/' + model + '/' + dataset_name + '/finetuned.pt')
        for dataset_name in exam_datasets[step]
    ]
    
    # 更新 paramslist
    paramslist = []
    paramslist += [tuple(v.detach().requires_grad_().cpu() for _, v in pretrained_model_dic.items())]  # pretrains
    paramslist += [tuple(v.detach().requires_grad_().cpu() for _, v in tv.vector.items()) for tv in task_vectors]  # task vectors

    # 清理 GPU 缓存
    torch.cuda.empty_cache()

    # 同一个pretrain_model下创建新的 AdaMerging 模型
    adamerging_mtl_model = AdaMerging(paramslist, model, names, exam_datasets[step])

    print('init lambda:')
    print(adamerging_mtl_model.lambdas())
    print('collect_trainable_params:')
    print(list(adamerging_mtl_model.collect_trainable_params()))

    #减少epoch为100
    epochs=100
    # 初始化优化器
    optimizer = torch.optim.Adam(adamerging_mtl_model.collect_trainable_params(), lr=1e-3, betas=(0.9, 0.999), weight_decay=0.)

    # 评估初始模型
    Total_ACC = 0.
    for dataset_name in exam_datasets[step]:
        image_encoder = adamerging_mtl_model.get_image_encoder()
        classification_head = adamerging_mtl_model.get_classification_head(dataset_name)
        metrics = eval_single_dataset_preprocess_head(image_encoder, classification_head, dataset_name, args)
        Total_ACC += metrics['top1']
        log.info('Eval: init: ' + ' dataset: ' + str(dataset_name) + ' ACC: ' + str(metrics['top1']))
    
    log.info('Eval: init: ' + ' Avg ACC:' + str(Total_ACC / len(exam_datasets[step])) + '\n')
    # 训练阶段
    for epoch in range(epochs):
        losses = 0.
        seen_dataset=[]
        #获得见过的所有数据集
        for i in range(step+1):
            for seen_name in exam_datasets[i]:
                seen_dataset.append(seen_name)
        #处理见过的所有数据集
        for dataset_name in seen_dataset:  
            dataset = get_dataset(dataset_name, pretrained_model.val_preprocess, location=args.data_location, batch_size=16)
            dataloader = get_dataloader_shuffle(dataset)

            for i, data in enumerate(tqdm.tqdm(dataloader)):
                data = maybe_dictionarize(data)
                x = data['images'].to(args.device)
                y = data['labels'].to(args.device)

                outputs = adamerging_mtl_model(x, dataset_name)
                loss = softmax_entropy(outputs).mean(0)
                losses += loss

                if i > 0: 
                    break

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        # 对于最后一个step进行评估
        if ((epoch + 1) % 100) == 0 and step==len(exam_datasets)-1:
            log.info(str(list(adamerging_mtl_model.lambdas().data)))

            Total_ACC = 0.
            for dataset_name in exam_datasets[step]:
                image_encoder = adamerging_mtl_model.get_image_encoder()
                classification_head = adamerging_mtl_model.get_classification_head(dataset_name)
                metrics = eval_single_dataset_preprocess_head(image_encoder, classification_head, dataset_name, args)
                Total_ACC += metrics['top1']
                log.info('Eval: Epoch: ' + str(epoch) + ' dataset: ' + str(dataset_name) + ' ACC: ' + str(metrics['top1']))
            
            log.info('Eval: Epoch: ' + str(epoch) + ' Avg ACC:' + str(Total_ACC / len(exam_datasets[step])) + '\n')
    with torch.no_grad():
        best_tv = {}
        for (name, param) in zip(names, adamerging_mtl_model.get_image_encoder().parameters()):
            best_tv[name] = param.cpu() - pretrained_model_dic[name].cpu()

        # 保存最佳 task_vector
        best_task_vectors.append(TaskVector(vector=best_tv))