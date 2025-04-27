import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import time
import sys
import tqdm
sys.path.append('/home/ykd/project/model_merging/AdaMerging/')

import torch
from task_vectors import TaskVector
from eval import eval_single_dataset, eval_single_dataset_head, eval_single_dataset_preprocess_head
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
dataset_to_lambda = {
    'SUN397': [[1, 0, 0, 0, 0, 0, 0, 0]],
    'Cars': [[0, 1, 0, 0, 0, 0, 0, 0]],
    'RESISC45': [[0, 0, 1, 0, 0, 0, 0, 0]],
    'EuroSAT': [[0, 0, 0, 1, 0, 0, 0, 0]],
    'SVHN': [[0, 0, 0, 0, 1, 0, 0, 0]],
    'GTSRB': [[0, 0, 0, 0, 0, 1, 0, 0]],
    'MNIST': [[0, 0, 0, 0, 0, 0, 1, 0]],
    'DTD': [[0, 0, 0, 0, 0, 0, 0, 1]],
}
model = 'ViT-B-32'
source_root_path = '/home/trunk/RTrunk1/ykd/model_merging/'
args = parse_arguments()
args.data_location = source_root_path+'dataset/'
args.model = model
args.save = source_root_path+model
args.logs_path = '~/project/model_merging/AdaMerging/taskarithmetic/src/logs/' + model
pretrained_checkpoint = source_root_path + model + '/zeroshot.pt'

str_time_ = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
log = create_log_dir(args.logs_path, 'log_{}_Task_wise_AdaMerging.txt'.format(str_time_))
args.log = log

task_vectors = [TaskVector(pretrained_checkpoint, source_root_path+model+'/'+dataset_name+'/finetuned.pt') for dataset_name in exam_datasets]

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
        print("alph", alph)
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

model = ModelWrapper(pretrained_model, exam_datasets)
model = model.to(args.device)
_, names = make_functional(model)

paramslist = []
paramslist += [tuple(v.detach().requires_grad_().cpu() for _, v in pretrained_model_dic.items())] # pretrain
paramslist += [tuple(v.detach().requires_grad_().cpu() for _, v in tv.vector.items())  for i, tv in enumerate(task_vectors)] # task vectors



adamerging_mtl_model = AdaMerging(paramslist, model, names, exam_datasets)

from datasets.registry import get_dataset
from datasets.common import get_dataloader, maybe_dictionarize, get_dataloader_shuffle
from modeling import ImageClassifier
from merging_cofficient import get_merging_cofficients
import utils
import torch.nn.functional as F

ralpha = get_merging_cofficients('tw_adamerging', 'ViT-B-32')  
alpha = torch.Tensor(ralpha)

Total_ACC = 0.
for dataset_name in ['SUN397', 'Cars', 'RESISC45', 'EuroSAT', 'SVHN', 'GTSRB', 'MNIST', 'DTD']:
    lambdas_value = dataset_to_lambda[dataset_name]
    adamerging_mtl_model.lambdas_raw = torch.nn.Parameter(torch.Tensor(lambdas_value))
    print(f'{dataset_name} init lambda:')
    print(adamerging_mtl_model.lambdas())
    image_encoder = adamerging_mtl_model.get_image_encoder()

    classification_head = adamerging_mtl_model.get_classification_head(dataset_name)

    model = ImageClassifier(image_encoder, classification_head)

    model.eval()


    dataset = get_dataset(dataset_name, model.val_preprocess, location=args.data_location,  batch_size=args.batch_size)
    dataloader = get_dataloader(dataset, is_train=False, args=args, image_encoder=None)
    device = args.device
    # 假设 txt 文件路径
    output_file_path = f"/home/ykd/project/model_merging/AdaMerging/embedding/{dataset_name}_embedding.txt"
    with open(output_file_path, 'w') as f:
        with torch.no_grad():
            top1, correct, n = 0., 0., 0.
            sample_count = 0
            for i, data in enumerate(tqdm.tqdm(dataloader)):
                data = maybe_dictionarize(data)
                x = data['images'].to(device)
                y = data['labels'].to(device)

                logits = utils.get_logits(x, model)
                
                logits_list = logits.cpu().tolist()

                for idx in range(len(y)):
                    f.write(f"Sample {sample_count}, Label: {y[idx].item()}\n")  # 样本序号
                    f.write(f"Logits model: {logits_list[idx]}\n")
                    sample_count += 1