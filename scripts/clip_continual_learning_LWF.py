import re
from _common import *

import logging
import os
import itertools
import pandas as pd
import torch
from torch import nn
from torch.optim import Adam
from copy import deepcopy
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import hydra
from omegaconf import DictConfig


from clip_checkpoint_path import (
    CHECKPOINT_DIR,
    finetuned_model_path,
    pretrained_model_path,
    sam_retraining_model_path,
)
from timer import timer
from src.adamerging import softmax_entropy
from src.clip_eval import eval_single_dataset, eval_single_dataset_preprocess_head
# 注意：这里假设您已经将前面定义的多模型 Optimal Transport 融合类保存为 src/optimal_transport_mask.py
from src.tasks.shortest_route_mask import *
from src.tasks.shortest_route_classification_heads import *
from src.datasets.common import maybe_dictionarize
from src.heads import get_classification_head
from src.task_vectors import StateDict, TaskVector
from src.task_wise_fusion import *
from src.task_wise_fusion import check_parameterNamesMatch
from src.utils import num_parameters, timeit_context
from tqdm.autonotebook import tqdm

log = logging.getLogger(__name__)
 

class Program:
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        cfg.save = str(CHECKPOINT_DIR / cfg.model)
        cfg.data_location = str(DATA_DIR)

        save_dir = RESULTS_DIR / cfg.model
        if cfg.version is not None:
            save_dir = save_dir / f"version_{cfg.version}"
        os.makedirs(save_dir, exist_ok=True)
        self.results_path = save_dir / "clip_optimal_transport.csv"
        self.ckpt_dir = self.results_path.parent / os.path.basename(self.results_path).split(".")[0]
        self.ckpt_path = self.ckpt_dir / os.path.basename(self.results_path).replace(".csv", ".pt")
        self.individual_results_path = save_dir / "clip_optimal_transport_individuals.csv"
        log.info(f'Results will be saved to "{self.results_path}"')

    def run(self):
        self.load_models()
        self.load_datasets()
        self.Shortest_Route_Fusion()
        self.eval_individuals()
            
        
    def eval_individuals(self):
        log.info("Start eval individuals (optimal transport version)")
        cfg = self.cfg

        # 修正此处加载方式
        loaded_dict = torch.load(
            self.ckpt_path, map_location="cpu"
        )["merged_state_dict"]

        device = next(iter(self.task_vectors[0].values())).device
        merged_state_dict = {k: v.to(device) for k, v in loaded_dict.items()}

        results = {"dataset": [], "acc": []}
        Total_ACC = 0
        for dataset_idx, dataset_name in enumerate(tqdm(cfg.datasets, desc="Evaluating individual models")):
            model = deepcopy(self.pretrained_model)

            # 对于每个任务，我们将融合后的参数加到预训练模型上
            # state_dict = model.state_dict()
            # for n, p in merged_state_dict.items():
            #     if n in state_dict:
            #         state_dict[n] = state_dict[n] + p.to(state_dict[n].device)
            #model.load_state_dict(state_dict)
            
            model.load_state_dict(merged_state_dict)
            model = model.cuda(1)

            metrics = eval_single_dataset_preprocess_head(
                model,
                self.classification_heads[dataset_name],
                dataset_name,
                cfg,
                dataloader=self.test_loaders[dataset_name],
            )
            Total_ACC += metrics["top1"]
            log.info(f"Eval: dataset: {dataset_name} ACC: {metrics['top1']:.2f}")

            results["dataset"].append(dataset_name)
            results["acc"].append(metrics["top1"])

        log.info(f"Eval: Avg ACC: {Total_ACC/len(cfg.datasets):.2f}\n")
        pd.DataFrame(results).to_csv(self.individual_results_path, index=False)


    @torch.no_grad()
    def eval_model_on_datasets(self, epoch_idx: int, results: dict):
        model = deepcopy(self.pretrained_model)

        # 对于每个任务，我们将融合后的参数加到预训练模型上
        state_dict = model.state_dict()
        for n, p in self.fused_state_dict.items():
            if n in state_dict:
                state_dict[n] = state_dict[n] + p.to(state_dict[n].device)
        model.load_state_dict(state_dict)
        model = model.cuda(1)

        self.model.eval()
        Total_ACC = 0
        for dataset_name in self.cfg.datasets:
            classification_head = self.classification_heads[dataset_name]
            metrics = eval_single_dataset_preprocess_head(
                self.model,
                classification_head,
                dataset_name,
                self.cfg,
                dataloader=self.test_loaders[dataset_name],
            )
            Total_ACC += metrics["top1"]
            log.info(f"Eval: dataset: {dataset_name} ACC: {metrics['top1']:.2f}")

            if results is not None:
                results["epoch"].append(epoch_idx)
                results["dataset"].append(dataset_name)
                results["acc"].append(metrics["top1"])
        log.info(f"Eval: Avg ACC: {Total_ACC/len(self.cfg.datasets):.2f}\n")

    def load_models(self):
        cfg = self.cfg

        if cfg.sam_retraining:
            log.info("SAM retrained model is used")
            _finetuned_model_path = sam_retraining_model_path
        else:
            _finetuned_model_path = finetuned_model_path
        with timeit_context():
            log.info("load models")
            pretrained_model: nn.Module = torch.load(pretrained_model_path(cfg.model), map_location="cpu")
            task_vectors: List[StateDict] = [
                TaskVector(
                    pretrained_checkpoint=pretrained_model_path(cfg.model),
                    finetuned_checkpoint=_finetuned_model_path(cfg.model, dataset_name),
                ).vector
                for dataset_name in tqdm(cfg.datasets)
            ]
            # Check if the parameter names in the task vectors match
            check_parameterNamesMatch(task_vectors)

        self.pretrained_model = pretrained_model
        #! if gpu memory is not enough, comment the following line
        if cfg.model == "ViT-B-32" or cfg.model == "ViT-B-16":
            task_vectors = [{k: v.cuda(1) for k, v in tv.items()} for tv in task_vectors]
        # elif cfg.model == "ViT-L-14":
        #     temp = []
        #     for tv in task_vectors[:4]:
        #         temp.append({k: v.cuda(1, non_blocking=True) for k, v in tv.items()})
        #     for tv in task_vectors[4:]:
        #         temp.append({k: v.cuda(2, non_blocking=True) for k, v in tv.items()})
        #     task_vectors = temp
        self.task_vectors = task_vectors

        self.classification_heads = {dataset_name: get_classification_head(cfg, dataset_name).cuda(1) for dataset_name in cfg.datasets}


    def Shortest_Route_Fusion(self):
        pretrained_model = self.pretrained_model
        task_vectors = self.task_vectors

        for p in pretrained_model.parameters():
            p.detach_().requires_grad_(False)
        
        # 初始化第一个mask
        self.masks_pre_init = ShortestRouteMask(
            temperature=0.5,
            state_dict=task_vectors[0],
            init_value=0,
            draw_sample=True,
        )
        self.masks_post_init = ShortestRouteMask(
            temperature=0.5,
            state_dict=task_vectors[1],
            init_value=0,
            draw_sample=True,
        )
        
        self.masks_pre = self.masks_pre_init._draw_mask()
        self.masks_post = self.masks_post_init._draw_mask()
        
        # 逐步融合后续任务
        for i in range(0, len(task_vectors) - 1):

            post_task_dataset = self.cfg.datasets[i+1]
            pre_task_dataset = self.cfg.datasets[i]
            task_vector_pre = task_vectors[i]
            task_vector_post = task_vectors[i+1]
            
            pre_task_dataloader = self.shuffled_test_loader_iters[pre_task_dataset]
            post_task_dataloader = self.shuffled_test_loader_iters[post_task_dataset]
            self.sr_masks_pre, self.sr_masks_post, self.merged_state_dict = compute_sr_mask(
                task_vector_pre=task_vector_pre,
                task_vector_post=task_vector_post, 
                pretrained_model=deepcopy(pretrained_model),
                masks_pre=self.masks_pre,
                masks_post=self.masks_post,
                pre_task_dataloader=pre_task_dataloader,
                post_task_dataloader= post_task_dataloader,
                )

            self.classification_heads[pre_task_dataset] = compute_sr_classification_heads_single(
                task_vector_pre=task_vector_pre,
                pretrained_model=deepcopy(pretrained_model),
                merged_state_dict=self.merged_state_dict,
                classification_head_pre=self.classification_heads[pre_task_dataset],
                pre_task_dataloader=pre_task_dataloader,
            )

            if(i == len(task_vectors) - 2):
                self.classification_heads[post_task_dataset] = compute_sr_classification_heads_single(
                    task_vector_pre=task_vector_post,
                    pretrained_model=deepcopy(pretrained_model),
                    merged_state_dict=self.merged_state_dict,
                    classification_head_pre=self.classification_heads[post_task_dataset],
                    pre_task_dataloader=post_task_dataloader,
                )
            
            # 更新融合后的参数
            merged_vector = deepcopy(task_vectors[i])
            for k, v in self.merged_state_dict.items():
                if k in pretrained_model.state_dict():
                    merged_vector[k] = self.merged_state_dict[k].to("cuda:1") - pretrained_model.state_dict()[k].to("cuda:1")

            task_vectors[i+1] = merged_vector
            
            self.masks_pre = self.masks_pre_init._draw_mask()
            self.masks_post = self.masks_post_init._draw_mask()
    
        # 保存检查点
        os.makedirs(self.ckpt_dir, exist_ok=True)
        torch.save(
            {
                "merged_state_dict": self.merged_state_dict,
            },
            self.ckpt_path,
        )
        

    def load_datasets(self):
        cfg = self.cfg
        if cfg.corruption is None:
            from src.datasets.registry import get_dataset
        else:
            from src.datasets.corruption.registry import get_dataset

        datasets_kwargs = dict(
            location=cfg.data_location,
            batch_size=16,
            num_workers=8,
        )
        if cfg.corruption is not None:
            datasets_kwargs["corruption"] = cfg.corruption
        datasets = {
            dataset_name: get_dataset(
                dataset_name,
                self.pretrained_model.val_preprocess,
                **datasets_kwargs,
            )
            for dataset_name in cfg.datasets
        }
        shuffled_test_loaders = {
            dataset_name: dataset.test_loader_shuffle for dataset_name, dataset in datasets.items()
        }
        shuffled_test_loader_iters = {
            dataset_name: iter(itertools.cycle(dataloader))
            for dataset_name, dataloader in shuffled_test_loaders.items()
        }
        self.datasets = datasets
        self.test_loaders = {name: ds.test_loader for name, ds in datasets.items()}
        self.shuffled_test_loader_iters = shuffled_test_loader_iters


@hydra.main(config_path=str(CONFIG_DIR), config_name="default", version_base=None)
def main(cfg: DictConfig) -> None:
    Program(cfg).run()


if __name__ == "__main__":
    main()
