from random import shuffle
from _common import *

log = logging.getLogger(__name__)

from src.adamerging import AdaMerging, ModelWrapper, make_functional, softmax_entropy
from src.clip_eval import eval_single_dataset_preprocess_head
from src.heads import ClassificationHead, get_classification_head
from src.modeling import ImageEncoder
from src.task_vectors import StateDict, TaskVector
from src.task_wise_fusion import *
from src.task_wise_fusion import check_parameterNamesMatch
from src.utils import num_parameters, timeit_context
from torch.utils.data import DataLoader
from src.tasks.salient_weight_mask import get_named_linears

from clip_checkpoint_path import CHECKPOINT_DIR, finetuned_model_path, pretrained_model_path, sam_retraining_model_path

torch.multiprocessing.set_sharing_strategy('file_system')


# ... 省略上方与数据加载、模型初始化等相关的代码逻辑 ...

from sklearn.decomposition import PCA

def evaluate(
    cfg,
    image_encoder: ImageEncoder,
    classification_heads: Dict[str, ClassificationHead],
    exam_datasets: List[str],
    results: Dict[str, List[Any]],
    dataloaders: Dict[str, DataLoader] = None,
    epoch_idx: int = 0,
):
    """评估函数，计算并记录各个数据集的 top1 准确率。"""
    Total_ACC = 0.0
    for dataset_name in exam_datasets:
        classification_head = classification_heads[dataset_name]
        metrics = eval_single_dataset_preprocess_head(
            image_encoder,
            classification_head,
            dataset_name,
            cfg,
            dataloader=dataloaders[dataset_name] if dataloaders is not None else None,
        )
        Total_ACC += metrics["top1"]
        log.info(f"[Eval] Epoch: {epoch_idx} | Dataset: {dataset_name} | ACC: {metrics['top1']}")

        results["epoch"].append(epoch_idx)
        results["dataset"].append(dataset_name)
        results["acc"].append(metrics["top1"])

    log.info(
        "Avg ACC across tasks: {:.2f}%\n".format(Total_ACC / len(exam_datasets))
    )


@hydra.main(config_path=str(CONFIG_DIR), config_name="default", version_base=None)
def main(cfg: DictConfig) -> None:
    model = cfg.model
    cfg.data_location = DATA_DIR
    cfg.save = str(CHECKPOINT_DIR / cfg.model)
    exam_datasets: List[str] = cfg.datasets
    
    from src.datasets.common import maybe_dictionarize
    from src.datasets.registry import get_dataset

    pretrained_model: nn.Module = torch.load(pretrained_model_path(model))

    task_vectors: List[StateDict] = [
        TaskVector(
            pretrained_checkpoint=pretrained_model_path(model),
            finetuned_checkpoint=(sam_retraining_model_path if cfg.sam_retraining else finetuned_model_path)(model, dataset_name),
        ).vector for dataset_name in exam_datasets
    ]

    check_parameterNamesMatch(task_vectors)

    # 创建TaskWiseMergedModel
    task_wise_weight = get_task_wise_weights(len(task_vectors), init_values=0.2 if cfg.sam_retraining else 0.3)
    module = TaskWiseMergedModel(
        pretrained_model=pretrained_model,
        task_wise_weight=task_wise_weight,
        task_vectors=task_vectors,
        clamp_weights=True,
    ).cuda(1)

    module.train_preprocess = pretrained_model.train_preprocess
    module.val_preprocess = pretrained_model.val_preprocess

    if cfg.sam_retraining:
        save_dir = RESULTS_DIR / "sam_retraining" / cfg.model
    else:
        save_dir = RESULTS_DIR / cfg.model
    os.makedirs(save_dir, exist_ok=True)
    results = {"epoch": [], "dataset": [], "acc": []}
    results_path = save_dir / "task_wise_adamerging.csv"

    classification_heads = {
        ds: get_classification_head(cfg, ds).cuda(1)
        for ds in exam_datasets
    }

    datasets = {
        ds: get_dataset(
            ds,
            preprocess=module.val_preprocess,
            location=cfg.data_location,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers
        )
        for ds in exam_datasets
    }
    
    dataloaders = {ds: datasets[ds].test_loader_shuffle for ds in exam_datasets}
    shuffled_iters = {ds: iter(itertools.cycle(dataloaders[ds])) for ds in exam_datasets}
    
    shuffled_test_loaders = {dataset_name: dataset.test_loader_shuffle for dataset_name, dataset in datasets.items()}
    shuffled_test_loader_iters = {dataset_name: iter(itertools.cycle(dataloader)) for dataset_name, dataloader in shuffled_test_loaders.items()}

    criterion = nn.CrossEntropyLoss()

    # ========== Step 1: 训练融合权重 (task_wise_weight) =============
    checkpoint_path = save_dir / "fusion_weights_checkpoint.pt"

    if checkpoint_path.exists():
        log.info("Loading checkpoint weights...")
        module.task_wise_weight.data.copy_(torch.load(checkpoint_path, map_location=cfg.device))
        module.task_wise_weight.requires_grad = False
        module.merge_weights()
    else:
        module.train()
        optimizer = torch.optim.Adam([module.task_wise_weight], lr=1e-3, betas=(0.9, 0.999))

        for step in tqdm(range(epochs := 1)):  # 这里可以自定义融合权重训练的 epoch
            total_loss = 0
            for ds in exam_datasets:
                batch = next(shuffled_test_loader_iters[ds])
                batch = maybe_dictionarize(batch)
                x, y = batch["images"].to(cfg.device), batch["labels"].to(cfg.device)

                outputs = classification_heads[ds](module(x))
                loss = criterion(outputs, y)
                total_loss += loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            module.merge_weights()

        torch.save(module.task_wise_weight.detach().cpu(), checkpoint_path)
        module.task_wise_weight.requires_grad = False
        log.info("Checkpoint weights saved.")

    # ========== Step 2: 根据finetuned weight与merging weight的分布差距，通过 PCA 调整分类头 =============
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde
    from sklearn.manifold import TSNE

    for i, dataset_name in enumerate(cfg.datasets):
        task_vector = task_vectors[i]

        # 计算单个任务权重
        task_individual_model = deepcopy(pretrained_model)
        individual_state_dict = task_individual_model.state_dict()
        
        for n, p in task_vector.items():
            if n in individual_state_dict:
                individual_state_dict[n] += p.to(individual_state_dict[n].device)
        task_individual_model.load_state_dict(individual_state_dict)
        task_individual_model.to(cfg.device)
            
        merged_model = deepcopy(pretrained_model)
        merged_model.load_state_dict(module.merged_state_dict)

        linear_layers_individual = get_named_linears(task_individual_model)
        linear_layers_fusion = get_named_linears(merged_model)
        
        # 收集所有层的权重向量
        all_weights = []
        labels = []
        layer_names = []
        
        for name in linear_layers_fusion.keys():
            if name in linear_layers_individual:
                # 展平权重并收集
                ind_w = linear_layers_individual[name].weight.detach().flatten().cpu().numpy()
                fus_w = linear_layers_fusion[name].weight.detach().flatten().cpu().numpy()
                
                all_weights.append(ind_w)
                labels.append(0)  # 0表示Individual
                layer_names.append(f"{name}_ind")
                
                all_weights.append(fus_w)
                labels.append(1)  # 1表示Fusion 
                layer_names.append(f"{name}_fus")
        
        # 转换为numpy数组
        X = np.vstack(all_weights)  # shape: (2*num_layers, D)
        y = np.array(labels)
        
        # 执行t-SNE降维
        tsne = TSNE(n_components=2, 
                    perplexity=15, 
                    random_state=42,
                    n_iter=1000)
        X_tsne = tsne.fit_transform(X)
        
        # 可视化
        plt.figure(figsize=(12, 8))
        
        # 按类别绘制
        for cls, marker in zip([0, 1], ['o', 's']):
            mask = (y == cls)
            plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1], 
                        marker=marker,
                        label='Individual' if cls==0 else 'Fusion',
                        alpha=0.6,
                        edgecolors='w')
        
        # 添加层名称标注
        for i, name in enumerate(layer_names):
            plt.annotate(name.split('_')[0],  # 显示原始层名
                        (X_tsne[i, 0], X_tsne[i, 1]),
                        textcoords="offset points",
                        xytext=(0,5),
                        ha='center',
                        fontsize=6)
        
        plt.title(f"t-SNE Projection of Weights ({dataset_name})")
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
        plt.legend()
        plt.grid(alpha=0.3)
        
        # 保存结果
        plt.savefig(os.path.join(save_dir, f"{dataset_name}_tsne_analysis.png"), 
                dpi=150, 
                bbox_inches='tight')
        plt.close()


    # 保存融合后的权重
    os.makedirs(results_path.parent / os.path.basename(results_path).split(".")[0], exist_ok=True)
    torch.save(
        module.task_wise_weight.detach().cpu(),
        results_path.parent
            / os.path.basename(results_path).split(".")[0]
            / os.path.basename(results_path).replace(".csv", ".pt"),
    )

    # 最终评估并保存结果
    evaluate(
        cfg,
        image_encoder=module,
        classification_heads=classification_heads,
        dataloaders=shuffled_test_loaders,
        exam_datasets=exam_datasets,
        results=results,
        epoch_idx=step + 1,
    )
    pd.DataFrame(results).to_csv(results_path, index=False)

    log.info(f"Optimized training finished. Results saved to {results_path}")


if __name__ == "__main__":
    main()

