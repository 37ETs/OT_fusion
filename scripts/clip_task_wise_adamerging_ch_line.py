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

    # 保存融合后的权重
    os.makedirs(results_path.parent / os.path.basename(results_path).split(".")[0], exist_ok=True)
    torch.save(
        module.task_wise_weight.detach().cpu(),
        results_path.parent
            / os.path.basename(results_path).split(".")[0]
            / os.path.basename(results_path).replace(".csv", ".pt"),
    )

    # ========== Step 2: 使用融合后模型 + 默认分类头 先做一次评估 =========
    module.eval()
    # evaluate(
    #     cfg,
    #     image_encoder=module,
    #     classification_heads=classification_heads,
    #     dataloaders=shuffled_test_loaders,
    #     exam_datasets=exam_datasets,
    #     results=results,
    #     epoch_idx=0,
    # )
    # pd.DataFrame(results).to_csv(results_path, index=False)

    # ========== Step 3: 使用 Weight Imprinting 对分类头进行几何修正 =========
    #
    # 说明：
    #  - 这里假设我们可以使用 "datasets[ds].train_loader" 里的少量训练数据（含标签），
    #    如果要换成 test_loader 也可行，只要有标签或能伪标签即可。
    #  - 仅示例性给出一次性处理：提取全部特征后更新权重。
    #
    # ==================== Method 5: 多轮原型向量更新示例 ======================
# 假设:
#   module               # 融合后主干 (冻结)
#   classification_heads # dict, 每个数据集一个分类头
#   unlabeled_loader     # dict, 每个数据集对应的无标签 DataLoader
#   exam_datasets        # list, 数据集名称
#   cfg.device           # 设备 (e.g. "cuda:0")
#   maybe_dictionarize() # 处理 batch 的函数
# 其他: alpha, gamma, num_iterations 可根据需求调整

    module.eval()
    for param in module.parameters():
        param.requires_grad = False

    num_iterations = 5  # 分步迭代轮数
    alpha = 0.2         # 小步长系数
    gamma = 50.0        # 原型向量单位化后乘的尺度

    for iter_idx in range(num_iterations):
        log.info(f"[Prototype Update] Iteration {iter_idx+1}/{num_iterations}")
        
        for ds in exam_datasets:
            classification_head = classification_heads[ds]
            train_loader = datasets[ds].train_loader
            classification_head.eval()
            max_batch = 5

            # 如果是 nn.Linear, 则 classification_head.fc.weight 为 (num_classes, feature_dim)
            # 也可能是 classification_head.weight / classification_head.fc_xxx 等，视你的实现而定
            
            # 准备存储特征和伪标签
            #   S_k: list of features for predicted class k
            num_classes = classification_head.weight.shape[0]
            S_k = [[] for _ in range(num_classes)]

            # 1) 前向无标签数据，推断伪标签
            with torch.no_grad():
                for batch_idx, batch in enumerate(train_loader):
                    if batch_idx >= max_batch:
                        break
                    batch = maybe_dictionarize(batch)
                    x = batch["images"].to(cfg.device)
                    
                    # 提取特征
                    z = module(x)  # [B, D]
                    # 用当前分类头做预测
                    logits = classification_head(z)          # [B, K]
                    probs = torch.softmax(logits, dim=1)     # [B, K]
                    pred_labels = probs.argmax(dim=1)        # [B]
                    
                    # 将这些特征根据预测类别分类存放
                    for i, c in enumerate(pred_labels):
                        S_k[c.item()].append(z[i].detach())

            # 2) 计算每个类别的平均向量
            #    然后做 "小步长移动"
            new_weight = classification_head.weight.data.clone()  # 复制旧权重
            
            for k in range(num_classes):
                if len(S_k[k]) == 0:
                    # 如果本轮里没有样本被判为类 k，则保留原权重(也可考虑其他处理)
                    continue
                
                # 类 k 的原型向量
                feats_k = torch.stack(S_k[k], dim=0)  # [Nk, D]
                mu_k = feats_k.mean(dim=0)           # [D]

                # 归一化并乘以 gamma
                mu_k_norm = mu_k.norm().clamp_min(1e-6)
                mu_k_unit = mu_k / mu_k_norm
                proto_k = gamma * mu_k_unit  # 原型(单位化后乘以固定尺度)

                # 当前权重 w_k
                w_k = classification_head.weight.data[k]  # [D]

                # 小步长更新: w_k <- (1-alpha)*w_k + alpha* proto_k
                w_k_new = (1.0 - alpha) * w_k + alpha * proto_k
                new_weight[k] = w_k_new
            
            # 应用更新
            classification_head.weight.data = new_weight
            
            # 偏置可视需求处理: 保持原偏置 / 逐步更新 / 置零 等
            # 这里只是示例保留原bias不变
            # fc_layer.bias.data.zero_()   # 如果想置零，就这样写

        # 在每个 iteration 结束后，你可以选做一次评估(若有带标签的验证集/测试集)
        # 也可在所有 iteration 完成后统一评估
        # ...
        
    log.info("Multiple-step prototype alignment finished!")

    
    # ========== Step 4: 再次评估并保存结果 =========
    evaluate(
        cfg,
        image_encoder=module,
        classification_heads=classification_heads,
        dataloaders=shuffled_test_loaders,
        exam_datasets=exam_datasets,
        results=results,
        epoch_idx=1,
    )
    pd.DataFrame(results).to_csv(results_path, index=False)

    log.info(f"Optimized training finished. Results saved to {results_path}")


if __name__ == "__main__":
    main()
