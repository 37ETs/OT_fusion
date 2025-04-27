from random import shuffle
from tabnanny import check
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
from visualization import *

from clip_checkpoint_path import CHECKPOINT_DIR, finetuned_model_path, pretrained_model_path, sam_retraining_model_path

torch.multiprocessing.set_sharing_strategy('file_system')

import matplotlib.pyplot as plt
import seaborn as sns


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
    # Store initial classification head state dicts
    classification_heads_before = {
        ds: {k: p.clone() for k, p in classification_heads[ds].named_parameters()}
        for ds in exam_datasets
    }

    datasets = {ds: get_dataset(ds, preprocess=module.val_preprocess, location=cfg.data_location,
                                batch_size=cfg.batch_size, num_workers=cfg.num_workers)
                for ds in exam_datasets}
    
    dataloaders = {ds: datasets[ds].test_loader_shuffle for ds in exam_datasets}
    shuffled_iters = {ds: iter(itertools.cycle(dataloaders[ds])) for ds in exam_datasets}
    
    shuffled_test_loaders = {dataset_name: dataset.test_loader_shuffle for dataset_name, dataset in datasets.items()}
    shuffled_test_loader_iters = {dataset_name: iter(itertools.cycle(dataloader)) for dataset_name, dataloader in shuffled_test_loaders.items()}

    criterion = nn.CrossEntropyLoss()
    
    # Step 1: 训练融合权重 (task_wise_weight)，如果检查点存在则直接加载
    checkpoint_path = save_dir / "fusion_weights_checkpoint.pt"

    # Step 1: 训练融合权重 (task_wise_weight)，如果检查点存在则直接加载
    if checkpoint_path.exists():
        log.info("Loading checkpoint weights...")
        module.task_wise_weight.data.copy_(torch.load(checkpoint_path, map_location=cfg.device))
        module.task_wise_weight.requires_grad = False
        module.merge_weights()
    else:
        # ========== Step 1: 训练融合权重 (task_wise_weight) =============
        module.eval()
        outer_optimizer = torch.optim.Adam([module.task_wise_weight], lr=1e-3, betas=(0.9, 0.999))

        for step in tqdm(range(epochs := 200)):  # 自定义融合权重训练次数
            outer_loss = 0.0
            for ds in exam_datasets:
                batch = next(shuffled_test_loader_iters[ds])
                batch = maybe_dictionarize(batch)
                x, y = batch["images"].to(cfg.device), batch["labels"].to(cfg.device)

                outputs = classification_heads[ds](module(x))
                loss = criterion(outputs, y)
                outer_loss += loss

            outer_optimizer.zero_grad()
            outer_loss.backward()
            outer_optimizer.step()

            module.merge_weights()

        # 固定融合权重，不再更新
        torch.save(
            module.task_wise_weight.detach().cpu(),
            checkpoint_path,
        )
        module.task_wise_weight.requires_grad = False
        module.merge_weights()


        # 在分类头训练之前添加检查点加载
    classification_heads_checkpoint_path = save_dir / "classification_heads_checkpoint.pt"

    # 加载分类头检查点（如果存在）
    if classification_heads_checkpoint_path.exists():
        log.info("Loading classification heads checkpoint...")
        classification_heads_state_dict = torch.load(classification_heads_checkpoint_path, map_location=cfg.device)
        for ds in exam_datasets:
            classification_heads[ds].load_state_dict(classification_heads_state_dict[ds])
    else:
        # ========== Step 2: 训练分类头 (classification_heads) ============
        inner_optimizers = {
            ds: torch.optim.Adam(classification_heads[ds].parameters(), lr=1e-3, betas=(0.9, 0.999))
            for ds in exam_datasets
        }

        for ds in exam_datasets:
            classification_heads[ds].train()
            for p in classification_heads[ds].parameters():
                p.requires_grad = True

            for step in  tqdm(range(epochs := 1000)):  # 自定义分类头训练次数
                batch = next(shuffled_iters[ds])
                batch = maybe_dictionarize(batch)
                x, y = batch["images"].to(cfg.device), batch["labels"].to(cfg.device)

                inner_optimizers[ds].zero_grad()
                outputs = classification_heads[ds](module(x))
                loss = criterion(outputs, y)
                loss.backward()
                inner_optimizers[ds].step()
                
        # 保存分类头检查点# 训练完成后保存分类头检查点
        classification_heads_state_dict = {ds: classification_heads[ds].state_dict() for ds in exam_datasets}
        torch.save(classification_heads_state_dict, classification_heads_checkpoint_path)
        log.info(f"Classification heads checkpoint saved to {classification_heads_checkpoint_path}")   
        
    os.makedirs(results_path.parent / os.path.basename(results_path).split(".")[0], exist_ok=True)
    torch.save(
        module.task_wise_weight.detach().cpu(),
        results_path.parent / os.path.basename(results_path).split(".")[0] / os.path.basename(results_path).replace(".csv", ".pt"),
    )      

    # After training, visualize distributions
    # merged_model_dict = {
    #     k: p.clone() for k, p in module.named_parameters() if 'linear' in k and p.dim() == 2
    # }
    for ds in exam_datasets:
        classification_head_after = {k: p.clone() for k, p in classification_heads[ds].named_parameters()}
        visualize_distributions_svd(classification_heads_before[ds], classification_head_after, ds, save_dir)

    evaluate(
        cfg,
        image_encoder=module,
        classification_heads=classification_heads,
        dataloaders=shuffled_test_loaders,
        exam_datasets=exam_datasets,
        results=results,
        epoch_idx=2,
    )
    pd.DataFrame(results).to_csv(results_path, index=False)

    log.info(f"Optimized training finished. Results saved to {results_path}")


if __name__ == "__main__":
    main()
