import torch
import torch.nn as nn
import tqdm
import gc
import functools
from collections import defaultdict
from typing import List
from src.datasets.common import get_dataloader, maybe_dictionarize

def get_named_linears(module):
    return {name: m for name, m in module.named_modules() if isinstance(m, nn.Linear)}

@torch.no_grad()
def find_salient_weight_mask(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    keep_ratio: float = 0.0001,
    device: torch.device = torch.device("cuda:1"),
    max_batches: int = 10
):
    named_linears = get_named_linears(model)
    input_feat = defaultdict(list)

    def cache_input_hook(m, x, y, layer_name):
        """
        x is always a tuple of inputs. For a single-tensor input,
        x[0] is the actual data. We flatten everything except the last
        dimension so shape becomes [N_total, in_features].
        """
        x = x[0].detach()  # e.g. [batch_size, seq_len, in_features]
        if x.dim() > 2:
            # Flatten batch & any extra dims into one
            x = x.view(-1, x.size(-1))   # [N_total, in_features]
        input_feat[layer_name].append(x.cpu())

    # Register forward hooks on each Linear layer
    handles = []
    for name, layer in named_linears.items():
        # Use a small wrapper to bind 'name' correctly
        h = layer.register_forward_hook(
            lambda m, x, y, n=name: cache_input_hook(m, x, y, n)
        )
        handles.append(h)

    # Forward pass (collect up to max_batches of input activations)
    with torch.no_grad():
        for i, data in enumerate(tqdm.tqdm(dataloader, leave=False)):
            if i >= max_batches:
                break
            data = maybe_dictionarize(data)
            x = data["images"].to(device)
            _ = model(x)
            gc.collect()
            torch.cuda.empty_cache()

    # Remove hooks and empty cache
    for h in handles:
        h.remove()
    torch.cuda.empty_cache()

    # Build the masks
    # mask_dict = {}
    # for name, feats in input_feat.items():
    #     # Concatenate all collected activations along batch dimension
    #     feats_cat = torch.cat(feats, dim=0)  # [N_total, in_features]
    #     # Mean absolute activation per input dimension
    #     mean_amp = feats_cat.abs().mean(dim=0)  # [in_features]

    #     # Keep top-K% of channels by mean amplitude
    #     threshold = torch.quantile(mean_amp, 1 - keep_ratio)
    #     keep_mask_1d = (mean_amp >= threshold).float()  # [in_features]

    #     # Expand to 2D mask matching the layer’s weights [out_f, in_f]
    #     out_f, in_f = named_linears[name].weight.shape
    #     keep_mask_2d = keep_mask_1d.unsqueeze(0).expand(out_f, in_f)

    #     # Save mask keyed by parameter name
    #     mask_dict[name + ".weight"] = keep_mask_2d

    # return mask_dict
    mask_dict = {}
    for name, feats in input_feat.items():
        feats_cat = torch.cat(feats, dim=0)
        layer = named_linears[name]
        W_abs = layer.weight.abs().to(feats_cat.device)
        
        # 计算联合重要性（激活均值 × 权重绝对值）
        mean_amp = feats_cat.abs().mean(dim=0)
        importance = W_abs * mean_amp.unsqueeze(0)  # [out_f, in_f]
        
        # 全局阈值选择
        threshold = torch.quantile(importance.flatten(), 1 - keep_ratio)
        mask = (importance >= threshold).float()
        mask_dict[name + ".weight"] = mask
        
    return mask_dict



import torch
import torch.nn as nn
import tqdm
import gc
from collections import defaultdict
from src.datasets.common import get_dataloader, maybe_dictionarize

def get_named_linears(module):
    return {name: m for name, m in module.named_modules() if isinstance(m, nn.Linear)}

def compute_grad_importance(model, dataloader, loss_fn, device, max_batches=100):
    model.eval()
    named_linears = get_named_linears(model)
    
    # 存储每个权重的贡献
    importance_scores = {name: torch.zeros_like(layer.weight, device=device)
                         for name, layer in named_linears.items()}
    
    for i, data in enumerate(tqdm.tqdm(dataloader, leave=False)):
        if i >= max_batches:
            break
        data = maybe_dictionarize(data)
        inputs = data["images"].to(device)
        labels = data["labels"].to(device)  # 假设监督任务，这里可按需求调整
        
        model.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        
        for name, layer in named_linears.items():
            if layer.weight.grad is not None:
                # 采用一阶泰勒贡献评估重要性
                importance_scores[name] += (layer.weight * layer.weight.grad).abs()
        
        gc.collect()
        torch.cuda.empty_cache()
    
    # 对重要性进行归一化
    for name in importance_scores:
        importance_scores[name] /= max_batches
    
    return importance_scores

def generate_importance_mask(importance_scores, keep_ratio):
    mask_dict = {}
    for name, scores in importance_scores.items():
        threshold = torch.quantile(scores.flatten(), 1 - keep_ratio)
        mask = (scores >= threshold).float()
        mask_dict[name + ".weight"] = mask#.cpu()
    return mask_dict


