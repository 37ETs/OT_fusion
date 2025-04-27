import torch
from torch.optim import Adam
from geomloss import SamplesLoss


# 二阶段训练版本
def compute_ot_mask_two_phase(
    state_dict_0: dict,
    state_dict_1: dict,
    masks_pre: dict,
    masks_post: dict,
    lr: float = 0.1,
    blur: float = 0.08,
    lmd: float = 0.01,
    max_epochs: int = 100,
    chunk_size: int = 20000,
    early_stop_patience: int = 10,
):
    """
    二阶段训练版本：在第0、2、4...个epoch只训练 masks_pre（固定 masks_post），
    在第1、3、5...个epoch只训练 masks_post（固定 masks_pre）。
    同时保留分块计算 + 早停机制以提高效率。
    ----------
    参数：
        state_dict_0, state_dict_1: 分别是两个下游任务的参数增量 (task vector)
        masks_pre, masks_post: 初始掩码 (nn.Parameter或tensor)，将被优化
        lr: 学习率
        blur: Sinkhorn 模糊系数
        lmd: 稀疏正则系数
        max_epochs: 最大迭代次数
        chunk_size: 每次迭代在 flatten 后的参数里随机采样多少个元素
        early_stop_patience: 若 loss 多轮不下降则提前停止
    ----------
    返回：
        ot_masks_pre, ot_masks_post: 经过 sigmoid 后的掩码张量（软掩码）
    """

    # 准备 sinkhorn loss
    sinkhorn_loss = SamplesLoss(loss="sinkhorn", blur=blur)

    # 确保所有 masks 都是 nn.Parameter，方便优化器管理
    for k in masks_pre.keys():
        if not isinstance(masks_pre[k], torch.nn.Parameter):
            masks_pre[k] = torch.nn.Parameter(masks_pre[k])
        if not isinstance(masks_post[k], torch.nn.Parameter):
            masks_post[k] = torch.nn.Parameter(masks_post[k])

    # 为 masks_pre 和 masks_post 分别创建独立的优化器
    optimizer_pre = Adam(masks_pre.values(), lr=lr)
    optimizer_post = Adam(masks_post.values(), lr=lr)

    best_loss = float("inf")
    no_improve_count = 0  # 用于早停

    # 开始训练
    for epoch in range(max_epochs):
        # 判断当前 epoch 训练哪一部分
        train_pre = (epoch % 2 == 0)  # 偶数 epoch 训练 pre；奇数 epoch 训练 post

        if train_pre:
            optimizer_pre.zero_grad()
        else:
            optimizer_post.zero_grad()

        total_loss = 0.0

        # 逐层（逐 key）处理，做分块/随机采样
        for k in state_dict_0.keys():
            p0 = state_dict_0[k].view(-1)
            p1 = state_dict_1[k].view(-1)

            length = p0.shape[0]
            if length <= chunk_size:
                idx = torch.arange(length, device=p0.device)
            else:
                idx = torch.randperm(length, device=p0.device)[:chunk_size]

            sample_p0 = p0[idx]
            sample_p1 = p1[idx]

            # 同样对掩码进行 flatten + 采样
            # 注意：只计算当前要训练的那部分损失
            if train_pre:
                # 只训练 masks_pre => masks_post 暂时固定，仅在计算对齐时可能会用到固定值(如1)
                mask_pre_sigmoid = torch.sigmoid(masks_pre[k].view(-1)[idx])
                # 这里我们可以让 post 掩码先不参与 sinkhorn（也有些实现会让它 = 1 或不包含在 loss 内）
                # 下面是只对 pre 掩码那侧做 OT，可以尝试让对面保持原样等。
                masked_tensor_0_pre = sample_p0 * mask_pre_sigmoid
                masked_tensor_1_pre = sample_p1 * mask_post_sigmoid

                # 计算 OT
                ot_loss_pre = sinkhorn_loss(
                    masked_tensor_0_pre.view(-1, 1),
                    masked_tensor_1_pre.view(-1, 1)
                )
                # 稀疏正则
                reg_pre = torch.norm(mask_pre_sigmoid, p=1)

                # post 相关的正则可选做/不做
                # 如果希望完全固定 post，就不加 reg_post
                # 若想保持原策略，也可加上:
                reg_post = 0.0

                layer_loss = ot_loss_pre + lmd * (reg_pre + reg_post)

            else:
                # 只训练 masks_post => masks_pre 暂时固定
                mask_post_sigmoid = torch.sigmoid(masks_post[k].view(-1)[idx])
                masked_tensor_0_post = sample_p0 * mask_pre_sigmoid
                masked_tensor_1_post = sample_p1 * mask_post_sigmoid

                ot_loss_post = sinkhorn_loss(
                    masked_tensor_0_post.view(-1, 1),
                    masked_tensor_1_post.view(-1, 1)
                )
                reg_post = torch.norm(mask_post_sigmoid, p=1)

                # pre 不参与更新时，可不加 reg_pre
                reg_pre = 0.0

                layer_loss = ot_loss_post + lmd * (reg_pre + reg_post)

            total_loss += layer_loss

        # 反向传播 + 更新（只针对当前所训练的掩码）
        total_loss.backward()
        if train_pre:
            optimizer_pre.step()
        else:
            optimizer_post.step()

        # early stopping 检查
        curr_loss_val = total_loss.item()
        if curr_loss_val < best_loss - 1e-6:
            best_loss = curr_loss_val
            no_improve_count = 0
        else:
            no_improve_count += 1

        # 日志打印
        if epoch % 10 == 0:
            which_mask = "masks_pre" if train_pre else "masks_post"
            print(f"[Epoch {epoch} - Train {which_mask}] total_loss = {curr_loss_val:.4f} (best={best_loss:.4f})")

        # 如果超过 patience 次没有改进，则提前结束
        if no_improve_count >= early_stop_patience:
            print(f"Early stopping at epoch={epoch}")
            break

    # 最后将参数 sigmoid 化得到最终的掩码
    ot_masks_pre = {k: torch.sigmoid(v) for k, v in masks_pre.items()}
    ot_masks_post = {k: torch.sigmoid(v) for k, v in masks_post.items()}
    return ot_masks_pre, ot_masks_post

# 同时训练 masks_pre & masks_post
def compute_ot_mask(
    state_dict_0: dict,
    state_dict_1: dict,
    masks_pre: dict,
    masks_post: dict,
    lr: float = 0.1,
    blur: float = 0.08,
    lmd: float = 0.01,
    max_epochs: int = 100,
    chunk_size: int = 20000,
    early_stop_patience: int = 10,
):
    """
    同时训练 masks_pre & masks_post，并采用分块计算 + 早停来加速。
    ----------
    参数：
        state_dict_0, state_dict_1: 分别是两个下游任务的参数增量 (task vector)
        masks_pre, masks_post: 初始掩码 (nn.Parameter或tensor)，将被优化
        lr: 学习率
        blur: Sinkhorn 模糊系数
        lmd: 稀疏正则系数
        max_epochs: 最大迭代次数
        chunk_size: 每次迭代在 flatten 后的参数里随机采样多少个元素
        early_stop_patience: 若 loss 多轮不下降则提前停止
    ----------
    返回：
        ot_masks_pre, ot_masks_post: 经过 sigmoid 后的掩码张量（软掩码）
    """

    # 准备 sinkhorn loss
    sinkhorn_loss = SamplesLoss(loss="sinkhorn", blur=blur)

    # 确保所有 masks 都是可训练的 nn.Parameter
    # 通常在外面已 wrap 成 Parameter，这里再确保一下
    for k in masks_pre.keys():
        if not isinstance(masks_pre[k], torch.nn.Parameter):
            masks_pre[k] = torch.nn.Parameter(masks_pre[k])
        if not isinstance(masks_post[k], torch.nn.Parameter):
            masks_post[k] = torch.nn.Parameter(masks_post[k])

    #把 pre + post 放进同一个优化器
    params_to_optimize = list(masks_pre.values()) + list(masks_post.values())
    optimizer = Adam(params_to_optimize, lr=lr)

    best_loss = float("inf")
    no_improve_count = 0  # 用于早停

    # 开始训练
    for epoch in range(max_epochs):
        optimizer.zero_grad()
        total_loss = 0.0

        # 逐层（逐 key）处理
        # 对于每个参数张量，我们先 flatten，然后随机采样一部分来计算 Sinkhorn
        for k in state_dict_0.keys():
            # flatten
            p0 = state_dict_0[k].view(-1)
            p1 = state_dict_1[k].view(-1)

            # 采样索引
            length = p0.shape[0]
            if length <= chunk_size:
                # 如果该层参数量 < chunk_size，直接全部使用
                idx = torch.arange(length, device=p0.device)
            else:
                # 否则随机采样 chunk_size 个元素
                idx = torch.randperm(length, device=p0.device)[:chunk_size]

            # 取出采样后元素
            sample_p0 = p0[idx]
            sample_p1 = p1[idx]

            # 计算掩码 (sigmoid)，并同样只取采样位置
            mask_pre_sigmoid = torch.sigmoid(masks_pre[k].view(-1)[idx])
            mask_post_sigmoid = torch.sigmoid(masks_post[k].view(-1)[idx])

            # 计算 sinkhorn loss
            masked_tensor_0 = sample_p0 * mask_pre_sigmoid
            masked_tensor_1 = sample_p1 * mask_post_sigmoid
            ot_loss = sinkhorn_loss(
                masked_tensor_1.view(-1, 1),
                masked_tensor_0.view(-1, 1)
                
            )

            # 稀疏正则：对全部 mask 做 l1 范数
            # 这里也可以只在采样区间做近似，或在完整张量上做
            reg_pre = torch.norm(mask_pre_sigmoid, p=1)
            reg_post = torch.norm(mask_post_sigmoid, p=1)

            # 累加 loss
            total_loss += ot_loss + lmd * (reg_pre + reg_post)

        # 反向传播 + 更新
        total_loss.backward()
        optimizer.step()

        # early stopping 检查
        curr_loss_val = total_loss.item()
        if curr_loss_val < best_loss - 1e-6:  # 下降幅度大于阈值则认为有改进
            best_loss = curr_loss_val
            no_improve_count = 0
        else:
            no_improve_count += 1

        # 打印或记录日志
        if epoch % 10 == 0:
            print(f"[Epoch {epoch}] total_loss = {curr_loss_val:.4f} (best={best_loss:.4f})")

        # 如果超过 patience 次没有改进，则提前结束
        if no_improve_count >= early_stop_patience:
            print(f"Early stopping at epoch={epoch}")
            break

    # 最后将参数 sigmoid 化得到最终的掩码
    ot_masks_pre = {k: torch.sigmoid(v) for k, v in masks_pre.items()}
    ot_masks_post = {k: torch.sigmoid(v) for k, v in masks_post.items()}
    return ot_masks_pre, ot_masks_post

# 二阶段训练版本，但不使用早停机制
def compute_ot_mask(state_dict_0: StateDict, state_dict_1: StateDict, masks_pre: StateDict, masks_post: StateDict):
    # 确保所有 masks 是叶子张量
    for k in masks_pre.keys():
        masks_pre[k] = masks_pre[k].detach().requires_grad_(True)
        masks_post[k] = masks_post[k].detach().requires_grad_(True)

    # 初始化优化器
    optimizer_pre = Adam(masks_pre.values(), lr=1e-3)
    optimizer_post = Adam(masks_post.values(), lr=1e-3)
    sinkhorn_loss = SamplesLoss(loss="sinkhorn", blur=0.05)
    lmd = 0.01  # 正则化系数

    for epoch in range(1000):
        # Phase 1: Train all ot_mask_pre
        optimizer_pre.zero_grad()
        total_loss_pre = 0
        for k in state_dict_0.keys():
            mask_pre_sigmoid = torch.sigmoid(masks_pre[k])
            masked_tensor_0_pre = state_dict_0[k] * mask_pre_sigmoid
            masked_tensor_1_pre = state_dict_1[k] * mask_pre_sigmoid
            ot_loss_pre = sinkhorn_loss(masked_tensor_0_pre.view(-1, 1), masked_tensor_1_pre.view(-1, 1))
            loss_sparse_pre = torch.norm(mask_pre_sigmoid, p=1)
            total_loss_pre += ot_loss_pre + lmd * loss_sparse_pre
        total_loss_pre.backward()
        optimizer_pre.step()

        # Phase 2: Train all ot_mask_post
        optimizer_post.zero_grad()
        total_loss_post = 0
        for k in state_dict_0.keys():
            mask_post_sigmoid = torch.sigmoid(masks_post[k])
            masked_tensor_0_post = state_dict_0[k] * mask_post_sigmoid
            masked_tensor_1_post = state_dict_1[k] * mask_post_sigmoid
            ot_loss_post = sinkhorn_loss(masked_tensor_0_post.view(-1, 1), masked_tensor_1_post.view(-1, 1))
            loss_sparse_post = torch.norm(mask_post_sigmoid, p=1)
            total_loss_post += ot_loss_post + lmd * loss_sparse_post
        total_loss_post.backward()
        optimizer_post.step()

        # Logging
        if epoch % 100 == 0:
            print(f"Epoch: {epoch}, total_loss_pre: {total_loss_pre.item()}, total_loss_post: {total_loss_post.item()}")

    # 返回所有 mask 的 sigmoid 激活值
    ot_masks_pre = {k: torch.sigmoid(v) for k, v in masks_pre.items()}
    ot_masks_post = {k: torch.sigmoid(v) for k, v in masks_post.items()}
    return ot_masks_pre, ot_masks_post
 