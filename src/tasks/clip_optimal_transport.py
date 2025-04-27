import torch
from torch import nn, Tensor
from torch.optim import Adam
from geomloss import SamplesLoss

class ClipOptimalTransport(nn.Module):
    def __init__(self, state_dict_0: dict, state_dict_1: dict, reg: float = 0.01, n_iter: int = 10):
        super().__init__()
        self.reg = reg
        self.n_iter = n_iter
        self.masks = self._initialize_masks(state_dict_0)
        self.state_dict_0 = state_dict_0
        self.state_dict_1 = state_dict_1

    def _initialize_masks(self, state_dict: dict) -> dict:
        masks = {}
        for k, v in state_dict.items():
            masks[k] = nn.Parameter(torch.rand_like(v))
        return masks

    def compute_ot_mask(self, source_tensor_0: Tensor, source_tensor_1: Tensor, mask: Tensor) -> Tensor:
        optimizer = Adam([mask], lr=1e-3)
        sinkhorn_loss = SamplesLoss(loss="sinkhorn", blur=0.05)
        lmd = 0.01  # Regularization coefficient

        for epoch in range(1000):
            optimizer.zero_grad()
            mask_sigmoid = torch.sigmoid(mask)
            masked_tensor_0 = source_tensor_0 * mask_sigmoid
            masked_tensor_1 = source_tensor_1 * mask_sigmoid
            ot_loss = sinkhorn_loss(masked_tensor_0.view(-1, 1), masked_tensor_1.view(-1, 1))
            loss_sparse = torch.norm(mask_sigmoid, p=1)
            loss = ot_loss + lmd * loss_sparse
            loss.backward()
            optimizer.step()
            if epoch % 100 == 0:
                print(f'Epoch: {epoch}, loss: {loss.item()}')
        return mask_sigmoid

    def fusion(self, state_dict_0: dict, state_dict_1: dict) -> dict:
        merged_state_dict = {}
        for key, mask_param in self.masks.items():
            mask = torch.sigmoid(mask_param)
            merged_state_dict[key] = mask * (state_dict_0[key] + state_dict_1[key]) / 2
        return merged_state_dict

    def parameters(self, recurse: bool = True):
        return self.masks.values()

    def to(self, device):
        super().to(device)
        for mask in self.masks.values():
            mask.data = mask.data.to(device)
        return self