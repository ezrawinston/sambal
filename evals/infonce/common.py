
import torch, numpy as np

def pool_wordpieces_mean(H_full: torch.Tensor, groups):
    """
    H_full: [T_sub, D] tensor
    groups: list[list[int]] word -> list of subword indices
    returns H_words: [N, D] mean over subwords
    """
    rows = []
    for g in groups:
        idx = torch.tensor(g, dtype=torch.long, device=H_full.device)
        rows.append(H_full.index_select(0, idx).mean(dim=0))
    return torch.stack(rows, dim=0)  # [N,D]

@torch.no_grad()
def infonce_multi_positive(features: torch.Tensor,
                           groups: torch.Tensor,
                           tau: float = 0.1,
                           batch_size: int = 2048,
                           normalize: bool = True) -> float:
    """
    Compute multi-positive InfoNCE loss over all anchors.
    features: [N,D] (float32/float64) on device (CPU or CUDA)
    groups:   [N] int64 group ids (same device)
    tau: temperature
    Returns mean loss (lower is better). For reporting as "higher is better",
    you can use -(loss).
    """
    assert features.size(0) == groups.size(0)
    F = features
    if normalize:
        F = F / (F.norm(dim=1, keepdim=True) + 1e-12)
    N = F.size(0)
    total_loss = 0.0
    total_count = 0
    for start in range(0, N, batch_size):
        stop = min(N, start + batch_size)
        Q = F[start:stop]  # [b,D]
        sim = (Q @ F.T) / tau  # [b,N]

        # mask out self
        idx = torch.arange(start, stop, device=F.device)
        sim[torch.arange(stop - start, device=F.device), idx] = -1e9

        gQ = groups[start:stop].unsqueeze(1)  # [b,1]
        pos_mask = (gQ == groups.unsqueeze(0))  # [b,N]
        pos_mask[torch.arange(stop - start, device=F.device), idx] = False

        # numerical stability
        m = sim.max(dim=1, keepdim=True).values
        exp_sim = torch.exp(sim - m)
        pos_exp = (exp_sim * pos_mask).sum(dim=1)
        all_exp = exp_sim.sum(dim=1) + 1e-12

        valid = pos_exp > 0
        if valid.any():
            loss = -torch.log(pos_exp[valid] / all_exp[valid]).sum()
            total_loss += loss.item()
            total_count += int(valid.sum().item())

    return total_loss / max(1, total_count)
