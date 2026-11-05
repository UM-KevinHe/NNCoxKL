import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F


class _Loss(torch.nn.Module):
    def __init__(self, reduction: str = 'mean') -> None:
        super().__init__()
        self.reduction = reduction


def _reduction(loss: Tensor, reduction: str = 'mean') -> Tensor:
    if reduction == 'none':
        return loss
    elif reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    raise ValueError(
        f"`reduction` = {reduction} is not valid. Use 'none', 'mean' or 'sum'.")


class CoxKLLOSS(_Loss):
    def forward(self, log_h: Tensor, time: Tensor, event: Tensor, 
                eta: Tensor, risk_scores: Tensor, 
                alpha: float = 1.0,
                ridge_lambda: float = 0.0,
                model: torch.nn.Module = None,
                eps: float = 1e-7,
                reduction: str = 'mean') -> Tensor:
        """Negative log-likelihood of the Cox KL model. 
        See `cox_kl_loss` for details.

        Arguments:
            eta (Tensor) -- The eta parameter of the Cox KL model, a lager eta means more weight on the external data.
        Returns:
            torch.tensor -- The negative log-likelihood loss.
        """
        return cox_kl_loss_breslow(
            log_h=log_h,
            time=time,
            event=event,
            eta=eta,
            risk_scores=risk_scores,
            alpha=alpha,
            ridge_lambda=ridge_lambda,
            model=model,
            eps=eps,
            reduction=reduction
        )
        
        # return cox_kl_loss_breslow_L2(log_h, time, event, eta, risk_scores, eps, reduction, lambda_rank=1.0, sigma_rank=1.0, rank_kind="exp")


def cox_kl_loss(log_h: Tensor, time: Tensor, event: Tensor, eta: Tensor, risk_scores: Tensor, eps: float = 1e-7, reduction: str = 'mean') -> Tensor:

    if event.dtype is torch.bool:
        event = event.float()

    # check if time is increasing
    if not torch.all(time[1:] >= time[:-1]):
        print("time is not increasing")
        print(f"time: {time}")
        raise ValueError("time is not increasing")

    log_h = log_h.view(-1)
    event = event.view(-1)
    # n = len(event)
    gamma = log_h.max()

    # eta = torch.from_numpy(np.array(eta)).float()

    if risk_scores is not None:
        risk_scores = risk_scores.view(-1)
    else:
        risk_scores = event.new_zeros(event.size())

    # risk_scores, log_h: [n]
    # event: [n]，其中要把 event==0 的行置零
    eps = 1e-12
    device = risk_scores.device
    dtype = risk_scores.dtype

    # --- 构造 P、Q 的“后缀归一化” ---
    # 先稳定化再 exp
    num_p = torch.exp((risk_scores - risk_scores.max()
                       ).to(dtype))             # [n]
    num_q = torch.exp((log_h - log_h.max()).to(dtype))                   # [n]

    # 后缀和（从右往左累加）
    den_p = torch.flip(torch.cumsum(
        torch.flip(num_p, (0,)), dim=0), (0,))     # [n]
    den_q = torch.flip(torch.cumsum(
        torch.flip(num_q, (0,)), dim=0), (0,))     # [n]

    # 形成 [n, n]：第 i 行按 j>=i 的尾部和做归一化
    P = num_p.view(1, -1).expand(num_p.numel(), -1) / \
        (den_p.view(-1, 1) + eps)  # [n,n]
    Q = num_q.view(1, -1).expand(num_q.numel(), -1) / \
        (den_q.view(-1, 1) + eps)  # [n,n]

    # --- 向量化 KL over 上三角（含对角）---
    n = P.size(0)
    tri = torch.triu(torch.ones(n, n, dtype=torch.bool,
                     device=device))         # 上三角掩码
    # PQ_ratio_log = (P.clamp_min(eps) / Q.clamp_min(eps)).log()  
    PQ_ratio_log = -Q.log()
    kl = (P.clamp_min(eps) * PQ_ratio_log).masked_fill(~tri,0).sum(dim=1)      # [n]

    # 只保留 event==1 的行
    kl = kl.to(torch.float32)
    kl = kl.masked_fill(event == 0, 0.0)

    log_h_flip = torch.flip(log_h, dims=(0,))
    log_cumsum_h = torch.flip(log_h_flip.sub(gamma).exp().cumsum(
        0).add(eps).log().add(gamma), dims=(0,))
    # llk = -((event + eta*risk_scores).mul(log_h).div(1+eta).sub(log_cumsum_h.mul(event)))
    llk = (-(event.mul(log_h).sub(log_cumsum_h.mul(event))) + eta * kl) / (1+eta)

    # check if any llk is nan or negative number
    # if torch.isnan(llk).any() or (llk < 0).any():
    if llk.mean() < 0:
        print("llk is nan or negative number")
        print(f"llk mean: {llk.mean()}")
        print(f"log_h range: {log_h.min()}, {log_h.max()}")
        print(
            f"log_cumsum_h range: {log_cumsum_h.min()}, {log_cumsum_h.max()}")
        print(
            f"(event + eta*risk_scores) range: {(event + eta*risk_scores).min()}, {(event + eta*risk_scores).max()}")
        print(f"event number: {event.sum()}")
        print(
            f"range of log_cumsum_h.mul(event): {log_cumsum_h.mul(event).min()}, {log_cumsum_h.mul(event).max()}")
        print(f"eta: {eta}")
        print(f"gamma: {gamma}")

    return _reduction(llk, reduction)

def cox_kl_loss_breslow(log_h: Tensor, time: Tensor, event: Tensor, eta: Tensor, risk_scores: Tensor,
                        alpha: float = 1.0, ridge_lambda: float = 0.0, model: torch.nn.Module = None,
                        eps: float = 1e-7, reduction: str = 'mean') -> Tensor:
    if ridge_lambda > 0.0 and model is None:
        raise ValueError(
            f"ridge_lambda={ridge_lambda} was specified but model is None. "
            "Pass the model so ridge penalty can be computed, or set ridge_lambda=0.0."
        )

    batch_size = log_h.shape[0]
    if event.dtype is torch.bool:
        event = event.float()

    # check if time is increasing
    if not torch.all(time[1:] >= time[:-1]):
        print("time is not increasing")
        print(f"time: {time}")
        raise ValueError("time is not increasing")

    log_h = log_h.view(-1)
    event = event.view(-1)
    # n = len(event)
    gamma = log_h.max()

    # eta = torch.from_numpy(np.array(eta)).float()

    if risk_scores is not None:
        risk_scores = risk_scores.view(-1)
    else:
        risk_scores = event.new_zeros(event.size())

    # risk_scores, log_h: [n]
    # event: [n]，其中要把 event==0 的行置零
    eps = 1e-12
    device = risk_scores.device
    dtype = risk_scores.dtype

    # ---- 风险集 log Σexp(η)：用 logcumsumexp 从末尾做后缀和 ----
    # 对于升序时间，位置 i 的后缀就是 R_t：{j: t_j >= t_i}
    log_risk_per_row = torch.logcumsumexp(log_h.flip(
        0), dim=0).flip(0)  # = log Σ_{j∈R_i} e^{η_j}

    # ---- 把相同时间点合并（Breslow）----
    # 取连续唯一时间点的计数，得到每组的起始下标
    _, counts = torch.unique_consecutive(time, return_counts=True)
    group_starts = torch.cat(
        [counts.new_zeros(1), counts.cumsum(0)[:-1]]
    ).long()
    G = counts.numel()  # 组数
    group_ids = torch.repeat_interleave(
        torch.arange(G, device=time.device), counts)

    # 组内求和：∑_{i∈D_t}η_i 以及 d_t=∑_{i∈D_t}1
    eta_event_sum = torch.zeros(G, dtype=log_h.dtype, device=log_h.device)
    d_t = torch.zeros(G, dtype=log_h.dtype, device=log_h.device)
    eta_event_sum.scatter_add_(0, group_ids, event * log_h)
    d_t.scatter_add_(0, group_ids, event)

    # 每组的 log Σ_{j∈R_t} e^{η_j} 取该组起始行的值即可
    log_risk_t = log_risk_per_row[group_starts] + eps * 0.0  # 保持梯度/设备

    # ---- Breslow 目标：负对数似然 ----
    nll_per_group = -(eta_event_sum - d_t * log_risk_t)
    lik = nll_per_group.sum()

    # --- APPLY TEMPERATURE SCALING HERE ---
    # Scale the external risk scores by alpha. 
    # If alpha > 1, distribution becomes sharper (more confident).
    # If 0 < alpha < 1, distribution becomes flatter (softer).
    scaled_risk_scores = risk_scores * alpha

    delta_max = scaled_risk_scores.max() # Recalculate max for stability
    logh_max = log_h.max()

    # Numerators (P is teacher, Q is student)
    num_p = torch.exp((scaled_risk_scores - delta_max).to(dtype))
    num_q = torch.exp((log_h - logh_max).to(dtype))

    # Denominators (Cumulative sums from bottom up)
    den_p = torch.flip(torch.cumsum(torch.flip(num_p, (0,)), dim=0), (0,))
    den_q = torch.flip(torch.cumsum(torch.flip(num_q, (0,)), dim=0), (0,))

    # --- REMOVED: Entropy of P (S1 / term t1) calculations ---
    # We only need the Cross Entropy part: - Sum(P * log Q)
    
    log_num_q = (log_h - logh_max).to(dtype) # log(num_q)
    
    # t2 = num_p * log_num_q  => This is the unnormalized P * log(unnormalized Q)
    t2 = num_p * log_num_q
    
    # Suffix sum of t2
    suf_t2 = torch.flip(torch.cumsum(torch.flip(t2, (0,)), dim=0), (0,))

    # Extract values at group starts (Risk sets)
    den_p_g = den_p[group_starts] + eps
    den_q_g = den_q[group_starts] + eps
    S2_g = suf_t2[group_starts]   # Sum_{j>=s} num_p_j * log num_q_j

    # --- SIMPLIFIED FORMULA (Cross Entropy Loss) ---
    # Original KL: (S1_g - S2_g)/den_p_g - log(den_p_g) + log(den_q_g)
    #
    # 1. (S1_g / den_p_g) depends only on Teacher (Constant) -> REMOVED
    # 2. (- log(den_p_g)) depends only on Teacher (Constant) -> REMOVED
    # 3. Remaining terms depend on Student (log_h) -> KEPT
    
    kl_g_proxy = - (S2_g / den_p_g) + den_q_g.log()

    # Weight by number of events
    kl = (kl_g_proxy * d_t).sum()

    # Combine losses
    lik = (lik + eta * kl) / (1 + eta)

    if lik.mean() < 0:
        print("llk is nan or negative number")
        print(f"llk mean: {lik.mean()}")

    base_loss = lik / batch_size

    if ridge_lambda > 0.0 and model is not None:
        ridge_penalty = torch.zeros(1, dtype=base_loss.dtype, device=base_loss.device)
        for param in model.parameters():
            if param.requires_grad:
                ridge_penalty = ridge_penalty + param.pow(2).sum()
        base_loss = base_loss + ridge_lambda * ridge_penalty

    return base_loss


def pairwise_ranking_L2(
    log_h: Tensor, time: Tensor, event: Tensor,
    sigma: float = 1.0,
    kind: str = "exp",   # "exp" = DeepHit式, "logistic" = softplus
) -> Tensor:
    """
    成对排序损失 L2：
      Pairs = {(i,j): event[i]==1, time[i] < time[j]}
      loss  = mean_{(i,j)} phi( log_h[i] - log_h[j] )
      其中 phi(z) = exp(-z/sigma) 或 softplus(-z/sigma).

    复杂度 O(n^2)，n 很大时可以做分块/负采样。
    """
    log_h = log_h.view(-1)
    time = time.view(-1)
    event = event.view(-1).float()

    # 形成两两差分: z_ij = log_h[i] - log_h[j]
    zi = log_h.unsqueeze(1)   # [n,1]
    zj = log_h.unsqueeze(0)   # [1,n]
    z = zi - zj              # [n,n]

    ti = time.unsqueeze(1)
    tj = time.unsqueeze(0)
    ei = event.unsqueeze(1)

    # 可比较对的掩码：i 是事件且 t_i < t_j
    M = (ei == 1.0) & (ti < tj)

    if kind == "exp":
        # DeepHit式: exp(-(x-y)/sigma)
        phi = torch.exp(-z / sigma)
    elif kind == "logistic":
        # pairwise logistic: log(1 + exp(-(x-y)/sigma))
        phi = F.softplus(-z / sigma)
    else:
        raise ValueError("kind must be 'exp' or 'logistic'.")

    loss_sum = (phi * M).sum()
    num_pairs = M.sum().clamp_min(1)
    return loss_sum / num_pairs


def cox_kl_loss_breslow_L2(log_h: Tensor, time: Tensor, event: Tensor, eta: Tensor, risk_scores: Tensor, eps: float = 1e-7, reduction: str = 'mean',
                           lambda_rank: float = 0.0,     # 排序权重 λ（默认0 = 不启用）
                           sigma_rank: float = 1.0,      # 排序温度 σ
                           rank_kind: str = "exp") -> Tensor:
    batch_size = log_h.shape[0]
    if event.dtype is torch.bool:
        event = event.float()

    # check if time is increasing
    if not torch.all(time[1:] >= time[:-1]):
        print("time is not increasing")
        print(f"time: {time}")
        raise ValueError("time is not increasing")

    log_h = log_h.view(-1)
    event = event.view(-1)
    # n = len(event)
    gamma = log_h.max()

    # eta = torch.from_numpy(np.array(eta)).float()

    if risk_scores is not None:
        risk_scores = risk_scores.view(-1)
    else:
        risk_scores = event.new_zeros(event.size())

    # risk_scores, log_h: [n]
    # event: [n]，其中要把 event==0 的行置零
    eps = 1e-12
    device = risk_scores.device
    dtype = risk_scores.dtype

    # # --- 构造 P、Q 的“后缀归一化” ---
    # # 先稳定化再 exp
    # num_p = torch.exp((risk_scores - risk_scores.max()).to(dtype))             # [n]
    # num_q = torch.exp((log_h       - log_h.max()).to(dtype))                   # [n]

    # # 后缀和（从右往左累加）
    # den_p = torch.flip(torch.cumsum(torch.flip(num_p, (0,)), dim=0), (0,))     # [n]
    # den_q = torch.flip(torch.cumsum(torch.flip(num_q, (0,)), dim=0), (0,))     # [n]

    # # 形成 [n, n]：第 i 行按 j>=i 的尾部和做归一化
    # P = num_p.view(1, -1).expand(num_p.numel(), -1) / (den_p.view(-1, 1) + eps)  # [n,n]
    # Q = num_q.view(1, -1).expand(num_q.numel(), -1) / (den_q.view(-1, 1) + eps)  # [n,n]

    # # --- 向量化 KL over 上三角（含对角）---
    # n = P.size(0)
    # tri = torch.triu(torch.ones(n, n, dtype=torch.bool, device=device))         # 上三角掩码
    # PQ_ratio_log = (P.clamp_min(eps) / Q.clamp_min(eps)).log()                  # [n,n]
    # kl = (P.clamp_min(eps) * PQ_ratio_log).masked_fill(~tri, 0).sum(dim=1)      # [n]

    # # 只保留 event==1 的行
    # kl = kl.to(torch.float32)
    # kl = kl.masked_fill(event == 0, 0.0)
    # kl = kl.sum()

    # ---- 风险集 log Σexp(η)：用 logcumsumexp 从末尾做后缀和 ----
    # 对于升序时间，位置 i 的后缀就是 R_t：{j: t_j >= t_i}
    log_risk_per_row = torch.logcumsumexp(log_h.flip(
        0), dim=0).flip(0)  # = log Σ_{j∈R_i} e^{η_j}

    # ---- 把相同时间点合并（Breslow）----
    # 取连续唯一时间点的计数，得到每组的起始下标
    _, counts = torch.unique_consecutive(time, return_counts=True)
    group_starts = torch.cat(
        [counts.new_zeros(1), counts.cumsum(0)[:-1]]
    ).long()
    G = counts.numel()  # 组数
    group_ids = torch.repeat_interleave(
        torch.arange(G, device=time.device), counts)

    # 组内求和：∑_{i∈D_t}η_i 以及 d_t=∑_{i∈D_t}1
    eta_event_sum = torch.zeros(G, dtype=log_h.dtype, device=log_h.device)
    d_t = torch.zeros(G, dtype=log_h.dtype, device=log_h.device)
    eta_event_sum.scatter_add_(0, group_ids, event * log_h)
    d_t.scatter_add_(0, group_ids, event)

    # 每组的 log Σ_{j∈R_t} e^{η_j} 取该组起始行的值即可
    log_risk_t = log_risk_per_row[group_starts] + eps * 0.0  # 保持梯度/设备

    # ---- Breslow 目标：负对数似然 ----
    nll_per_group = -(eta_event_sum - d_t * log_risk_t)
    lik = nll_per_group.sum()

    # ---------- KL 的 Breslow 版本 ----------
    # 先做稳定化的“分子”与后缀“分母”（与原实现一致，但只在组起点处取值）
    # P ~ teacher: risk_scores，Q ~ model: log_h
    delta_max = risk_scores.max()
    logh_max = log_h.max()

    num_p = torch.exp((risk_scores - delta_max).to(dtype))     # [n]
    num_q = torch.exp((log_h - logh_max).to(dtype))     # [n]

    # 后缀和（作为每个 i 的风险集分母）
    den_p = torch.flip(torch.cumsum(
        torch.flip(num_p, (0,)), dim=0), (0,))     # [n]
    den_q = torch.flip(torch.cumsum(
        torch.flip(num_q, (0,)), dim=0), (0,))     # [n]

    # 需要 Σ_{j≥s} num_p_j*log num_p_j 与 Σ_{j≥s} num_p_j*log num_q_j 的后缀和
    log_num_p = (risk_scores - delta_max).to(dtype)  # = log(num_p)
    log_num_q = (log_h - logh_max).to(dtype)  # = log(num_q)
    t1 = num_p * log_num_p                           # [n]
    t2 = num_p * log_num_q                           # [n]
    suf_t1 = torch.flip(torch.cumsum(
        torch.flip(t1, (0,)), dim=0), (0,))       # [n]
    suf_t2 = torch.flip(torch.cumsum(
        torch.flip(t2, (0,)), dim=0), (0,))       # [n]

    # 取每组起点（统一风险集 R(t_g)）
    den_p_g = den_p[group_starts] + eps
    den_q_g = den_q[group_starts] + eps
    S1_g = suf_t1[group_starts]   # Σ_{j≥s} num_p_j * log num_p_j
    S2_g = suf_t2[group_starts]   # Σ_{j≥s} num_p_j * log num_q_j

    # KL_g = (S1_g - S2_g)/den_p_g - log(den_p_g) + log(den_q_g)
    kl_g = (S1_g - S2_g) / den_p_g - den_p_g.log() + den_q_g.log()

    # 用并发事件数 d_t 加权（d_t=0 的组自然不贡献）
    kl = (kl_g * d_t).sum()

    
    
    # ===== L2 排序项（可选）=====
    if lambda_rank > 0:
        l2 = pairwise_ranking_L2(
            log_h=log_h, time=time, event=event,
            sigma=sigma_rank, kind=rank_kind
        )
        lik = (lik/100 + eta * kl) / (1+eta) + lambda_rank * l2
    else:
        lik = (lik/100 + eta * kl) / (1+eta)
        
    # print(f"lik: {nll_per_group.sum().item():.3f}, kl: {kl.item():.3f}, l2: {l2.item():.3f}")

    # check if any llk is nan or negative number
    # if torch.isnan(llk).any() or (llk < 0).any():
    if lik.mean() < 0:
        print("llk is nan or negative number")
        print(f"llk mean: {lik.mean()}")
        print(f"log_h range: {log_h.min()}, {log_h.max()}")

    return lik / batch_size
