import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch import Tensor


class VAEQuantizer(nn.Module):
    """
    Continuous alternative to VQ: a lightweight VAE bottleneck.

    Interface-compatible with VQ/NSVQ/EMAVQ used by `LatentLAMModel`:
      - forward(...) returns (quantized, perplexity, indices, entropy_loss, vq_loss)
      - inference(...) returns (quantized, indices[, distances/logits/probs when requested])

    Notes:
      - `vq_loss` is the KL divergence loss (optionally weighted by `beta`)
      - `perplexity`, `indices`, `entropy_loss` are not applicable and returned as zeros / None
      - Accepts and ignores extra kwargs so existing `vq_kwargs` configs won't break.
      - Now operates directly in code_dim space (projection layers moved to encoder/decoder).
    """

    def __init__(
        self,
        code_dim: int = 64,
        beta: float = 5e-5,
        clamp_logvar: float | None = 10.0,
        layer_norm: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.code_dim = int(code_dim)
        self.beta = float(beta)
        self.clamp_logvar = float(clamp_logvar) if clamp_logvar is not None else None

        self.pre_norm = nn.LayerNorm(self.code_dim) if layer_norm else nn.Identity()
        self.mu = nn.Linear(self.code_dim, self.code_dim)
        self.logvar = nn.Linear(self.code_dim, self.code_dim)

        # for logging parity with VQ modules
        self.last_kl_loss: Tensor | None = None
        # mutual information estimate (see forward for details)
        self.last_mutual_info: Tensor | None = None
        # KL(q(z) || p(z)) term used in MI computation
        self.last_qz_kl: Tensor | None = None

    def _encode(self, nodes: Tensor) -> tuple[Tensor, Tensor]:
        # Expect nodes: [B, Q, D] or [B, D]; average over query dim to a single latent
        if nodes.dim() == 2:
            nodes = nodes.unsqueeze(1)  # [B, 1, D]
        nodes_pooled = nodes.mean(dim=1, keepdim=True)  # [B, 1, D]
        h = self.pre_norm(nodes_pooled)
        mu = self.mu(h)
        logvar = self.logvar(h)
        if self.clamp_logvar is not None:
            logvar = torch.clamp(logvar, min=-self.clamp_logvar, max=self.clamp_logvar)
        return mu, logvar

    @staticmethod
    def _kl_divergence(mu: Tensor, logvar: Tensor) -> Tensor:
        # KL(q(z|x) || N(0, I)) = 0.5 * sum(mu^2 + exp(logvar) - 1 - logvar)
        kl = 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar)
        return kl.sum(dim=-1)  # [...], sum over latent dim

    def forward(self, nodes: Tensor) -> tuple[Tensor, Tensor, Tensor | None, Tensor, Tensor, Tensor, Tensor]:
        mu, logvar = self._encode(nodes)
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        z = mu + eps * std  # reparameterized sample
        quantized = z

        # mean KL across batch/query positions (standard VAE objective)
        kl_per_sample = 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar).sum(dim=-1)  # [...]
        kl_loss = kl_per_sample.mean()
        self.last_kl_loss = kl_loss.detach()

        perplexity = nodes.new_tensor(0.0)
        indices = None
        entropy_loss = nodes.new_tensor(0.0)
        vq_loss = kl_loss * self.beta
        return quantized, perplexity, indices, entropy_loss, vq_loss

    @torch.no_grad()
    def inference(
        self,
        nodes: Tensor,
        user_specific=None,
        return_distance: bool = False,
        return_logits: bool = False,
        return_probs: bool = False,
        temperature: float = 1.0,
        sample: bool = False,
        return_stats: bool = False,
    ):
        # Deterministic by default: use mean. Optional sampling for analysis.
        mu, logvar = self._encode(nodes)
        if sample:
            std = (0.5 * logvar).exp() * float(temperature)
            z = mu + torch.randn_like(std) * std
        else:
            z = mu
        quantized = z
        indices = None
        if return_distance or return_logits or return_probs:
            return quantized, indices, None, None, None
        if return_stats:
            return quantized, indices, mu, logvar
        return quantized, indices


class AEQuantizer(nn.Module):
    """
    Simple linear bottleneck used when vq_type='ae'.
    - Keeps interface compatible with VQ/VAE modules.
    - Now operates directly in code_dim space (projection layers moved to encoder/decoder).
    """

    def __init__(
        self,
        code_dim: int = 128,
        layer_norm: bool = False,
        codebook_size: int | None = None,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.code_dim = int(code_dim)
        # Align with VQ API: expose codebook_size for downstream components
        self.codebook_size = int(codebook_size) if codebook_size is not None else int(code_dim)
        self.pre_norm = nn.LayerNorm(self.code_dim) if layer_norm else nn.Identity()
        self.last_nodes_norm: Tensor | None = None
        # Keep parity with VQ modules that expose this attribute
        self.nodes_norm: Tensor | None = None

    def forward(self, nodes: Tensor):
        # nodes: [B, Q, D] or [B, D]
        nodes_proj = self.pre_norm(nodes)
        with torch.no_grad():
            norm_val = torch.norm(nodes_proj, p=2, dim=-1).mean()
            self.last_nodes_norm = norm_val
            self.nodes_norm = norm_val
        quantized = nodes_proj
        batch = nodes.shape[0]
        num_queries = nodes.shape[1] if nodes.dim() > 1 else 1
        indices = torch.zeros((batch, num_queries), device=nodes.device, dtype=torch.long)
        zero_scalar = nodes.new_tensor(0.0)
        perplexity = zero_scalar
        entropy_loss = zero_scalar
        vq_loss = zero_scalar
        return quantized, perplexity, indices, entropy_loss, vq_loss

    @torch.no_grad()
    def inference(
        self,
        nodes: Tensor,
        user_specific=None,
        return_distance: bool = False,
        return_logits: bool = False,
        return_probs: bool = False,
        temperature: float = 1.0,
        *args,
        **kwargs,
    ):
        nodes_proj = self.pre_norm(nodes)
        quantized = nodes_proj
        norm_val = torch.norm(nodes_proj, p=2, dim=-1).mean()
        self.last_nodes_norm = norm_val
        self.nodes_norm = norm_val
        batch = nodes.shape[0]
        num_queries = nodes.shape[1] if nodes.dim() > 1 else 1
        indices = torch.zeros((batch, num_queries), device=nodes.device, dtype=torch.long)
        if return_distance or return_logits or return_probs:
            return quantized, indices, None, None, None
        return quantized, indices


class VQ(nn.Module):
    def __init__(
        self,
        codebook_size: int = 1024,
        code_dim: int = 128,
        discarding_threshold: float = 0.01,
        initialization: str = "uniform",
        data_dependent_init: bool = True,
        kmeans_iters: int = 10,
        kmeans_max_samples: int = 100000,
        beta: float = 0.25,
        orthogonal_loss_weight: float = 0.0,
        lambda_sample_entropy: float = 0.0,
        lambda_codebook_entropy: float = 0.0,
        max_code_replaced_per_step: int = 2,
        use_cosine_sim: bool = False,
        layer_norm: bool = False,
        scale: float = 1.0,
        use_soft_assignment: bool = False,
        use_temperature_schedule: bool = False,
        temperature_start: float = 1.0,
        temperature_end: float = 0.1,
        temperature_decay_steps: int = 10000,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.code_dim = code_dim
        self.discarding_threshold = discarding_threshold
        self.eps = 1e-12
        self.data_dependent_init = bool(data_dependent_init)
        self.kmeans_iters = int(kmeans_iters)
        self.kmeans_max_samples = int(kmeans_max_samples)
        self.beta = beta
        self.orthogonal_loss_weight = float(orthogonal_loss_weight)
        self.lambda_sample_entropy = float(lambda_sample_entropy)
        self.lambda_codebook_entropy = float(lambda_codebook_entropy)
        self.max_code_replaced_per_step = int(max_code_replaced_per_step)
        self.use_soft_assignment = bool(use_soft_assignment)
        self.use_temperature_schedule = bool(use_temperature_schedule)
        self.temperature_start = float(temperature_start)
        self.temperature_end = float(temperature_end)
        self.temperature_decay_steps = int(temperature_decay_steps)
        self.last_slot_inter_redundancy = None
        self.last_slot_inner_redundancy = None
        self.scale = scale
        if self.use_temperature_schedule:
            self.register_buffer("temperature_step", torch.tensor(0, dtype=torch.long))
            self.register_buffer("last_temperature", torch.tensor(float(self.temperature_start)))
        else:
            self.temperature_step = None
            self.last_temperature = torch.tensor(float(self.temperature_start))
        if initialization == "normal":
            codebooks_data = torch.randn(self.codebook_size, self.code_dim)
        elif initialization == "uniform":
            codebooks_data = torch.empty(self.codebook_size, self.code_dim)
            nn.init.uniform_(codebooks_data, -1 / self.codebook_size, 1 / self.codebook_size)
        else:
            raise ValueError("initialization must be either 'normal' or 'uniform'")

        self.codebooks = nn.Parameter(codebooks_data)
        self.use_cosine_sim = use_cosine_sim
        self.in_norm = (
            nn.LayerNorm(self.code_dim, elementwise_affine=not self.use_cosine_sim)
            if (layer_norm or self.use_cosine_sim)
            else nn.Identity()
        )
        self.register_buffer("node_count", torch.zeros(self.codebook_size, dtype=torch.long))
        # self.register_buffer('kmeans_initialized', torch.tensor(0, dtype=torch.bool))
        # self.initialized = torch.tensor(0, dtype=torch.bool)
        self.register_buffer("initialized", torch.tensor(0, dtype=torch.bool))
        if not self.data_dependent_init:
            self.initialized.fill_(True)

    def _norm_nodes(self, nodes: Tensor) -> Tensor:
        return F.normalize(nodes, p=2, dim=-1, eps=self.eps)

    def get_nodes_proj(self, nodes: Tensor) -> Tensor:
        return self.in_norm(nodes)

    def _get_indices(self, nodes: Tensor) -> Tensor:
        distances = self._compute_distance(nodes, self.codebooks)
        return torch.argmin(distances, dim=-1)

    def _compute_distance(
        self,
        inputs: Tensor,
        codebook: Tensor,
        return_normed: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor, Tensor]:
        if self.use_cosine_sim:
            inputs_normed = F.normalize(inputs, dim=-1, eps=self.eps)
            code_normed = F.normalize(codebook, dim=-1, eps=self.eps)
            sim = inputs_normed @ code_normed.t()
            dist = -sim
            return (dist, inputs_normed, code_normed) if return_normed else dist
        inputs_norm = (inputs * inputs).sum(dim=-1, keepdim=True)
        code_norm = (codebook * codebook).sum(dim=-1)
        dist = inputs_norm + code_norm - 2.0 * inputs @ codebook.t()
        dist = torch.clamp(dist, min=0.0)
        return (dist, inputs, codebook) if return_normed else dist

    @torch.no_grad()
    def _update_slot_redundancy_metrics(self, min_indices: Tensor) -> None:
        if min_indices.dim() == 1:
            B, Q = min_indices.shape[0], 1
            idx_codes = min_indices.view(B, 1)
        else:
            B, Q = min_indices.shape[0], min_indices.shape[1]
            idx_codes = min_indices

        if B == 0 or Q == 0:
            self.last_slot_inter_redundancy = None
            self.last_slot_inner_redundancy = None
            return

        K = self.codebook_size
        one_hot = F.one_hot(idx_codes, num_classes=K).to(torch.float32)

        #    unique_per_slot: [Q]
        slot_code_counts = one_hot.sum(dim=0)  # [Q, K]
        unique_per_slot = (slot_code_counts > 0).sum(dim=-1)  # [Q]
        dup_per_slot = 1.0 - unique_per_slot.to(torch.float32) / float(B)  # [Q]
        self.last_slot_inner_redundancy = dup_per_slot.mean()

        #    unique_per_sample: [B]
        sample_code_counts = one_hot.sum(dim=1)  # [B, K]
        unique_per_sample = (sample_code_counts > 0).sum(dim=-1)  # [B]
        dup_per_sample = 1.0 - unique_per_sample.to(torch.float32) / float(Q)  # [B]
        self.last_slot_inter_redundancy = dup_per_sample.mean()

    @torch.no_grad()
    def kmeans_init(
        self,
        nodes: Tensor,
        num_iters: int | None = None,
        max_samples: int | None = None,
    ) -> None:
        nodes = nodes.reshape(-1, self.code_dim)
        nodes = nodes.contiguous()
        if nodes.numel() == 0:
            return

        num_iters = int(num_iters) if num_iters is not None else int(self.kmeans_iters)
        max_samples = int(max_samples) if max_samples is not None else int(self.kmeans_max_samples)

        ddp_available = dist.is_available() and dist.is_initialized()
        if ddp_available:
            world_size = dist.get_world_size()
            gathered = [torch.zeros_like(nodes) for _ in range(world_size)]
            dist.all_gather(gathered, nodes)
            all_nodes = torch.cat(gathered, dim=0)
            rank = dist.get_rank()
        else:
            all_nodes = nodes
            rank = 0

        if all_nodes.shape[0] > max_samples:
            perm = torch.randperm(all_nodes.shape[0], device=all_nodes.device)
            all_nodes = all_nodes[perm[:max_samples]]

        K = self.codebook_size
        N = all_nodes.shape[0]

        if N == 0:
            return

        if N < K:
            extra_indices = torch.randint(0, N, (K - N,), device=all_nodes.device)
            all_nodes = torch.cat([all_nodes, all_nodes[extra_indices]], dim=0)
            N = all_nodes.shape[0]

        if rank == 0:
            print("=" * 50, "\n")
            print("Starting VQ process...")
            print("KMeans initializing codebooks... \n")
            print("=" * 50, "\n")
            init_perm = torch.randperm(N, device=all_nodes.device)
            centers = all_nodes[init_perm[:K]]  # [K, D]

            for _ in range(num_iters):
                nodes_norm = (all_nodes * all_nodes).sum(dim=1, keepdim=True)  # [N, 1]
                centers_norm = (centers * centers).sum(dim=1)  # [K]
                dist2 = nodes_norm + centers_norm.unsqueeze(0) - 2.0 * all_nodes @ centers.t()
                dist2 = torch.clamp(dist2, min=0.0)
                assignment = torch.argmin(dist2, dim=1)  # [N]

                one_hot = F.one_hot(assignment, num_classes=K).to(all_nodes.dtype)  # [N, K]
                counts = one_hot.sum(dim=0).clamp(min=1.0).unsqueeze(-1)  # [K, 1]
                centers = (one_hot.t() @ all_nodes) / counts  # [K, D]

        if rank == 0:
            centers_converted = centers.to(device=self.codebooks.data.device, dtype=self.codebooks.data.dtype)
            if self.use_cosine_sim:
                centers_converted = F.normalize(centers_converted, dim=-1, eps=self.eps)
            self.codebooks.data.copy_(centers_converted)

        if ddp_available:
            dist.broadcast(self.codebooks.data, src=0)

        self.initialized.fill_(True)

        self.reset_node_count()

    @torch.no_grad()
    def _maybe_data_dependent_init(self, nodes_code: Tensor) -> None:
        if not self.data_dependent_init:
            return
        if bool(self.initialized.item()):
            return
        self.kmeans_init(nodes_code)

    def entropy_loss(
        self,
        affinity: Tensor,
        temperature: float | Tensor | None = None,
    ) -> Tensor:
        assert affinity.dim() == 3, "affinity must be [B, T, K]"

        if temperature is None:
            temperature = 0.1 if self.use_cosine_sim else 1.0
        logits = affinity / temperature

        probs = F.softmax(logits, dim=-1)
        probs = probs.clamp(min=self.eps, max=1.0 - self.eps)

        # ----------------------------------------------------
        # ----------------------------------------------------
        per_sample_probs = probs.mean(dim=1).clamp(min=self.eps, max=1.0 - self.eps)  # [B, K]
        per_sample_log_probs = torch.log(per_sample_probs)
        per_sample_entropy = -torch.mean(torch.sum(per_sample_probs * per_sample_log_probs, dim=-1))

        # ----------------------------------------------------
        # ----------------------------------------------------
        avg_prob = per_sample_probs.mean(dim=0)  # [K]
        avg_prob = avg_prob.clamp(min=self.eps, max=1.0 - self.eps)
        codebook_entropy = -torch.sum(avg_prob * torch.log(avg_prob))

        # ----------------------------------------------------
        # ----------------------------------------------------
        self.last_sample_entropy = per_sample_entropy.detach()
        self.last_codebook_entropy = codebook_entropy.detach()

        entropy_aux_loss = (
            self.lambda_sample_entropy * per_sample_entropy - self.lambda_codebook_entropy * codebook_entropy
        )
        return entropy_aux_loss

    @torch.no_grad()
    def _compute_batch_perplexity(self, min_indices: Tensor) -> Tensor:
        if min_indices.numel() == 0:
            return torch.tensor(0.0, device=self.codebooks.device)
        counts = torch.bincount(min_indices.reshape(-1), minlength=self.codebook_size).to(
            self.codebooks.device
        )
        total = counts.sum()
        if total.item() == 0:
            return torch.tensor(0.0, device=self.codebooks.device)
        probs = (counts.float() / total.float()).clamp(min=self.eps, max=1.0)
        return torch.exp(-torch.sum(probs * torch.log(probs)))

    @torch.no_grad()
    def _compute_avg_unique_codes(self, min_indices: Tensor) -> Tensor:
        if min_indices.numel() == 0:
            avg_unique = torch.tensor(0.0, device=self.codebooks.device)
            self.last_avg_unique_codes = avg_unique
            return avg_unique

        batch_size = min_indices.shape[0]
        flat_indices = min_indices.view(batch_size, -1)  # [B, L]

        one_hot = F.one_hot(flat_indices, num_classes=self.codebook_size)  # [B, L, K]
        used_mask = one_hot.any(dim=1)
        unique_counts_tensor = used_mask.sum(dim=1).float()  # [B]
        avg_unique = unique_counts_tensor.mean()

        self.last_avg_unique_codes = avg_unique
        return avg_unique

    def orthogonality_loss(self) -> Tensor:
        if self.codebook_size <= 1:
            return self.codebooks.new_zeros(())
        normalized_codebooks = F.normalize(self.codebooks, dim=1, p=2, eps=self.eps)
        gram = normalized_codebooks @ normalized_codebooks.t()
        identity = torch.eye(self.codebook_size, device=gram.device, dtype=gram.dtype)
        off_diag = gram - identity
        loss = (off_diag.pow(2).sum() - torch.diagonal(off_diag).pow(2).sum()) / (
            self.codebook_size * (self.codebook_size - 1)
        )
        return loss.clamp_min(0.0)

    def forward(self, nodes: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        nodes_proj = self.in_norm(nodes)
        with torch.no_grad():
            self.nodes_norm = torch.norm(nodes_proj, p=2, dim=-1).mean()
        nodes_for_init = F.normalize(nodes_proj, dim=-1, eps=self.eps) if self.use_cosine_sim else nodes_proj
        self._maybe_data_dependent_init(nodes_for_init)
        distances, nodes_used, code_used = self._compute_distance(
            nodes_proj, self.codebooks, return_normed=True
        )
        # temperature scheduling only when enabled
        base_temperature = torch.tensor(
            self.temperature_start, device=self.codebooks.device, dtype=self.codebooks.dtype
        )
        temperature = self._get_temperature() if self.use_temperature_schedule else base_temperature
        if self.use_soft_assignment:
            # soft assignment over codebooks
            logits = (-distances) / temperature
            soft_weights = F.softmax(logits, dim=-1)
            soft_weights = soft_weights.clamp(min=self.eps, max=1.0)
            quantized_soft = torch.matmul(soft_weights.view(-1, self.codebook_size), code_used).view(
                *distances.shape[:-1], self.code_dim
            )
        else:
            quantized_soft = None

        min_indices = torch.argmin(distances, dim=-1)
        hard_quantized = F.embedding(min_indices, code_used)
        self._update_slot_redundancy_metrics(min_indices)

        self._compute_avg_unique_codes(min_indices)

        codebook_loss = F.mse_loss(hard_quantized, nodes_used.detach())
        commitment_loss = F.mse_loss(hard_quantized.detach(), nodes_used)
        self.last_commitment_loss = commitment_loss.detach()
        quantized_st = nodes_used + (hard_quantized - nodes_used).detach()
        if self.use_soft_assignment and quantized_soft is not None:
            # forward uses soft assignment, backward follows straight-through hard path
            quantized = quantized_st + (quantized_soft - quantized_st).detach()
        else:
            quantized = quantized_st
        if self.orthogonal_loss_weight > 0:
            orth_loss = self.orthogonality_loss()
            self.last_orthogonal_loss = orth_loss.detach()
            vq_loss = codebook_loss + self.beta * commitment_loss + self.orthogonal_loss_weight * orth_loss
        else:
            vq_loss = codebook_loss + self.beta * commitment_loss
            self.last_orthogonal_loss = None

        perplexity = self._compute_batch_perplexity(min_indices)

        entropy_temp = temperature if (self.use_temperature_schedule or self.use_soft_assignment) else None
        entropy_loss = self.entropy_loss(-distances, temperature=entropy_temp)
        self.node_count.index_add_(
            0, min_indices.reshape(-1), torch.ones_like(min_indices.reshape(-1), dtype=self.node_count.dtype)
        )
        self.compute_inter_code_stats()

        quantized_out = quantized

        return (quantized_out, perplexity, min_indices, entropy_loss, vq_loss)

    @torch.no_grad()
    def _get_temperature(self) -> Tensor:
        """
        Exponentially decayed temperature: start -> end over decay steps.
        """
        if self.temperature_decay_steps <= 0:
            current = self.temperature_end
        else:
            step = min(int(self.temperature_step.item()), self.temperature_decay_steps)
            ratio = step / float(self.temperature_decay_steps)
            decay = (self.temperature_end / self.temperature_start) ** ratio
            current = self.temperature_start * decay
            current = max(self.temperature_end, current)
        temp_tensor = torch.tensor(current, device=self.codebooks.device, dtype=self.codebooks.dtype)
        if self.training and self.use_temperature_schedule:
            self.temperature_step.add_(1)
        self.last_temperature = temp_tensor
        return temp_tensor

    @torch.no_grad()
    def compute_inter_code_stats(self) -> tuple[Tensor, Tensor]:
        if self.codebook_size <= 1:
            zero = torch.tensor(0.0, device=self.codebooks.device)
            return zero, zero
        if self.use_cosine_sim:
            code_normed = F.normalize(self.codebooks, dim=-1, eps=self.eps)
            sim = code_normed @ code_normed.t()
            eye = torch.eye(self.codebook_size, device=sim.device, dtype=torch.bool)
            sim_no_diag = sim.masked_fill(eye, -1.1)  # remove self
            max_sim = sim_no_diag.max()
            angles = torch.arccos(torch.clamp(sim_no_diag, -1.0 + 1e-6, 1.0 - 1e-6))
            min_angle = torch.arccos(torch.clamp(max_sim, -1.0 + 1e-6, 1.0 - 1e-6))
            avg_angle = angles.mean()
            self.last_min_inter_code_dist = min_angle
            self.last_avg_inter_code_dist = avg_angle
            return min_angle, avg_angle
        else:
            dists = torch.cdist(self.codebooks, self.codebooks)
            # mask self-distance
            masked = dists.where(
                dists > 1e-6, torch.tensor(float("inf"), device=dists.device, dtype=dists.dtype)
            )
            min_dist = masked.min()
            # exclude diagonal for mean
            off_diag = dists[~torch.eye(self.codebook_size, dtype=torch.bool, device=dists.device)]
            avg_dist = off_diag.mean()
            self.last_min_inter_code_dist = min_dist
            self.last_avg_inter_code_dist = avg_dist
            return min_dist, avg_dist

    @torch.no_grad()
    def replace_unused_codebooks(self) -> tuple[int, Tensor]:
        if not bool(self.initialized.item()):
            return 0, torch.empty(0, device=self.codebooks.device, dtype=torch.long)

        # std = self.codebooks.data.std()
        # eps_noise = 0.01 * torch.clamp(std, min=1e-5)
        eps_noise = 1e-10

        ddp_initialized = torch.distributed.is_available() and torch.distributed.is_initialized()
        if ddp_initialized:
            global_node_count = self.node_count.clone()
            torch.distributed.all_reduce(global_node_count, op=torch.distributed.ReduceOp.SUM)
            rank = torch.distributed.get_rank()
        else:
            global_node_count = self.node_count
            rank = 0

        total_count = global_node_count.sum()
        denom = total_count.clamp(min=1).float()

        usage_ratio = global_node_count.float() / denom
        unused_mask = usage_ratio < self.discarding_threshold
        used_mask = ~unused_mask

        unused_indices = torch.where(unused_mask)[0]
        used_indices = torch.where(used_mask)[0]

        if ddp_initialized:
            num_replaced_tensor = torch.zeros(1, device=self.codebooks.device, dtype=torch.long)

        num_unused = int(unused_indices.numel())
        num_replaced = 0

        replaced_indices_out = torch.full(
            (self.codebook_size,), -1, device=self.codebooks.device, dtype=torch.long
        )

        if rank == 0:
            if num_unused == 0:
                print("=" * 50, "\n")
                print("No unused codebooks to replace. global_node_count: ", global_node_count)
            else:
                print("=" * 50, "\n")
                print("global_node_count", global_node_count)

                max_per_step = self.max_code_replaced_per_step
                if max_per_step is not None and int(max_per_step) > 0:
                    max_per_step = int(max_per_step)
                    num_replaced = min(num_unused, max_per_step)
                else:
                    num_replaced = num_unused

                if num_replaced == 0:
                    print(
                        f"Found {num_unused} unused codebooks, "
                        f"but max_codebooks_replaced_per_step == 0, skip replacement."
                    )
                elif used_indices.numel() == 0:
                    self.codebooks.data += eps_noise * torch.randn_like(self.codebooks.data)
                    print("All codebooks are unused, adding noise to reactivate")
                else:
                    used_counts = global_node_count[used_indices]
                    used_counts_sum = used_counts.sum().float()
                    if used_counts_sum > 0:
                        probs = used_counts.float() / used_counts_sum
                    else:
                        probs = torch.ones_like(used_counts, dtype=torch.float) / used_indices.numel()

                    unused_counts = global_node_count[unused_indices]
                    _, sort_idx = torch.sort(unused_counts, descending=False)
                    unused_indices = unused_indices[sort_idx[:num_replaced]]

                    sampled_indices = torch.multinomial(probs, num_replaced, replacement=True)
                    sampled_used_indices = used_indices[sampled_indices]
                    replacements = self.codebooks.data[sampled_used_indices]
                    noise = eps_noise * torch.randn_like(replacements)
                    self.codebooks.data[unused_indices] = replacements + noise
                    print("=" * 50)
                    print(
                        "Replaced {} unused codebooks (found {}, cap: {}) using importance sampling based on node_count".format(
                            num_replaced, num_unused, max_per_step if max_per_step is not None else -1
                        )
                    )
                    replaced_indices_out[:num_replaced] = unused_indices

            if ddp_initialized:
                num_replaced_tensor.fill_(num_replaced)

        if ddp_initialized:
            torch.distributed.broadcast(self.codebooks.data, src=0)
            torch.distributed.barrier()
            torch.distributed.broadcast(num_replaced_tensor, src=0)
            torch.distributed.broadcast(replaced_indices_out, src=0)
            return int(num_replaced_tensor.item()), replaced_indices_out[replaced_indices_out >= 0]
        else:
            return num_replaced, replaced_indices_out[replaced_indices_out >= 0]

    def reset_node_count(self) -> None:
        # print("Resetting node count")
        # print("=" * 50)
        self.node_count.zero_()

    @torch.no_grad()
    def inference(
        self,
        nodes: Tensor,
        user_specific: int | list[int] | None = None,
        return_distance: bool = False,
        return_logits: bool = False,
        return_probs: bool = False,
        temperature: float | None = None,
    ) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None, Tensor | None]:
        batch_size = nodes.shape[0]
        nodes_proj = self.in_norm(nodes)

        distances: Tensor | None = None
        code_used: Tensor | None = None

        if user_specific is not None:
            if isinstance(user_specific, list):
                base_indices = torch.tensor(user_specific, device=nodes.device, dtype=torch.long)
                if base_indices.dim() != 1:
                    raise ValueError("user_specific as list must be a 1D list of indices.")
                min_indices = base_indices.unsqueeze(0).expand(batch_size, -1)
            else:
                min_indices = torch.full((batch_size,), user_specific, device=nodes.device, dtype=torch.long)
            code_used = (
                F.normalize(self.codebooks, dim=-1, eps=self.eps) if self.use_cosine_sim else self.codebooks
            )
        else:
            distances, nodes_used, code_used = self._compute_distance(
                nodes_proj, self.codebooks, return_normed=True
            )
            min_indices = torch.argmin(distances, dim=-1)

        if (return_distance or return_logits or return_probs) and distances is None:
            distances, _, code_used = self._compute_distance(nodes_proj, self.codebooks, return_normed=True)

        quantized = F.embedding(min_indices, code_used)
        quantized_out = quantized

        logits_out: Tensor | None = None
        probs_out: Tensor | None = None

        if distances is not None and (return_distance or return_logits or return_probs):
            temp = float(temperature) if temperature is not None else 1.0
            logits = -distances / temp
            logits_out = logits if return_logits else None
            if return_probs:
                probs_out = F.softmax(logits, dim=-1)

        if return_distance or return_logits or return_probs:
            return (
                quantized_out.reshape(batch_size, -1, self.code_dim),
                min_indices.view(batch_size, -1),
                distances,
                logits_out,
                probs_out,
            )

        return (
            quantized_out.reshape(batch_size, -1, self.code_dim),
            min_indices.view(batch_size, -1),
        )

    def codebook_reinit(self) -> None:
        if isinstance(self.codebooks, nn.Parameter):
            nn.init.uniform_(self.codebooks.data, -1 / self.codebook_size, 1 / self.codebook_size)
        self.reset_node_count()

    def get_codebooks(self) -> Tensor:
        return self.codebooks

    def get_codebook_size(self) -> int:
        return self.codebook_size


class EMAVQ(VQ):
    def __init__(
        self,
        ema_decay: float = 0.99,
        ema_eps: float = 1e-5,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.ema_decay = float(ema_decay)
        self.ema_eps = float(ema_eps)
        self.register_buffer("ema_cluster_size", torch.zeros(self.codebook_size))
        self.register_buffer("ema_embed_sum", torch.zeros(self.codebook_size, self.code_dim))
        self.register_buffer("ema_initialized", torch.tensor(False, dtype=torch.bool))
        # Codebook weights are maintained by EMA only
        self.codebooks.requires_grad_(False)

    @torch.no_grad()
    def _init_ema_from_codebooks(self) -> None:
        """Seed EMA statistics from current codebooks."""
        codebooks_data = self.codebooks.data
        if self.use_cosine_sim:
            codebooks_data = F.normalize(codebooks_data, dim=-1, eps=self.eps)
            self.codebooks.data.copy_(codebooks_data)
        self.ema_cluster_size.fill_(self.ema_eps)
        self.ema_embed_sum.copy_(codebooks_data)
        self.ema_initialized.fill_(True)

    @torch.no_grad()
    def _ema_update(self, min_indices: Tensor, nodes: Tensor) -> None:
        """EMA update of codebooks using current assignments."""
        if min_indices.numel() == 0:
            return
        flat_indices = min_indices.reshape(-1)
        flat_nodes = nodes.reshape(-1, self.code_dim)
        encodings = F.one_hot(flat_indices, num_classes=self.codebook_size).to(flat_nodes.dtype)
        # Local batch stats
        cluster_size = encodings.sum(dim=0)
        embed_sum = encodings.t() @ flat_nodes
        # All-reduce for DDP
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(cluster_size)
            dist.all_reduce(embed_sum)
        self.ema_cluster_size.mul_(self.ema_decay).add_(cluster_size * (1.0 - self.ema_decay))
        self.ema_embed_sum.mul_(self.ema_decay).add_(embed_sum * (1.0 - self.ema_decay))
        denom = self.ema_cluster_size + self.ema_eps * float(self.codebook_size)
        updated_codebook = self.ema_embed_sum / denom.unsqueeze(1)
        if self.use_cosine_sim:
            updated_codebook = F.normalize(updated_codebook, dim=-1, eps=self.eps)
        self.codebooks.data.copy_(updated_codebook)

    @torch.no_grad()
    def kmeans_init(
        self,
        nodes: Tensor,
        num_iters: int | None = None,
        max_samples: int | None = None,
    ) -> None:
        super().kmeans_init(nodes, num_iters=num_iters, max_samples=max_samples)
        self._init_ema_from_codebooks()

    @torch.no_grad()
    def codebook_reinit(self) -> None:
        super().codebook_reinit()
        self._init_ema_from_codebooks()

    @torch.no_grad()
    def replace_unused_codebooks(self) -> tuple[int, Tensor]:
        num_replaced, replaced_indices = super().replace_unused_codebooks()
        if num_replaced > 0 and replaced_indices.numel() > 0:
            codebooks_slice = self.codebooks.data[replaced_indices]
            if self.use_cosine_sim:
                codebooks_slice = F.normalize(codebooks_slice, dim=-1, eps=self.eps)
                self.codebooks.data[replaced_indices] = codebooks_slice
            self.ema_cluster_size[replaced_indices] = 1.0
            self.ema_embed_sum[replaced_indices] = codebooks_slice
        return num_replaced, replaced_indices

    def forward(self, nodes: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        # Data-dependent init (k-means) if needed
        nodes_proj = self.in_norm(nodes)
        with torch.no_grad():
            self.nodes_norm = torch.norm(nodes_proj, p=2, dim=-1).mean()
        nodes_for_init = F.normalize(nodes_proj, dim=-1, eps=self.eps) if self.use_cosine_sim else nodes_proj
        self._maybe_data_dependent_init(nodes_for_init)
        with torch.no_grad():
            if not bool(self.ema_initialized.item()):
                self._init_ema_from_codebooks()

        distances, nodes_used, code_used = self._compute_distance(
            nodes_proj, self.codebooks, return_normed=True
        )
        base_temperature = torch.tensor(
            self.temperature_start, device=self.codebooks.device, dtype=self.codebooks.dtype
        )
        temperature = self._get_temperature() if self.use_temperature_schedule else base_temperature
        if self.use_soft_assignment:
            logits = (-distances) / temperature
            soft_weights = F.softmax(logits, dim=-1)
            soft_weights = soft_weights.clamp(min=self.eps, max=1.0)
            quantized_soft = torch.matmul(soft_weights.view(-1, self.codebook_size), code_used).view(
                *distances.shape[:-1], self.code_dim
            )
        else:
            quantized_soft = None

        min_indices = torch.argmin(distances, dim=-1)
        hard_quantized = F.embedding(min_indices, code_used)

        # Metrics
        self._update_slot_redundancy_metrics(min_indices)
        self._compute_avg_unique_codes(min_indices)

        # EMA update
        self._ema_update(min_indices, nodes_used)

        # Losses
        commitment_loss = F.mse_loss(hard_quantized.detach(), nodes_used)
        self.last_commitment_loss = commitment_loss.detach()
        quantized_st = nodes_used + (hard_quantized - nodes_used).detach()
        if self.use_soft_assignment and quantized_soft is not None:
            quantized = quantized_st + (quantized_soft - quantized_st).detach()
        else:
            quantized = quantized_st
        vq_loss = self.beta * commitment_loss

        perplexity = self._compute_batch_perplexity(min_indices)
        entropy_temp = temperature if (self.use_temperature_schedule or self.use_soft_assignment) else None
        entropy_loss = self.entropy_loss(-distances, temperature=entropy_temp)
        self.node_count.index_add_(
            0, min_indices.reshape(-1), torch.ones_like(min_indices.reshape(-1), dtype=self.node_count.dtype)
        )
        self.compute_inter_code_stats()

        quantized_out = quantized

        return (
            quantized_out,
            perplexity,
            min_indices,
            entropy_loss,
            vq_loss,
        )


class NSVQ(VQ):
    def __init__(self, use_diveq: bool = True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_diveq = use_diveq

    def NSVQ_core(self, nodes: Tensor, hard_quantized: Tensor) -> Tensor:
        random_vector = torch.randn_like(nodes)
        norm_quantization_residual = torch.linalg.norm(nodes - hard_quantized, dim=-1, keepdim=True)
        norm_random_vector = torch.linalg.norm(random_vector, dim=-1, keepdim=True)
        vq_error = (norm_quantization_residual / (norm_random_vector + self.eps)) * random_vector
        return nodes + vq_error

    def DiVeQ_core(self, nodes: Tensor, hard_quantized: Tensor, noise_variance=1e-3):
        error_dir = hard_quantized - nodes
        error_dir_norm = error_dir.norm(dim=-1, keepdim=True)

        noised_dir = error_dir + torch.sqrt(
            torch.tensor(noise_variance, device=error_dir.device)
        ) * torch.randn_like(error_dir)
        unit_noised_dir = F.normalize(noised_dir, dim=-1, p=2)

        return nodes + error_dir_norm * unit_noised_dir.detach()

    def forward(self, nodes: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        nodes_proj = self.in_norm(nodes)
        nodes_for_init = F.normalize(nodes_proj, dim=-1, eps=self.eps) if self.use_cosine_sim else nodes_proj
        if self.training:
            self._maybe_data_dependent_init(nodes_for_init)
        distances, nodes_used, code_used = self._compute_distance(
            nodes_proj, self.codebooks, return_normed=True
        )
        with torch.no_grad():
            self.nodes_norm = torch.norm(nodes_used, p=2, dim=-1).mean()
        min_indices = torch.argmin(distances, dim=-1)
        hard_quantized = F.embedding(min_indices, code_used)
        self.last_commitment_loss = F.mse_loss(hard_quantized, nodes_used).detach()

        self._update_slot_redundancy_metrics(min_indices)

        self._compute_avg_unique_codes(min_indices)

        if self.use_diveq:
            quantized = self.DiVeQ_core(nodes_used, hard_quantized)
        else:
            quantized = self.NSVQ_core(nodes_used, hard_quantized)

        perplexity = self._compute_batch_perplexity(min_indices)
        self.node_count.index_add_(
            0, min_indices.reshape(-1), torch.ones_like(min_indices.reshape(-1), dtype=self.node_count.dtype)
        )
        entropy_loss = self.entropy_loss(-distances)
        if self.orthogonal_loss_weight > 0:
            orth_loss = self.orthogonality_loss()
            vq_loss = self.orthogonal_loss_weight * orth_loss
        else:
            vq_loss = torch.zeros_like(entropy_loss)

        quantized_out = quantized

        return (quantized_out, perplexity, min_indices, entropy_loss, vq_loss)
