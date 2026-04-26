"""
SWAG: Stochastic Weight Averaging-Gaussian

Implementation of SWAG for Bayesian uncertainty estimation in neural networks.
Based on: "A Simple Baseline for Bayesian Uncertainty Estimation in Deep Learning"
(Maddox et al., NeurIPS 2019)

Mathematical formulation:
- Posterior approximation: p(w|D) ≈ N(w_SWA, Σ_SWAG)
- Mean: w_SWA = (1/n) Σ w_i
- Covariance: Σ_SWAG = (1/2)(diag(σ²) + (1/(K-1))DD^T)
  where D = [w_1 - w_SWA, ..., w_K - w_SWA] (deviation matrix)

Supports two variants:
1. Low-rank: Full covariance with rank-limited deviation matrix (default)
2. Diagonal-only: Only diagonal covariance (faster, less memory)
"""

import torch
import torch.nn as nn
from collections import OrderedDict
import copy


class SWAG(nn.Module):
    """
    SWAG: Stochastic Weight Averaging-Gaussian

    Collects statistics of model weights during training and enables
    sampling from the approximate posterior for Bayesian inference.
    """

    def __init__(self, base_model, no_cov_mat=False, max_num_models=20, var_clamp=1e-30):
        """
        Args:
            base_model: PyTorch model to wrap
            no_cov_mat: If True, use diagonal-only SWAG (no low-rank component)
            max_num_models: Maximum number of weight snapshots to store for covariance
            var_clamp: Minimum variance value to prevent numerical issues
        """
        super(SWAG, self).__init__()

        self.base_model = base_model
        self.no_cov_mat = no_cov_mat
        self.max_num_models = max_num_models
        self.var_clamp = var_clamp

        # Statistics
        self.n_models = 0  # Number of models collected

        # Register buffers for mean and variance
        self.register_buffer('mean', None)
        self.register_buffer('sq_mean', None)  # E[w²] for computing variance

        # Deviation matrix for low-rank covariance (only if not no_cov_mat)
        if not self.no_cov_mat:
            self.register_buffer('cov_mat_sqrt', None)  # Stores deviations [D_1, D_2, ..., D_K]

        # Initialize buffers
        self._init_buffers()

    def _init_buffers(self):
        """Initialize mean, sq_mean, and deviation matrix buffers."""
        # Flatten all model parameters into a single vector
        params = []
        for param in self.base_model.parameters():
            params.append(param.data.view(-1))

        flat_params = torch.cat(params)
        num_params = flat_params.numel()

        # Initialize mean and sq_mean
        self.mean = torch.zeros(num_params, dtype=flat_params.dtype, device=flat_params.device)
        self.sq_mean = torch.zeros(num_params, dtype=flat_params.dtype, device=flat_params.device)

        # Initialize deviation matrix (low-rank component)
        if not self.no_cov_mat:
            self.cov_mat_sqrt = torch.zeros(
                (0, num_params),
                dtype=flat_params.dtype,
                device=flat_params.device
            )

    def _flatten_params(self, model=None):
        """Flatten model parameters into a single vector."""
        if model is None:
            model = self.base_model

        params = []
        for param in model.parameters():
            params.append(param.data.view(-1))
        return torch.cat(params)

    def _unflatten_params(self, flat_params, model=None):
        """Unflatten parameter vector back into model parameters."""
        if model is None:
            model = self.base_model

        offset = 0
        for param in model.parameters():
            numel = param.numel()
            param.data.copy_(flat_params[offset:offset + numel].view_as(param))
            offset += numel

    def collect_model(self, model=None):
        """
        Collect current model weights and update SWAG statistics.

        Updates:
        - Running mean: w_SWA = (n*w_SWA + w) / (n+1)
        - Running second moment: E[w²]
        - Deviation matrix: D = [w_1 - w_SWA, ..., w_K - w_SWA]

        Args:
            model: Model to collect (if None, uses base_model)
        """
        if model is None:
            model = self.base_model

        # Get current model parameters
        w = self._flatten_params(model)

        # Update running mean and second moment
        # mean_new = (n * mean_old + w) / (n + 1)
        self.mean.mul_(self.n_models).add_(w).div_(self.n_models + 1)
        self.sq_mean.mul_(self.n_models).add_(w ** 2).div_(self.n_models + 1)

        # Update deviation matrix (low-rank component)
        if not self.no_cov_mat:
            dev = (w - self.mean).unsqueeze(0)  # [1, num_params]

            # Append new deviation
            self.cov_mat_sqrt = torch.cat([self.cov_mat_sqrt, dev], dim=0)

            # Keep only last max_num_models deviations
            if self.cov_mat_sqrt.size(0) > self.max_num_models:
                self.cov_mat_sqrt = self.cov_mat_sqrt[-self.max_num_models:]

        self.n_models += 1

    def sample(self, scale=1.0, cov=True, seed=None):
        """
        Sample weights from SWAG posterior and set them in base_model.

        Sampling: w ~ N(w_SWA, scale² * Σ_SWAG)
        where Σ_SWAG = (1/2)(diag(σ²) + (1/(K-1))DD^T)

        Args:
            scale: Scaling factor for covariance (default: 1.0)
            cov: If True, include covariance term; if False, only use mean
            seed: Random seed for reproducibility

        Returns:
            Sampled weight vector
        """
        if seed is not None:
            torch.manual_seed(seed)

        if not cov or self.n_models == 0:
            # No sampling, just use mean
            self._unflatten_params(self.mean)
            return self.mean.clone()

        # Compute diagonal variance: σ² = E[w²] - E[w]²
        var = torch.clamp(self.sq_mean - self.mean ** 2, self.var_clamp)

        # Sample from diagonal component: z1 ~ N(0, diag(σ²))
        z1 = torch.randn_like(var)
        sample = self.mean + scale * torch.sqrt(var) * z1 * 0.5**0.5  # Scale by 1/√2

        # Add low-rank component if available
        if not self.no_cov_mat and self.cov_mat_sqrt.size(0) > 0:
            # Sample from low-rank component: z2 ~ N(0, (1/(K-1))DD^T)
            K = self.cov_mat_sqrt.size(0)
            z2 = torch.randn(K, device=self.cov_mat_sqrt.device, dtype=self.cov_mat_sqrt.dtype)

            # Compute (1/√(K-1)) * D^T * z2
            cov_sample = (self.cov_mat_sqrt.t() @ z2) / ((K - 1) ** 0.5)
            sample += scale * cov_sample * 0.5**0.5  # Scale by 1/√2

        # Set sampled weights in model
        self._unflatten_params(sample)

        return sample

    def set_swa(self):
        """Set model weights to SWA mean (no sampling)."""
        self._unflatten_params(self.mean)

    def get_space(self):
        """Get number of parameters being tracked."""
        return self.mean.numel()

    def get_variance(self):
        """
        Compute diagonal variance: σ² = E[w²] - E[w]²

        Returns:
            Variance vector
        """
        return torch.clamp(self.sq_mean - self.mean ** 2, self.var_clamp)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        """
        Save SWAG state including statistics.

        Returns:
            Dictionary with SWAG statistics and base model state
        """
        state = super(SWAG, self).state_dict(destination, prefix, keep_vars)

        # Add SWAG-specific state (wrap in tensor for Detectron2 compatibility)
        state[prefix + 'n_models'] = torch.tensor(self.n_models, dtype=torch.int64)

        return state

    def load_state_dict(self, state_dict, strict=True):
        """
        Load SWAG state including statistics.

        Args:
            state_dict: Dictionary with SWAG statistics
            strict: Whether to strictly enforce key matching
        """
        # Extract n_models if present (unwrap from tensor)
        if 'n_models' in state_dict:
            n_models_value = state_dict.pop('n_models')
            # Handle both tensor and int (for backward compatibility)
            if isinstance(n_models_value, torch.Tensor):
                self.n_models = int(n_models_value.item())
            else:
                self.n_models = int(n_models_value)

        # Load buffers and base model
        super(SWAG, self).load_state_dict(state_dict, strict=strict)

    def forward(self, *args, **kwargs):
        """Forward pass through base model."""
        return self.base_model(*args, **kwargs)

    def __repr__(self):
        return f"SWAG(n_models={self.n_models}, no_cov_mat={self.no_cov_mat}, max_num_models={self.max_num_models})"


def flatten_model_params(model):
    """
    Utility function to flatten all model parameters into a single vector.

    Args:
        model: PyTorch model

    Returns:
        Flattened parameter tensor
    """
    params = []
    for param in model.parameters():
        params.append(param.data.view(-1))
    return torch.cat(params)


def unflatten_model_params(flat_params, model):
    """
    Utility function to unflatten parameter vector back into model.

    Args:
        flat_params: Flattened parameter tensor
        model: PyTorch model to update
    """
    offset = 0
    for param in model.parameters():
        numel = param.numel()
        param.data.copy_(flat_params[offset:offset + numel].view_as(param))
        offset += numel
