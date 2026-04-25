# Байесовские подходы для оценки неопределенности в нейронных сетях

## 1. Stochastic Variational Inference (SVI) ⭐⭐⭐⭐⭐

**Математически самый строгий подход!**

### Теория

Вместо поиска точного posterior p(w|D), аппроксимируем его вариационным распределением q(w|θ):

```
ELBO = E_q(w)[log p(D|w)] - KL[q(w|θ) || p(w)]
```

Максимизируем ELBO относительно вариационных параметров θ.

### Математическая формулировка

Для весов w нейронной сети:
- Prior: p(w) = N(0, σ²_prior I)
- Variational posterior: q(w|θ) = N(μ, diag(σ²))
- θ = {μ, σ} - вариационные параметры

Reparametrization trick: w = μ + σ ⊙ ε, где ε ~ N(0, I)

### Преимущества
- Строгое байесовское обоснование через вариационный вывод
- Масштабируется на большие данные (mini-batch training)
- Можно использовать с pre-trained моделями
- Не требует полного переобучения (fine-tune только μ, σ)
- Uncertainty decomposition: aleatoric + epistemic

### Реализация для ODIN

```python
class BayesianConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        self.weight_mu = nn.Parameter(torch.randn(...))
        self.weight_rho = nn.Parameter(torch.randn(...))  # σ = log(1 + exp(ρ))
        
    def forward(self, x):
        weight_sigma = torch.log1p(torch.exp(self.weight_rho))
        weight = self.weight_mu + weight_sigma * torch.randn_like(self.weight_mu)
        return F.conv2d(x, weight, ...)
```

### Сложность
- Удваивает количество параметров (μ и σ для каждого веса)
- Требует несколько forward passes для оценки uncertainty
- KL divergence нужно добавить в loss

---

## 2. Laplace Approximation ⭐⭐⭐⭐

**Очень стабильный post-hoc метод!**

### Теория

Аппроксимируем posterior p(w|D) Гауссианом вокруг MAP estimate w*:

```
p(w|D) ≈ N(w*, H^(-1))
```

где H - Hessian матрица log p(w|D) в точке w*.

### Математическая формулировка

Taylor expansion второго порядка:
```
log p(w|D) ≈ log p(w*|D) - 1/2 (w - w*)^T H (w - w*)
```

Hessian: H = -∇²_w log p(w|D)|_{w=w*}

Для нейронных сетей: H ≈ ∇²_w L(w) + λI (Fisher Information Matrix)

### Преимущества
- Можно применить к уже обученной модели (post-hoc)
- Не требует переобучения
- Математически обоснован через Taylor expansion
- Эффективные методы для больших сетей (KFAC, diagonal approximation)

### Реализация для ODIN

```python
from laplace import Laplace

# После обучения ODIN
model = build_model(cfg)
model.load_state_dict(torch.load('checkpoint.pth'))

# Laplace approximation
la = Laplace(model, 'classification',
             subset_of_weights='last_layer',  # или 'all'
             hessian_structure='kron')  # или 'diag', 'full'

la.fit(train_loader)
la.optimize_prior_precision(method='marglik')

# Inference с uncertainty
pred_mean, pred_var = la(x, link_approx='probit')
```

### Варианты
- **Last-layer Laplace**: только последний слой (быстро, но менее точно)
- **Full Laplace**: все веса (точно, но дорого)
- **KFAC**: Kronecker-factored approximation (компромисс)

---

## 3. Normalizing Flows для Posterior ⭐⭐⭐⭐⭐

**Самый гибкий подход!**

### Теория

Моделируем сложный posterior через серию обратимых преобразований:

```
w = f_K ∘ f_{K-1} ∘ ... ∘ f_1(z), где z ~ N(0, I)
```

Change of variables formula:
```
log q(w) = log p(z) - Σ log|det J_f_k|
```

### Математическая формулировка

Для каждого flow f_k: R^d → R^d:
- Обратимость: f_k^(-1) существует
- Эффективное вычисление Jacobian: det J_f_k

Популярные flows:
- **RealNVP**: affine coupling layers
- **MAF** (Masked Autoregressive Flow): autoregressive transformations
- **Glow**: invertible 1x1 convolutions

### Преимущества
- Моделирует сложные multimodal posterior distributions
- Точнее, чем mean-field approximation (SVI)
- Можно комбинировать с pre-trained моделями
- Exact likelihood computation

### Реализация для ODIN

```python
import normflows as nf

# Base distribution
base = nf.distributions.DiagGaussian(param_dim)

# Flow layers
flows = []
for _ in range(num_flows):
    flows.append(nf.flows.MaskedAffineFlow(param_dim, hidden_dim))
    flows.append(nf.flows.ActNorm(param_dim))

# Normalizing flow model
flow_model = nf.NormalizingFlow(base, flows)

# Training
optimizer = torch.optim.Adam(flow_model.parameters(), lr=1e-3)
for epoch in range(num_epochs):
    z, log_q = flow_model.sample(batch_size)
    loss = -log_q.mean() + kl_divergence(...)
    loss.backward()
    optimizer.step()
```

### Сложность
- Требует обучения flow модели
- Больше параметров, чем SVI
- Медленнее inference, чем Laplace

---

## 4. SWAG (Stochastic Weight Averaging-Gaussian) ⭐⭐⭐⭐

**Очень эффективный и стабильный!**

### Теория

Аппроксимируем posterior через первые два момента траектории SGD:

```
p(w|D) ≈ N(w_SWA, Σ_SWA)
```

где:
- w_SWA = 1/T Σ w_t (среднее весов)
- Σ_SWA = 1/T Σ (w_t - w_SWA)(w_t - w_SWA)^T (ковариация)

### Математическая формулировка

Low-rank + diagonal approximation:
```
Σ_SWA = 1/2 (D + 1/(K-1) DD^T)
```

где D - diagonal matrix, D - matrix of deviations.

Sampling: w ~ N(w_SWA, Σ_SWA)

### Преимущества
- Требует только несколько эпох fine-tuning (20-30 epochs)
- Математически обоснован через SWA
- Лучше, чем MC Dropout (более калиброванная uncertainty)
- Не требует изменения архитектуры

### Реализация для ODIN

```python
from torchcontrib.optim import SWA

# После pre-training
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
swa_model = SWA(optimizer, swa_start=10, swa_freq=5, swa_lr=0.005)

# SWAG training
swag_model = SWAG(base_model, no_cov_mat=False, max_num_models=20)

for epoch in range(num_swag_epochs):
    train_epoch(model, optimizer)
    if epoch >= swa_start:
        swag_model.collect_model(model)

# Inference
swag_model.sample(scale=1.0)  # sample from posterior
predictions = []
for _ in range(num_samples):
    swag_model.sample()
    predictions.append(model(x))
```

### Варианты
- **SWAG-Diagonal**: только diagonal covariance (быстрее)
- **SWAG-Full**: full covariance (точнее, но дороже)
- **MultiSWAG**: multiple independent SWAG runs

---

## 5. Rank-1 Bayesian Neural Networks ⭐⭐⭐⭐

**Эффективная параметризация!**

### Теория

Low-rank approximation вариационных параметров:

```
q(w|θ) = N(μ, r r^T + diag(σ²))
```

где r - rank-1 vector, σ² - diagonal variance.

### Математическая формулировка

Вместо full covariance matrix (O(d²) параметров):
```
Σ = Σ_{i=1}^{R} r_i r_i^T + diag(σ²)
```

где R << d (обычно R = 1-5).

Reparametrization:
```
w = μ + r ⊙ ε_1 + σ ⊙ ε_2, где ε_1, ε_2 ~ N(0, I)
```

### Преимущества
- Меньше параметров, чем full Bayes by Backprop
- Более стабильное обучение (меньше variance в gradients)
- Лучше моделирует correlations, чем mean-field
- Масштабируется на большие сети

### Реализация для ODIN

```python
class Rank1BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features, rank=1):
        self.weight_mu = nn.Parameter(torch.randn(out_features, in_features))
        self.weight_r = nn.Parameter(torch.randn(rank, out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.randn(out_features, in_features))
        
    def forward(self, x):
        # Sample from rank-1 posterior
        eps_r = torch.randn(self.rank, 1, 1)
        eps_sigma = torch.randn_like(self.weight_sigma)
        
        weight = self.weight_mu
        weight += (self.weight_r * eps_r).sum(0)  # rank-1 component
        weight += self.weight_sigma * eps_sigma    # diagonal component
        
        return F.linear(x, weight)
```

### Сложность
- Больше параметров, чем diagonal SVI, но меньше, чем full
- Требует несколько forward passes для uncertainty
- Нужно подбирать rank R

---

## Сравнение подходов

| Подход | Точность | Скорость | Память | Post-hoc | Сложность реализации |
|--------|----------|----------|--------|----------|---------------------|
| SVI | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ❌ | Средняя |
| Laplace | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ✅ | Простая |
| Flows | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐ | ❌ | Сложная |
| SWAG | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ✅ | Простая |
| Rank-1 | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ❌ | Средняя |

## Рекомендации для ODIN

1. **Для быстрого прототипа**: Laplace (last-layer) или SWAG
2. **Для максимальной точности**: Normalizing Flows или Full SVI
3. **Для production**: SWAG или Rank-1 BNN (баланс точность/скорость)
4. **Для fine-tuning pre-trained**: Laplace или SWAG (не требуют переобучения)