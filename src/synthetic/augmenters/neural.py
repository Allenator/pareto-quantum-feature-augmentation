"""Neural network feature augmenters (torch)."""

import time

import numpy as np

from src.synthetic.augmenters.base import AugmenterResult, _make_result

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

_DEVICE = torch.device("cpu") if TORCH_AVAILABLE else None


class MLPAugmenter:
    """Trainable MLP feature extractor.

    Trains end-to-end with a linear head (MSE loss), then extracts
    the feature layer activations for downstream regression.
    Uses mini-batch SGD, cosine LR schedule, and early stopping.
    """

    def __init__(self, n_out: int = 10, hidden_dims: tuple[int, ...] = (32, 16),
                 n_epochs: int = 1000, lr: float = 1e-3, weight_decay: float = 1e-4,
                 batch_size: int = 256, patience: int = 50, seed: int = 42):
        if not TORCH_AVAILABLE:
            raise ImportError("torch is required for MLPAugmenter: uv add torch")
        self.n_out = n_out
        self.hidden_dims = hidden_dims
        self.n_epochs = n_epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.patience = patience
        self._seed = seed
        self.name = f"mlp_{n_out}"
        self._trained = False

    def _build_model(self, n_features: int):
        torch.manual_seed(self._seed)
        layers = []
        in_dim = n_features
        for h in self.hidden_dims:
            layers.extend([nn.Linear(in_dim, h), nn.ReLU()])
            in_dim = h
        layers.append(nn.Linear(in_dim, self.n_out))
        self._feature_extractor = nn.Sequential(*layers)
        self._head = nn.Linear(self.n_out, 1)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        self._build_model(X_train.shape[1])

        # Split off 10% for early stopping
        n = len(X_train)
        rng = np.random.default_rng(self._seed)
        perm = rng.permutation(n)
        n_val = max(1, n // 10)
        val_idx, train_idx = perm[:n_val], perm[n_val:]

        X_t = torch.tensor(X_train[train_idx], dtype=torch.float32)
        y_t = torch.tensor(y_train[train_idx], dtype=torch.float32).unsqueeze(1)
        X_v = torch.tensor(X_train[val_idx], dtype=torch.float32)
        y_v = torch.tensor(y_train[val_idx], dtype=torch.float32).unsqueeze(1)

        loader = DataLoader(TensorDataset(X_t, y_t), batch_size=self.batch_size, shuffle=True)
        params = list(self._feature_extractor.parameters()) + list(self._head.parameters())
        optimizer = torch.optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.n_epochs)
        loss_fn = nn.MSELoss()

        best_val_loss = float("inf")
        best_state = None
        wait = 0

        for epoch in range(self.n_epochs):
            self._feature_extractor.train()
            self._head.train()
            for X_b, y_b in loader:
                pred = self._head(self._feature_extractor(X_b))
                loss = loss_fn(pred, y_b)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()

            # Validation
            self._feature_extractor.eval()
            self._head.eval()
            with torch.no_grad():
                val_loss = loss_fn(self._head(self._feature_extractor(X_v)), y_v).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {
                    "fe": {k: v.clone() for k, v in self._feature_extractor.state_dict().items()},
                    "head": {k: v.clone() for k, v in self._head.state_dict().items()},
                }
                wait = 0
            else:
                wait += 1

            if (epoch + 1) % 100 == 0:
                print(f"  MLP epoch {epoch + 1}/{self.n_epochs}, val_loss: {val_loss:.4f}, best: {best_val_loss:.4f}")

            if wait >= self.patience:
                print(f"  MLP early stop at epoch {epoch + 1}, best val_loss: {best_val_loss:.4f}")
                break

        if best_state:
            self._feature_extractor.load_state_dict(best_state["fe"])
            self._head.load_state_dict(best_state["head"])
        self._trained = True

    def transform(self, X: np.ndarray) -> AugmenterResult:
        if not self._trained:
            raise RuntimeError("MLPAugmenter.fit() must be called before transform()")
        t0 = time.perf_counter()
        X_t = torch.tensor(X, dtype=torch.float32)
        self._feature_extractor.eval()
        with torch.no_grad():
            X_new = self._feature_extractor(X_t).numpy()
        elapsed = time.perf_counter() - t0
        n_train = sum(p.numel() for p in self._feature_extractor.parameters())
        n_train += sum(p.numel() for p in self._head.parameters())
        return _make_result(X, X_new, self.name, elapsed, n_trainable_params=n_train)


class AutoencoderAugmenter:
    """Autoencoder bottleneck feature extractor.

    Hybrid loss: reconstruction + supervised probe on bottleneck.
    """

    def __init__(self, bottleneck_dim: int = 4, hidden_dim: int = 8,
                 n_epochs: int = 1000, lr: float = 1e-3, weight_decay: float = 1e-4,
                 batch_size: int = 256, patience: int = 50,
                 supervised_weight: float = 0.5, seed: int = 42):
        if not TORCH_AVAILABLE:
            raise ImportError("torch is required for AutoencoderAugmenter: uv add torch")
        self.bottleneck_dim = bottleneck_dim
        self.hidden_dim = hidden_dim
        self.n_epochs = n_epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.patience = patience
        self.supervised_weight = supervised_weight
        self._seed = seed
        self.name = f"autoencoder_{bottleneck_dim}"
        self._trained = False

    def _build_model(self, n_features: int):
        torch.manual_seed(self._seed)
        self._encoder = nn.Sequential(
            nn.Linear(n_features, self.hidden_dim), nn.ReLU(),
            nn.Linear(self.hidden_dim, self.bottleneck_dim),
        )
        self._decoder = nn.Sequential(
            nn.Linear(self.bottleneck_dim, self.hidden_dim), nn.ReLU(),
            nn.Linear(self.hidden_dim, n_features),
        )
        self._probe = nn.Linear(self.bottleneck_dim, 1)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        self._build_model(X_train.shape[1])

        n = len(X_train)
        rng = np.random.default_rng(self._seed)
        perm = rng.permutation(n)
        n_val = max(1, n // 10)
        val_idx, train_idx = perm[:n_val], perm[n_val:]

        X_t = torch.tensor(X_train[train_idx], dtype=torch.float32)
        y_t = torch.tensor(y_train[train_idx], dtype=torch.float32).unsqueeze(1)
        X_v = torch.tensor(X_train[val_idx], dtype=torch.float32)
        y_v = torch.tensor(y_train[val_idx], dtype=torch.float32).unsqueeze(1)

        loader = DataLoader(TensorDataset(X_t, y_t), batch_size=self.batch_size, shuffle=True)
        params = (list(self._encoder.parameters()) + list(self._decoder.parameters())
                  + list(self._probe.parameters()))
        optimizer = torch.optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.n_epochs)
        mse = nn.MSELoss()
        sw = self.supervised_weight

        best_val_loss = float("inf")
        best_state = None
        wait = 0

        for epoch in range(self.n_epochs):
            self._encoder.train()
            self._decoder.train()
            self._probe.train()
            for X_b, y_b in loader:
                encoded = self._encoder(X_b)
                recon_loss = mse(self._decoder(encoded), X_b)
                sup_loss = mse(self._probe(encoded), y_b)
                loss = (1 - sw) * recon_loss + sw * sup_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()

            self._encoder.eval()
            self._decoder.eval()
            self._probe.eval()
            with torch.no_grad():
                enc_v = self._encoder(X_v)
                val_loss = ((1 - sw) * mse(self._decoder(enc_v), X_v)
                            + sw * mse(self._probe(enc_v), y_v)).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in self._encoder.state_dict().items()}
                wait = 0
            else:
                wait += 1

            if (epoch + 1) % 100 == 0:
                print(f"  AE epoch {epoch + 1}/{self.n_epochs}, val_loss: {val_loss:.4f}, best: {best_val_loss:.4f}")

            if wait >= self.patience:
                print(f"  AE early stop at epoch {epoch + 1}, best val_loss: {best_val_loss:.4f}")
                break

        if best_state:
            self._encoder.load_state_dict(best_state)
        self._trained = True

    def transform(self, X: np.ndarray) -> AugmenterResult:
        if not self._trained:
            raise RuntimeError("AutoencoderAugmenter.fit() must be called before transform()")
        t0 = time.perf_counter()
        X_t = torch.tensor(X, dtype=torch.float32)
        self._encoder.eval()
        with torch.no_grad():
            X_new = self._encoder(X_t).numpy()
        elapsed = time.perf_counter() - t0
        n_train = sum(p.numel() for p in self._encoder.parameters())
        n_train += sum(p.numel() for p in self._decoder.parameters())
        n_train += sum(p.numel() for p in self._probe.parameters())
        return _make_result(X, X_new, self.name, elapsed, n_trainable_params=n_train)


class LearnedRFFAugmenter:
    """Random Fourier Features with learned frequencies.

    Feature map: phi(x) = sqrt(2/D) * cos(W @ x + b)
    W and b are optimized end-to-end with a linear head.
    """

    def __init__(self, n_components: int = 10, n_epochs: int = 1000,
                 lr: float = 1e-3, weight_decay: float = 1e-4,
                 batch_size: int = 256, patience: int = 50, seed: int = 42):
        if not TORCH_AVAILABLE:
            raise ImportError("torch is required for LearnedRFFAugmenter: uv add torch")
        self.n_components = n_components
        self.n_epochs = n_epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.patience = patience
        self._seed = seed
        self.name = f"learned_rff_{n_components}"
        self._trained = False

    def _build_model(self, n_features: int):
        torch.manual_seed(self._seed)
        self._W = nn.Parameter(torch.randn(self.n_components, n_features) * 0.5)
        self._b = nn.Parameter(torch.rand(self.n_components) * 2 * torch.pi)
        self._head = nn.Linear(self.n_components, 1)

    def _phi(self, X_t):
        proj = X_t @ self._W.T + self._b
        return torch.cos(proj) * (2.0 / self.n_components) ** 0.5

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        self._build_model(X_train.shape[1])

        n = len(X_train)
        rng = np.random.default_rng(self._seed)
        perm = rng.permutation(n)
        n_val = max(1, n // 10)
        val_idx, train_idx = perm[:n_val], perm[n_val:]

        X_t = torch.tensor(X_train[train_idx], dtype=torch.float32)
        y_t = torch.tensor(y_train[train_idx], dtype=torch.float32).unsqueeze(1)
        X_v = torch.tensor(X_train[val_idx], dtype=torch.float32)
        y_v = torch.tensor(y_train[val_idx], dtype=torch.float32).unsqueeze(1)

        loader = DataLoader(TensorDataset(X_t, y_t), batch_size=self.batch_size, shuffle=True)
        params = [self._W, self._b] + list(self._head.parameters())
        optimizer = torch.optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.n_epochs)
        loss_fn = nn.MSELoss()

        best_val_loss = float("inf")
        best_W = None
        best_b = None
        wait = 0

        for epoch in range(self.n_epochs):
            self._head.train()
            for X_b, y_b in loader:
                pred = self._head(self._phi(X_b))
                loss = loss_fn(pred, y_b)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()

            self._head.eval()
            with torch.no_grad():
                val_loss = loss_fn(self._head(self._phi(X_v)), y_v).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_W = self._W.data.clone()
                best_b = self._b.data.clone()
                wait = 0
            else:
                wait += 1

            if (epoch + 1) % 100 == 0:
                print(f"  Learned RFF epoch {epoch + 1}/{self.n_epochs}, val_loss: {val_loss:.4f}, best: {best_val_loss:.4f}")

            if wait >= self.patience:
                print(f"  Learned RFF early stop at epoch {epoch + 1}, best val_loss: {best_val_loss:.4f}")
                break

        if best_W is not None:
            self._W.data = best_W
            self._b.data = best_b
        self._trained = True

    def transform(self, X: np.ndarray) -> AugmenterResult:
        if not self._trained:
            raise RuntimeError("LearnedRFFAugmenter.fit() must be called before transform()")
        t0 = time.perf_counter()
        X_t = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            X_new = self._phi(X_t).numpy()
        elapsed = time.perf_counter() - t0
        n_train = self._W.numel() + self._b.numel() + sum(p.numel() for p in self._head.parameters())
        return _make_result(X, X_new, self.name, elapsed, n_trainable_params=n_train)
