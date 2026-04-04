"""Neural network feature augmenters (torch, MPS backend on M4 Max)."""

import time

import numpy as np

from src.synthetic.augmenters.base import AugmenterResult, _make_result

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def _get_device():
    """Get the best available torch device."""
    if not TORCH_AVAILABLE:
        raise ImportError("torch is required for neural augmenters: uv add torch")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class MLPAugmenter:
    """Trainable MLP feature extractor.

    Architecture: Input(n_features) -> hidden layers -> output(n_out).
    Trained end-to-end with a linear head to minimize MSE on targets.
    After training, extracts the penultimate layer activations as features.
    """

    def __init__(self, n_out: int = 10, hidden_dims: tuple[int, ...] = (32, 16),
                 n_epochs: int = 200, lr: float = 1e-3, seed: int = 42):
        if not TORCH_AVAILABLE:
            raise ImportError("torch is required for MLPAugmenter: uv add torch")
        self.n_out = n_out
        self.hidden_dims = hidden_dims
        self.n_epochs = n_epochs
        self.lr = lr
        self._seed = seed
        self.name = f"mlp_{n_out}"
        self._model = None
        self._device = _get_device()
        self._trained = False

    def _build_model(self, n_features: int):
        torch.manual_seed(self._seed)
        layers = []
        in_dim = n_features
        for h in self.hidden_dims:
            layers.extend([nn.Linear(in_dim, h), nn.ReLU()])
            in_dim = h
        # Feature extraction layer (no activation — these become our features)
        layers.append(nn.Linear(in_dim, self.n_out))
        self._feature_extractor = nn.Sequential(*layers).to(self._device)
        # Prediction head
        self._head = nn.Linear(self.n_out, 1).to(self._device)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        self._build_model(X_train.shape[1])

        X_t = torch.tensor(X_train, dtype=torch.float32, device=self._device)
        y_t = torch.tensor(y_train, dtype=torch.float32, device=self._device).unsqueeze(1)

        params = list(self._feature_extractor.parameters()) + list(self._head.parameters())
        optimizer = torch.optim.Adam(params, lr=self.lr)
        loss_fn = nn.MSELoss()

        for epoch in range(self.n_epochs):
            self._feature_extractor.train()
            self._head.train()
            features = self._feature_extractor(X_t)
            pred = self._head(features)
            loss = loss_fn(pred, y_t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 50 == 0:
                print(f"  MLP epoch {epoch + 1}/{self.n_epochs}, loss: {loss.item():.4f}")

        self._trained = True

    def transform(self, X: np.ndarray) -> AugmenterResult:
        if not self._trained:
            raise RuntimeError("MLPAugmenter.fit() must be called before transform()")
        t0 = time.perf_counter()
        X_t = torch.tensor(X, dtype=torch.float32, device=self._device)
        self._feature_extractor.eval()
        with torch.no_grad():
            X_new = self._feature_extractor(X_t).cpu().numpy()
        elapsed = time.perf_counter() - t0
        return _make_result(X, X_new, self.name, elapsed)


class AutoencoderAugmenter:
    """Autoencoder bottleneck feature extractor (unsupervised)."""

    def __init__(self, bottleneck_dim: int = 4, hidden_dim: int = 8,
                 n_epochs: int = 200, lr: float = 1e-3, seed: int = 42):
        if not TORCH_AVAILABLE:
            raise ImportError("torch is required for AutoencoderAugmenter: uv add torch")
        self.bottleneck_dim = bottleneck_dim
        self.hidden_dim = hidden_dim
        self.n_epochs = n_epochs
        self.lr = lr
        self._seed = seed
        self.name = f"autoencoder_{bottleneck_dim}"
        self._encoder = None
        self._decoder = None
        self._device = _get_device()
        self._trained = False

    def _build_model(self, n_features: int):
        torch.manual_seed(self._seed)
        self._encoder = nn.Sequential(
            nn.Linear(n_features, self.hidden_dim), nn.ReLU(),
            nn.Linear(self.hidden_dim, self.bottleneck_dim),
        ).to(self._device)
        self._decoder = nn.Sequential(
            nn.Linear(self.bottleneck_dim, self.hidden_dim), nn.ReLU(),
            nn.Linear(self.hidden_dim, n_features),
        ).to(self._device)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        self._build_model(X_train.shape[1])

        X_t = torch.tensor(X_train, dtype=torch.float32, device=self._device)
        params = list(self._encoder.parameters()) + list(self._decoder.parameters())
        optimizer = torch.optim.Adam(params, lr=self.lr)
        loss_fn = nn.MSELoss()

        for epoch in range(self.n_epochs):
            self._encoder.train()
            self._decoder.train()
            encoded = self._encoder(X_t)
            decoded = self._decoder(encoded)
            loss = loss_fn(decoded, X_t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 50 == 0:
                print(f"  AE epoch {epoch + 1}/{self.n_epochs}, loss: {loss.item():.4f}")

        self._trained = True

    def transform(self, X: np.ndarray) -> AugmenterResult:
        if not self._trained:
            raise RuntimeError("AutoencoderAugmenter.fit() must be called before transform()")
        t0 = time.perf_counter()
        X_t = torch.tensor(X, dtype=torch.float32, device=self._device)
        self._encoder.eval()
        with torch.no_grad():
            X_new = self._encoder(X_t).cpu().numpy()
        elapsed = time.perf_counter() - t0
        return _make_result(X, X_new, self.name, elapsed)


class LearnedRFFAugmenter:
    """Random Fourier Features with learned frequencies.

    Feature map: phi(x) = sqrt(2/D) * cos(W @ x + b)
    W and b are optimized end-to-end with a linear head.
    """

    def __init__(self, n_components: int = 10, n_epochs: int = 200,
                 lr: float = 1e-3, seed: int = 42):
        if not TORCH_AVAILABLE:
            raise ImportError("torch is required for LearnedRFFAugmenter: uv add torch")
        self.n_components = n_components
        self.n_epochs = n_epochs
        self.lr = lr
        self._seed = seed
        self.name = f"learned_rff_{n_components}"
        self._device = _get_device()
        self._trained = False

    def _build_model(self, n_features: int):
        torch.manual_seed(self._seed)
        self._W = nn.Parameter(torch.randn(self.n_components, n_features, device=self._device) * 0.5)
        self._b = nn.Parameter(torch.rand(self.n_components, device=self._device) * 2 * 3.14159)
        self._head = nn.Linear(self.n_components, 1).to(self._device)

    def _phi(self, X_t):
        proj = X_t @ self._W.T + self._b
        return torch.cos(proj) * (2.0 / self.n_components) ** 0.5

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        self._build_model(X_train.shape[1])

        X_t = torch.tensor(X_train, dtype=torch.float32, device=self._device)
        y_t = torch.tensor(y_train, dtype=torch.float32, device=self._device).unsqueeze(1)

        params = [self._W, self._b] + list(self._head.parameters())
        optimizer = torch.optim.Adam(params, lr=self.lr)
        loss_fn = nn.MSELoss()

        for epoch in range(self.n_epochs):
            features = self._phi(X_t)
            pred = self._head(features)
            loss = loss_fn(pred, y_t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 50 == 0:
                print(f"  Learned RFF epoch {epoch + 1}/{self.n_epochs}, loss: {loss.item():.4f}")

        self._trained = True

    def transform(self, X: np.ndarray) -> AugmenterResult:
        if not self._trained:
            raise RuntimeError("LearnedRFFAugmenter.fit() must be called before transform()")
        t0 = time.perf_counter()
        X_t = torch.tensor(X, dtype=torch.float32, device=self._device)
        with torch.no_grad():
            X_new = self._phi(X_t).cpu().numpy()
        elapsed = time.perf_counter() - t0
        return _make_result(X, X_new, self.name, elapsed)
