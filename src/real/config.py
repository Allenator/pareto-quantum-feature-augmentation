"""Configuration dataclasses for the real financial data experiment."""

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True)
class RealDataConfig:
    """Real financial data configuration."""
    tickers: tuple[str, ...] = (
        "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL",
        "META", "BRK-B", "LLY", "JPM", "AVGO",
    )
    market_ticker: str = "^GSPC"
    start_date: str = "2022-01-01"
    end_date: str = "2025-12-31"
    prediction_horizon: int = 5  # 5-day forward excess return
    data_dir: str = "data/real"


@dataclass(frozen=True)
class BacktestConfig:
    """Walk-forward backtesting configuration."""
    train_window_days: int = 504  # ~2 years of trading days
    step_days: int = 1  # roll forward by 1 trading day
    min_train_samples: int = 200  # skip window if too few samples
    gap_days: int = 5  # gap between train end and test (= prediction horizon)


@dataclass(frozen=True)
class ExperimentConfig:
    """Top-level experiment configuration for real data."""
    data: RealDataConfig
    backtest: BacktestConfig
    augmenters: list  # list[AugmenterConfig] from src.synthetic.config
    models: list  # list[ModelConfig] from src.synthetic.config
    results_dir: str = "results/real"
    features_dir: str = "features/real"
    run_id: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%dT%H%M%S"))
    clip_range: float | None = 5.0
