"""
FlashGuard v4 — Automated Test Suite
======================================
Run with:
    pip install pytest pytest-flask
    pytest test_api_server.py -v
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd

# ── Make sure dotenv doesn't override test env ─────────────────────────────────
os.environ.setdefault("FLASK_ENV", "testing")
os.environ.setdefault("UPSTOX_TOKEN", "")
os.environ.setdefault("MODEL_THRESHOLD", "0.20")

# Import the app AFTER setting env vars
sys.path.insert(0, os.path.dirname(__file__))
from api_server import app, _norm, _engineer, _build_seq, _risk_band, _demo, _pick_features


# ═══════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def client():
    """Flask test client with testing mode on."""
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


@pytest.fixture
def sample_ohlcv():
    """A 200-row synthetic OHLCV DataFrame — representative of market data."""
    rng   = np.random.default_rng(0)
    n     = 200
    close = 1000.0 * np.cumprod(1 + rng.normal(0, 0.008, n))
    open_ = np.r_[close[0], close[:-1]] * (1 + rng.normal(0, 0.001, n))
    high  = np.maximum(open_, close) * (1 + rng.uniform(0.001, 0.01, n))
    low   = np.minimum(open_, close) * (1 - rng.uniform(0.001, 0.01, n))
    vol   = rng.integers(500_000, 5_000_000, n).astype(float)
    idx   = pd.date_range("2024-01-01", periods=n, freq="B")
    return pd.DataFrame({"Open": open_, "High": high, "Low": low,
                          "Close": close, "Volume": vol}, index=idx)


@pytest.fixture
def sample_ohlcv_lowercase(sample_ohlcv):
    """Same DataFrame but with lowercase column names (as some CSV exports produce)."""
    return sample_ohlcv.rename(columns=str.lower)


# ═══════════════════════════════════════════════════════════════════════════════
# UNIT TESTS: _norm
# ═══════════════════════════════════════════════════════════════════════════════

class TestNorm:
    def test_uppercase_passthrough(self, sample_ohlcv):
        """Already-uppercase columns should pass through unchanged."""
        out = _norm(sample_ohlcv.copy())
        assert list(out.columns) == ["Open", "High", "Low", "Close", "Volume"]

    def test_lowercase_columns_renamed(self, sample_ohlcv_lowercase):
        """Lowercase columns (open, high…) must be renamed to Title-Case."""
        out = _norm(sample_ohlcv_lowercase.copy())
        assert "Open"  in out.columns
        assert "Close" in out.columns
        assert "Volume" in out.columns

    def test_no_data_loss(self, sample_ohlcv):
        """Shape must be exactly preserved after normalisation."""
        out = _norm(sample_ohlcv.copy())
        assert out.shape == sample_ohlcv.shape


# ═══════════════════════════════════════════════════════════════════════════════
# UNIT TESTS: _engineer
# ═══════════════════════════════════════════════════════════════════════════════

class TestEngineer:
    EXPECTED_COLS = [
        "return", "log_return", "volume_change",
        "volatility_5", "volatility_10", "volatility_20", "volatility",
        "momentum_5", "momentum_10", "momentum",
        "VWAP", "vwap_diff", "high_low_spread",
        "open_close_return", "turnover_change", "price_acceleration",
    ]

    def test_all_feature_columns_present(self, sample_ohlcv):
        eng = _engineer(sample_ohlcv)
        for col in self.EXPECTED_COLS:
            assert col in eng.columns, f"Missing column: {col}"

    def test_no_infinite_values(self, sample_ohlcv):
        eng = _engineer(sample_ohlcv)
        numeric = eng.select_dtypes(include=[np.number])
        assert not np.isinf(numeric.values).any(), "Infinite values found in engineered features"

    def test_return_calculation(self, sample_ohlcv):
        """return column must match pct_change of Close."""
        eng = _engineer(sample_ohlcv)
        expected = sample_ohlcv["Close"].pct_change()
        pd.testing.assert_series_equal(eng["return"], expected, check_names=False)

    def test_high_low_spread_positive(self, sample_ohlcv):
        """high_low_spread must be non-negative (High >= Low always)."""
        eng = _engineer(sample_ohlcv)
        spread = eng["high_low_spread"].dropna()
        assert (spread >= 0).all(), "high_low_spread contains negative values"

    def test_vwap_formula(self, sample_ohlcv):
        """VWAP should equal (H + L + C) / 3."""
        eng = _engineer(sample_ohlcv)
        expected_vwap = (sample_ohlcv["High"] + sample_ohlcv["Low"] + sample_ohlcv["Close"]) / 3.0
        pd.testing.assert_series_equal(
            eng["VWAP"].reset_index(drop=True),
            expected_vwap.reset_index(drop=True),
            check_names=False,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# UNIT TESTS: _build_seq
# ═══════════════════════════════════════════════════════════════════════════════

class TestBuildSeq:
    def test_output_shape_10_features(self, sample_ohlcv):
        seq, _ = _build_seq(sample_ohlcv, timesteps=30, n_feat=10)
        assert seq.shape == (1, 30, 10), f"Unexpected shape: {seq.shape}"

    def test_output_shape_5_features(self, sample_ohlcv):
        seq, _ = _build_seq(sample_ohlcv, timesteps=30, n_feat=5)
        assert seq.shape == (1, 30, 5), f"Unexpected shape: {seq.shape}"

    def test_output_dtype_float32(self, sample_ohlcv):
        seq, _ = _build_seq(sample_ohlcv, timesteps=30, n_feat=10)
        assert seq.dtype == np.float32

    def test_no_nan_in_output(self, sample_ohlcv):
        seq, _ = _build_seq(sample_ohlcv, timesteps=30, n_feat=10)
        assert not np.isnan(seq).any(), "NaN values found in sequence output"

    def test_no_inf_in_output(self, sample_ohlcv):
        seq, _ = _build_seq(sample_ohlcv, timesteps=30, n_feat=10)
        assert not np.isinf(seq).any(), "Inf values found in sequence output"

    def test_raises_insufficient_rows(self, sample_ohlcv):
        """Should raise ValueError when there are fewer rows than timesteps needed."""
        tiny_df = sample_ohlcv.head(10)
        with pytest.raises(ValueError, match="Need"):
            _build_seq(tiny_df, timesteps=30, n_feat=10)

    def test_standardised_output(self, sample_ohlcv):
        """StandardScaler is applied — mean of each feature across timesteps ≈ 0."""
        seq, _ = _build_seq(sample_ohlcv, timesteps=len(sample_ohlcv) - 20, n_feat=10)
        # Mean of each feature across time dimension should be near zero
        col_means = seq[0].mean(axis=0)
        assert np.allclose(col_means, 0, atol=1e-1), \
            f"Features not standardised properly, means: {col_means}"


# ═══════════════════════════════════════════════════════════════════════════════
# UNIT TESTS: _risk_band
# ═══════════════════════════════════════════════════════════════════════════════

class TestRiskBand:
    def test_high_risk_at_threshold(self):
        assert _risk_band(0.20, threshold=0.20) == "HIGH RISK"

    def test_high_risk_above_threshold(self):
        assert _risk_band(0.85, threshold=0.20) == "HIGH RISK"

    def test_elevated_below_threshold(self):
        # 0.20 * 0.65 = 0.13
        assert _risk_band(0.15, threshold=0.20) == "ELEVATED"

    def test_stable_well_below_threshold(self):
        assert _risk_band(0.01, threshold=0.20) == "STABLE"

    def test_stable_at_zero(self):
        assert _risk_band(0.0, threshold=0.20) == "STABLE"

    def test_custom_threshold(self):
        # threshold=0.50 → elevated zone starts at 0.325
        assert _risk_band(0.40, threshold=0.50) == "ELEVATED"
        assert _risk_band(0.55, threshold=0.50) == "HIGH RISK"
        assert _risk_band(0.10, threshold=0.50) == "STABLE"


# ═══════════════════════════════════════════════════════════════════════════════
# UNIT TESTS: _demo
# ═══════════════════════════════════════════════════════════════════════════════

class TestDemo:
    def test_returns_dataframe(self):
        df = _demo()
        assert isinstance(df, pd.DataFrame)

    def test_expected_columns(self):
        df = _demo()
        assert set(["Open","High","Low","Close","Volume"]).issubset(df.columns)

    def test_high_gte_low(self):
        df = _demo()
        assert (df["High"] >= df["Low"]).all()

    def test_reasonable_row_count(self):
        df = _demo(period="6mo")
        assert len(df) >= 30

    def test_reproducible_seed(self):
        """Values from the same seed must be identical.
        The index is excluded because it is anchored to pd.Timestamp.utcnow()
        and will shift slightly between two successive calls."""
        df1 = _demo(seed=42)
        df2 = _demo(seed=42)
        pd.testing.assert_frame_equal(
            df1.reset_index(drop=True),
            df2.reset_index(drop=True),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# INTEGRATION TESTS: API endpoints
# ═══════════════════════════════════════════════════════════════════════════════

class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        resp = client.get("/api/health")
        assert resp.status_code == 200

    def test_health_has_status_ok(self, client):
        data = client.get("/api/health").get_json()
        assert data["status"] == "ok"

    def test_health_has_keras_field(self, client):
        data = client.get("/api/health").get_json()
        assert "keras" in data

    def test_health_has_models_list(self, client):
        data = client.get("/api/health").get_json()
        assert isinstance(data["models"], list)

    def test_health_has_loaded_list(self, client):
        data = client.get("/api/health").get_json()
        assert isinstance(data["loaded"], list)


class TestModelsEndpoint:
    def test_models_returns_200(self, client):
        resp = client.get("/api/models")
        assert resp.status_code == 200

    def test_models_returns_list(self, client):
        data = client.get("/api/models").get_json()
        assert isinstance(data, list)


class TestPredictEndpoint:
    def test_predict_returns_400_without_tensorflow(self, client):
        """Without TF installed / real model, predict should return 400 gracefully."""
        resp = client.post("/api/predict", json={
            "ticker": "RELIANCE",
            "period": "6mo",
            "interval": "1d",
        })
        # Either 200 (model loaded) or 400 (no model) — never a 500 crash
        assert resp.status_code in (200, 400)

    def test_predict_response_has_error_or_probability(self, client):
        resp = client.post("/api/predict", json={
            "ticker": "TCS",
            "period": "6mo",
            "interval": "1d",
        })
        data = resp.get_json()
        assert "probability" in data or "error" in data


class TestPickFeatures:
    def test_10_features(self):
        feats = _pick_features(10)
        assert len(feats) == 10

    def test_5_features(self):
        feats = _pick_features(5)
        assert len(feats) == 5

    def test_14_features(self):
        feats = _pick_features(14)
        assert len(feats) == 14

    def test_unknown_falls_back_to_truncated_10(self):
        feats = _pick_features(7)
        assert len(feats) == 7  # slices FEATURES_10[:7]
