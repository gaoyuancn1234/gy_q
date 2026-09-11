#!/usr/bin/env python3
"""GatedFactorNet — 可学习 Gate 替代手工因子择时规则

架构:
  Market State [13] → Gate Network → gate_weights [8] → expand via group_mask → per_factor_weights [210]
  Factor Features [210] × per_factor_weights [210] → weighted_factors [210]
  weighted_factors → Prediction Head (MLP) → prediction [1]

Gate 用 Sigmoid (因子组不互斥，各组独立 scale [0,1])。
Sparse loss 鼓励 gate 稀疏，防退化为全 1。

Exp 015b
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ── 因子分组定义 ──

FACTOR_GROUPS = {
    "momentum": [
        # Alpha158 kbar/价格 + ROC + MA + KUP/KLOW + SUMP/SUMN/SUMD (涨跌天数)
        "KMID", "KMID2", "KLEN", "KSFT", "KSFT2", "KUP", "KUP2", "KLOW", "KLOW2",
        "ROC", "MA",  # prefix: ROC5..ROC60, MA5..MA60
        "OPEN0", "HIGH0", "LOW0",
        "SUMP", "SUMN", "SUMD",  # prefix: SUMP5..SUMD60
        "IMAX", "IMIN", "IMXD",  # prefix: index of max/min
        "RANK",  # prefix: RANK5..RANK60
        # 扩展动量因子
        "MOM_SKIP1_5", "MOM_SKIP1_10", "MOM_SKIP1_20", "MOM_3M_1M",
        "REVERSION_5", "REVERSION_20", "UP_RATIO_20",
        "QA_MULTI_PERIOD_SIGN_CONSENSUS", "QA_MOMENTUM_DECAY_NORMED",
    ],
    "volatility": [
        # Alpha158: STD, BETA, RSQR
        "STD", "BETA", "RSQR",  # prefix: STD5..STD60, BETA5..BETA60, RSQR5..RSQR60
        # 扩展波动率因子
        "ATR_14", "ATR_RATIO", "GK_VOL_20",
        "INTRADAY_RANGE", "INTRADAY_RANGE_MA5",
        "VOL_RATIO_5_20", "HIGH_LOW_CORR_10", "CLOSE_RANGE_POS",
        "QA_COMPRESSION_BREAKOUT",
    ],
    "volume": [
        # Alpha158: VMA, VSTD, WVMA, VSUMP/VSUMN/VSUMD
        "VMA", "VSTD", "WVMA",  # prefix: VMA5..VMA60, VSTD5..VSTD60
        "VSUMP", "VSUMN", "VSUMD",  # prefix: VSUMP5..VSUMD60
        # 扩展换手率/成交因子
        "TURN_MA5", "TURN_MA10", "TURN_MA20", "TURN_MA60",
        "TURN_STD20", "TURN_SURGE", "TURN_RATIO_5_20", "TURN_RANK_5",
        "VOLUME_SURPRISE", "AMT_SURPRISE",
        "QA_VOL_CONCENTRATION_DRIFT", "QA_VOL_DISPERSION_MEAN_REV",
        "QA_VOLCONC_PRICEMOM_DIV",
    ],
    "price_volume_corr": [
        # Alpha158: CORD, CORR, CNTP, CNTN, CNTD
        "CORD", "CORR", "CNTP", "CNTN", "CNTD",  # prefix match
        # 扩展量价相关因子
        "PV_CORR_10", "PV_CORR_20", "PV_CORR_CHANGE",
        "AMT_PRICE_CORR_10",
        "QA_PVCORR_DELTA_VOL", "QA_UPDAY_VOL_RATIO",
        "QA_UPDOWN_VOL_DIFF_10", "QA_UPVOL_SPREAD_NORM",
        "QA_SELLPRESS_RATIO", "QA_VOL_PRICE_DIVERGENCE_SLOPE",
        "QA_ELASTICITY_FLIP_PERSISTENCE",
    ],
    "vwap": [
        "VWAP", "VWAP_RATIO", "VWAP_MA5_RATIO", "VWAP_MA10_RATIO", "VWAP_MA20_RATIO",
        "QA_VWAP_ESCAPE_VELOCITY", "QA_VWAP_RANGE_POS", "QA_VWAP_GRAVITY_DRIFT",
        "QA_VWAP_DECAY_ACCELERATION",
    ],
    "valuation": [
        "PE_TTM", "PB", "PS_TTM", "EP", "BP", "SP",
        "PE_MA20", "PE_STD20", "PE_ZSCORE", "PB_ZSCORE",
        "PE_CHANGE_20", "PB_CHANGE_20", "PE_PRICE_CORR",
        "EP_MOM20", "BP_TURN", "MV_VOL",
    ],
    "market_cap": [
        "LOG_MV", "LOG_CIRC_MV", "MV_RATIO", "MV_RANK", "NEG_LOG_MV", "MV_CHANGE_20",
    ],
    "microstructure": [
        # Alpha158: RESI, MAX, MIN, QTLU, QTLD, RSV
        "RESI", "MAX", "MIN", "QTLU", "QTLD", "RSV",  # prefix match
        # 扩展微观结构因子
        "QA_SHADOW_ASYMMETRY", "QA_CHANNEL_TURN_MOMENTUM",
        "QA_QUIET_VOL_SURGE_20", "QA_VOL_HIGHLOW_ASYM",
        "QA_ASYMMETRIC_REBOUND_STRENGTH", "QA_INTRADAY_PATH_ASYM",
        "QA_LIQUIDITY_RESILIENCE_RATIO", "QA_CLOSING_IMPACT_DECAY",
        "QA_CONCAVITY_CONSISTENCY_10", "QA_DOWNVOL_ASYM_RATIO",
        "UP_VOL_RATIO_20",
        "WQ_ALPHA6", "WQ_ALPHA12", "WQ_ALPHA15", "WQ_ALPHA22",
        "WQ_ALPHA26", "WQ_ALPHA28", "WQ_ALPHA33", "WQ_ALPHA38",
        "WQ_ALPHA41", "WQ_ALPHA45", "WQ_ALPHA54", "WQ_ALPHA68",
        "WQ_ALPHA73", "WQ_ALPHA84", "WQ_ALPHA101",
    ],
}

N_GROUPS = len(FACTOR_GROUPS)
GROUP_NAMES = list(FACTOR_GROUPS.keys())


def classify_all_factors(feature_names: list[str]) -> dict[str, int]:
    """将因子名映射到组 index。未匹配因子放入 microstructure (catch-all)。

    匹配策略: 先精确匹配，再按前缀长度降序匹配 (更长前缀优先)。
    例如 VWAP_MA20_RATIO 精确匹配 vwap 组，VWAP0 前缀匹配 momentum 组中的 VWAP0。
    """
    # 精确匹配表
    exact_to_group = {}
    # 前缀匹配表 (按长度降序排列，确保长前缀优先)
    prefix_to_group = {}
    for gidx, (gname, members) in enumerate(FACTOR_GROUPS.items()):
        for m in members:
            exact_to_group[m] = gidx
            prefix_to_group[m] = gidx

    # 按长度降序排列前缀，确保 VWAP_MA5_RATIO > VWAP > V
    sorted_prefixes = sorted(prefix_to_group.keys(), key=len, reverse=True)

    result = {}
    catchall = GROUP_NAMES.index("microstructure")
    for fname in feature_names:
        # 1. 精确匹配
        if fname in exact_to_group:
            result[fname] = exact_to_group[fname]
            continue

        # 2. 前缀匹配 (长度降序)
        matched = False
        for prefix in sorted_prefixes:
            if fname.startswith(prefix):
                result[fname] = prefix_to_group[prefix]
                matched = True
                break

        if not matched:
            result[fname] = catchall

    return result


def build_group_mask(feature_names: list[str]) -> torch.Tensor:
    """构建 (n_features, n_groups) 二进制矩阵。

    group_mask[i, g] = 1 if feature i belongs to group g.
    """
    mapping = classify_all_factors(feature_names)
    n_features = len(feature_names)
    mask = torch.zeros(n_features, N_GROUPS)
    for i, fname in enumerate(feature_names):
        gidx = mapping[fname]
        mask[i, gidx] = 1.0
    return mask


# ── Market State 计算 ──

class MarketStateComputer:
    """计算 market state 特征 (13 维): regime one-hot(3) + ma_ratio(1) + index_ret(1) + index_vol(1) + rolling_ic(7)

    仅用历史数据，防前视偏差。per-date 计算，broadcast 到当日所有股票。
    """

    def __init__(self, feature_names: list[str]):
        self.feature_names = feature_names
        self.group_mapping = classify_all_factors(feature_names)
        self._index_data = None

    def _load_index_data(self) -> pd.Series:
        """加载 CSI300 指数收盘价序列"""
        if self._index_data is not None:
            return self._index_data
        from qlib.data import D
        df = D.features(
            ["SH000300"],
            ["$close"],
            start_time="2016-01-01",
            end_time="2027-01-01",
        )
        if isinstance(df.index, pd.MultiIndex):
            self._index_data = df.droplevel(0)["$close"]
        else:
            self._index_data = df["$close"]
        return self._index_data

    def _detect_regime(self, index_close: pd.Series, date: pd.Timestamp) -> np.ndarray:
        """简易 regime 检测: MA10 vs MA60 交叉"""
        hist = index_close[index_close.index <= date].tail(70)
        if len(hist) < 60:
            return np.array([0, 0, 1], dtype=np.float32)  # sideways

        ma10 = hist.tail(10).mean()
        ma60 = hist.tail(60).mean()
        ratio = ma10 / ma60 - 1

        if ratio > 0.02:
            return np.array([1, 0, 0], dtype=np.float32)  # bull
        elif ratio < -0.02:
            return np.array([0, 1, 0], dtype=np.float32)  # bear
        return np.array([0, 0, 1], dtype=np.float32)  # sideways

    def _compute_index_features(self, index_close: pd.Series, date: pd.Timestamp) -> np.ndarray:
        """MA ratio + 20d return + 20d volatility"""
        hist = index_close[index_close.index <= date].tail(70)
        if len(hist) < 60:
            return np.zeros(3, dtype=np.float32)

        ma10 = hist.tail(10).mean()
        ma60 = hist.tail(60).mean()
        ma_ratio = float(ma10 / ma60 - 1)

        ret_20d = float(hist.iloc[-1] / hist.iloc[-21] - 1) if len(hist) >= 21 else 0.0
        returns = hist.pct_change().dropna().tail(20)
        vol_20d = float(returns.std() * (252 ** 0.5)) if len(returns) >= 10 else 0.0

        return np.array([ma_ratio, ret_20d, vol_20d], dtype=np.float32)

    def compute_all_dates_fast(self, factor_df: pd.DataFrame, label: pd.Series) -> pd.DataFrame:
        """批量计算所有日期的 market state 特征 (优化版)。

        使用 groupby + 向量化 rank correlation 替代逐日 loc + spearman。

        Returns:
            DataFrame indexed by date, 13 columns.
        """
        from scipy.stats import rankdata

        index_close = self._load_index_data()
        dates = factor_df.index.get_level_values(0).unique().sort_values()

        # 1. Regime + index features (per date, fast)
        regime_feats = {}
        for dt in dates:
            regime = self._detect_regime(index_close, dt)
            idx_feat = self._compute_index_features(index_close, dt)
            regime_feats[dt] = np.concatenate([regime, idx_feat])

        # 2. Rolling IC — 向量化批量计算
        # 预计算组-因子列映射
        group_col_indices = {}
        all_cols = list(factor_df.columns)
        for gidx in range(N_GROUPS - 1):
            member_cols = [f for f in self.feature_names if self.group_mapping.get(f, -1) == gidx]
            indices = [all_cols.index(c) for c in member_cols if c in all_cols]
            group_col_indices[gidx] = indices

        # 预计算每组的组内均值 → 逐日 IC
        # 先按日期 groupby，然后批量计算
        n_groups_ic = N_GROUPS - 1
        n_dates = len(dates)
        ic_matrix = np.zeros((n_dates, n_groups_ic), dtype=np.float32)

        # 预转为 numpy 加速
        factor_vals = factor_df.values  # (n_samples, n_features)
        if isinstance(label, pd.DataFrame):
            label_vals = label.iloc[:, 0].values
        else:
            label_vals = label.values

        # 构建日期索引映射
        date_level = factor_df.index.get_level_values(0)
        label_date_level = label.index.get_level_values(0) if isinstance(label.index, pd.MultiIndex) else label.index

        # 用 groupby 获取每日的行范围 (排序后连续)
        # 确保按日期排序
        date_breaks = np.where(np.diff(date_level.codes if hasattr(date_level, 'codes') else pd.factorize(date_level)[0]))[0] + 1
        date_starts = np.concatenate([[0], date_breaks])
        date_ends = np.concatenate([date_breaks, [len(factor_vals)]])

        # label 对齐 — 假设 factor_df 和 label 有相同的 index
        for di in range(n_dates):
            s, e = date_starts[di], date_ends[di]
            if e - s < 10:  # 样本太少跳过
                continue

            day_factors = factor_vals[s:e]  # (n_stocks, n_features)
            day_labels = label_vals[s:e]    # (n_stocks,)

            # 跳过全 NaN
            valid_mask = ~np.isnan(day_labels)
            if valid_mask.sum() < 10:
                continue

            label_rank = rankdata(day_labels[valid_mask])

            for gidx in range(n_groups_ic):
                col_idx = group_col_indices[gidx]
                if not col_idx:
                    continue
                # 组内均值
                group_mean = np.nanmean(day_factors[valid_mask][:, col_idx], axis=1)
                if np.all(np.isnan(group_mean)):
                    continue
                # nan 处理
                gm_valid = ~np.isnan(group_mean)
                if gm_valid.sum() < 10:
                    continue
                factor_rank = rankdata(group_mean[gm_valid])
                label_rank_sub = label_rank[gm_valid]
                # Pearson on ranks = Spearman
                n = len(factor_rank)
                ic = np.corrcoef(factor_rank, label_rank_sub)[0, 1]
                if not np.isnan(ic):
                    ic_matrix[di, gidx] = ic

        # Rolling 60 日均值 (向量化)
        window = 60
        rolling_ic_matrix = np.zeros_like(ic_matrix)
        for i in range(n_dates):
            start = max(0, i - window + 1)
            rolling_ic_matrix[i] = ic_matrix[start:i + 1].mean(axis=0)

        # 3. 组合
        columns = (
            ['regime_bull', 'regime_bear', 'regime_sideways']
            + ['ma_ratio', 'index_ret_20d', 'index_vol_20d']
            + [f'rolling_ic_{g}' for g in GROUP_NAMES[:-1]]
        )
        rows = {}
        for i, dt in enumerate(dates):
            rows[dt] = np.concatenate([regime_feats[dt], rolling_ic_matrix[i]])

        return pd.DataFrame.from_dict(rows, orient='index', columns=columns)


# ── PyTorch 模型组件 ──

class MarketGate(nn.Module):
    """Gate 网络: market_state [13] → gate_weights [8]"""

    def __init__(self, n_state: int = 13, n_groups: int = N_GROUPS, hidden: int = 32, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_state, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_groups),
            nn.Sigmoid(),
        )

    def forward(self, market_state: torch.Tensor) -> torch.Tensor:
        """market_state: (batch, n_state) → gate_weights: (batch, n_groups)"""
        return self.net(market_state)


class PredictionHead(nn.Module):
    """MLP 预测头: weighted_factors [n_features] → prediction [1]"""

    def __init__(self, n_features: int, hidden1: int = 128, hidden2: int = 64, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden1),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden2),
            nn.Dropout(dropout),
            nn.Linear(hidden2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class GatedFactorNet(nn.Module):
    """完整的 Gated Factor Network。

    forward:
        market_state: (batch, 13)
        factors: (batch, n_features)
        → prediction: (batch,)

    gate_weights 通过 group_mask 展开到 per-factor weights.
    """

    def __init__(self, n_features: int, group_mask: torch.Tensor,
                 n_state: int = 13, gate_hidden: int = 32, gate_dropout: float = 0.2,
                 head_hidden1: int = 128, head_hidden2: int = 64, head_dropout: float = 0.3):
        super().__init__()
        self.register_buffer('group_mask', group_mask)  # (n_features, n_groups)
        self.gate = MarketGate(n_state, group_mask.shape[1], gate_hidden, gate_dropout)
        self.head = PredictionHead(n_features, head_hidden1, head_hidden2, head_dropout)

    def forward(self, market_state: torch.Tensor, factors: torch.Tensor) -> torch.Tensor:
        gate_weights = self.gate(market_state)  # (batch, n_groups)
        # expand: (batch, n_groups) @ (n_groups, n_features) → (batch, n_features)
        per_factor_w = gate_weights @ self.group_mask.T  # (batch, n_features)
        weighted = factors * per_factor_w
        return self.head(weighted)

    def get_gate_weights(self, market_state: torch.Tensor) -> torch.Tensor:
        """提取 gate 权重 (用于 gate_lgb 策略)"""
        self.eval()
        with torch.no_grad():
            return self.gate(market_state)


# ── Dataset ──

class GatedFactorDataset(Dataset):
    """打包 market_state + factor_features + label。"""

    def __init__(self, market_state: np.ndarray, factors: np.ndarray, labels: np.ndarray):
        self.market_state = torch.tensor(market_state, dtype=torch.float32)
        self.factors = torch.tensor(factors, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.market_state[idx], self.factors[idx], self.labels[idx]


# ── 训练循环 ──

def train_gated_model(
    train_state: np.ndarray, train_factors: np.ndarray, train_labels: np.ndarray,
    valid_state: np.ndarray, valid_factors: np.ndarray, valid_labels: np.ndarray,
    group_mask: torch.Tensor,
    n_epochs: int = 100,
    batch_size: int = 4096,
    lr_gate: float = 1e-3,
    lr_head: float = 5e-4,
    weight_decay: float = 1e-4,
    lambda_sparse: float = 0.001,
    patience: int = 15,
    grad_clip: float = 1.0,
    verbose: bool = True,
) -> tuple[GatedFactorNet, dict]:
    """训练 GatedFactorNet。

    Returns:
        (model, info_dict) 其中 info_dict 含 best_epoch, best_val_loss 等
    """
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    n_features = train_factors.shape[1]
    model = GatedFactorNet(n_features, group_mask.to(device)).to(device)

    # 差异化学习率
    gate_params = list(model.gate.parameters())
    head_params = list(model.head.parameters())
    optimizer = torch.optim.AdamW([
        {'params': gate_params, 'lr': lr_gate},
        {'params': head_params, 'lr': lr_head},
    ], weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs // 2)

    train_ds = GatedFactorDataset(train_state, train_factors, train_labels)
    valid_ds = GatedFactorDataset(valid_state, valid_factors, valid_labels)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size * 2, shuffle=False)

    best_val_loss = float('inf')
    best_state = None
    best_epoch = 0
    no_improve = 0

    for epoch in range(1, n_epochs + 1):
        # Train
        model.train()
        total_loss = 0.0
        n_batches = 0
        for state_b, factors_b, labels_b in train_loader:
            state_b, factors_b, labels_b = state_b.to(device), factors_b.to(device), labels_b.to(device)

            pred = model(state_b, factors_b)
            mse_loss = nn.functional.mse_loss(pred, labels_b)

            # Sparse regularization on gate weights
            gate_w = model.gate(state_b)
            sparse_loss = gate_w.mean()

            loss = mse_loss + lambda_sparse * sparse_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_train_loss = total_loss / max(n_batches, 1)

        # Validate
        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for state_b, factors_b, labels_b in valid_loader:
                state_b, factors_b, labels_b = state_b.to(device), factors_b.to(device), labels_b.to(device)
                pred = model(state_b, factors_b)
                val_loss += nn.functional.mse_loss(pred, labels_b).item()
                val_batches += 1

        avg_val_loss = val_loss / max(val_batches, 1)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if verbose and epoch % 10 == 0:
            print(f"    Epoch {epoch:3d}: train_loss={avg_train_loss:.6f}  val_loss={avg_val_loss:.6f}  "
                  f"best={best_epoch} (no_improve={no_improve})")

        if no_improve >= patience:
            if verbose:
                print(f"    Early stopping at epoch {epoch} (best={best_epoch})")
            break

    # Restore best
    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.to(device)

    info = {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "total_epochs": epoch,
        "device": str(device),
    }
    return model, info


def train_mlp_no_gate(
    train_factors: np.ndarray, train_labels: np.ndarray,
    valid_factors: np.ndarray, valid_labels: np.ndarray,
    n_epochs: int = 100,
    batch_size: int = 4096,
    lr: float = 5e-4,
    weight_decay: float = 1e-4,
    patience: int = 15,
    grad_clip: float = 1.0,
    verbose: bool = True,
) -> tuple[PredictionHead, dict]:
    """训练不带 Gate 的 MLP (消融对比)。"""
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    n_features = train_factors.shape[1]
    model = PredictionHead(n_features).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs // 2)

    train_ds = torch.utils.data.TensorDataset(
        torch.tensor(train_factors, dtype=torch.float32),
        torch.tensor(train_labels, dtype=torch.float32),
    )
    valid_ds = torch.utils.data.TensorDataset(
        torch.tensor(valid_factors, dtype=torch.float32),
        torch.tensor(valid_labels, dtype=torch.float32),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size * 2, shuffle=False)

    best_val_loss = float('inf')
    best_state = None
    best_epoch = 0
    no_improve = 0

    for epoch in range(1, n_epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for factors_b, labels_b in train_loader:
            factors_b, labels_b = factors_b.to(device), labels_b.to(device)
            pred = model(factors_b)
            loss = nn.functional.mse_loss(pred, labels_b)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        scheduler.step()

        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for factors_b, labels_b in valid_loader:
                factors_b, labels_b = factors_b.to(device), labels_b.to(device)
                pred = model(factors_b)
                val_loss += nn.functional.mse_loss(pred, labels_b).item()
                val_batches += 1

        avg_val_loss = val_loss / max(val_batches, 1)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if verbose and epoch % 10 == 0:
            print(f"    Epoch {epoch:3d}: val_loss={avg_val_loss:.6f}  best={best_epoch}")

        if no_improve >= patience:
            if verbose:
                print(f"    Early stopping at epoch {epoch} (best={best_epoch})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.to(device)

    return model, {"best_epoch": best_epoch, "best_val_loss": best_val_loss, "total_epochs": epoch}
