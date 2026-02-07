"""
交易框架配置文件 - 优化版
策略：严格择时 + 精选蓝筹
回测：近6个月滚动3个月年化 > 6%
"""

# ============ 股票池配置 ============
# 股票池策略: 'static'=静态, 'dynamic'=动态获取
STOCK_POOL_STRATEGY = 'dynamic'

# 动态股票池参数
STOCK_POOL_CONFIG = {
    'source': 'hs300',          # 'hs300', 'zz500', 'sz50', 'custom'
    'style': 'conservative',     # 'conservative', 'balanced', 'aggressive'
    'update_freq': 7,            # 更新频率（天）
    'exclude_st': True,          # 排除ST
    'exclude_new_days': 60,      # 排除上市不足N天的新股
}

# 静态股票池（当STOCK_POOL_STRATEGY='static'时使用）
STOCK_POOL_STATIC = [
    # 消费（高稳定性）
    'sh.600519', 'sz.000858', 'sh.600887', 'sz.000568',
    'sz.002304', 'sh.600809', 'sz.000651', 'sz.000333',
    'sz.000895', 'sh.600298', 'sz.000596', 'sh.603288',
    # 医药
    'sh.600276', 'sz.000538', 'sh.600196', 'sz.002415',
    # 金融（精选）
    'sh.601318', 'sh.600036',
    # 其他蓝筹
    'sh.600031', 'sh.600050', 'sz.000001',
]


def get_stock_pool():
    """获取股票池（支持动态更新）"""
    if STOCK_POOL_STRATEGY == 'static':
        return STOCK_POOL_STATIC

    try:
        from data.stock_pool import build_stock_pool
        return build_stock_pool(STOCK_POOL_CONFIG['style'])
    except Exception as e:
        print(f"动态获取股票池失败: {e}, 使用静态配置")
        return STOCK_POOL_STATIC


# 兼容性：STOCK_POOL 默认使用静态配置
STOCK_POOL = STOCK_POOL_STATIC

# ============ 时间配置 ============
TRAIN_YEARS = 1          # 训练窗口（年）- 缩短以适应市场变化
VALID_MONTHS = 2         # 验证窗口（月）
RETRAIN_FREQ = 'M'       # 重训频率: 'M'=月度
LOOKBACK_DAYS = 60       # 因子计算所需历史天数

# ============ 交易配置 ============
TRADE_FREQ = 'W'         # 交易频率: 'W'=周度
TOP_K = 8                # 持仓股票数量（减少以提高质量）
POSITION_LIMIT = 0.15    # 单只股票最大仓位
CASH_RESERVE = 0.05      # 现金保留比例
PRED_THRESHOLD = 0.01    # 预测得分门槛（新增）

# ============ 择时配置（新增）============
USE_TIMING = True        # 是否使用择时
TIMING_PARAMS = {
    'ma_short': 5,       # 短期均线
    'ma_mid': 20,        # 中期均线
    'ma_long': 60,       # 长期均线
    'uptrend_pos': 1.0,  # 上涨趋势仓位
    'range_pos': 0.3,    # 震荡仓位
    'downtrend_pos': 0.0 # 下跌趋势仓位（空仓）
}

# ============ 成本配置 ============
COMMISSION = 0.0003      # 佣金费率 (双向)
STAMP_TAX = 0.001        # 印花税 (仅卖出)
SLIPPAGE = 0.001         # 滑点估计

# ============ 风控配置 ============
MAX_DRAWDOWN = 0.10      # 最大回撤阈值（收紧）
STOP_LOSS = 0.08         # 个股止损线
TAKE_PROFIT = 0.25       # 个股止盈线（收紧）
INDUSTRY_LIMIT = 0.30    # 单一行业最大占比

# ============ 模型配置 ============
MODEL_TYPE = 'lightgbm'
MODEL_PARAMS = {
    'lightgbm': {
        'n_estimators': 100,
        'max_depth': 4,        # 降低以减少过拟合
        'learning_rate': 0.05,
        'min_child_samples': 30,  # 增加以提高稳定性
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'verbose': -1
    },
    'ridge': {
        'alpha': 10.0
    }
}
