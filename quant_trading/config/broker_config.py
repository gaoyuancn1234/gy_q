"""
券商接口配置
"""

from dataclasses import dataclass
from typing import Optional
from enum import Enum


class BrokerType(Enum):
    """券商类型"""
    # A股券商
    HUATAI = "huatai"           # 华泰证券
    ZHONGXIN = "zhongxin"       # 中信证券
    GUOTAI = "guotai"           # 国泰君安
    HAITONG = "haitong"         # 海通证券
    EASYTRADER = "easytrader"   # 通用券商（通过easytrader）

    # 港股/美股券商
    FUTU = "futu"               # 富途证券
    TIGER = "tiger"             # 老虎证券
    IB = "ib"                   # 盈透证券
    LONGBRIDGE = "longbridge"   # 长桥证券


@dataclass
class BrokerConfig:
    """券商配置基类"""
    broker_type: BrokerType
    account: str = ""
    password: str = ""

    # API配置
    api_host: str = ""
    api_port: int = 0

    # 是否启用
    enabled: bool = False

    # 超时设置（秒）
    timeout: int = 30

    # 重试次数
    retry_times: int = 3


@dataclass
class AStockBrokerConfig(BrokerConfig):
    """A股券商配置"""

    # 客户端路径（用于easytrader）
    client_path: str = ""

    # 交易密码（可能与登录密码不同）
    trade_password: str = ""

    # 验证码处理方式
    captcha_handler: str = "manual"  # manual, ocr, api

    # 是否使用同花顺客户端
    use_ths: bool = False

    # 委托方式
    order_type: str = "limit"  # limit, market

    @classmethod
    def huatai(cls, account: str, password: str, **kwargs) -> 'AStockBrokerConfig':
        """华泰证券配置"""
        return cls(
            broker_type=BrokerType.HUATAI,
            account=account,
            password=password,
            api_host="https://xtquant.huatai.com",
            api_port=443,
            enabled=True,
            **kwargs
        )

    @classmethod
    def easytrader(cls, account: str, password: str, client_path: str, **kwargs) -> 'AStockBrokerConfig':
        """通用券商配置（使用easytrader）"""
        return cls(
            broker_type=BrokerType.EASYTRADER,
            account=account,
            password=password,
            client_path=client_path,
            enabled=True,
            **kwargs
        )


@dataclass
class HKStockBrokerConfig(BrokerConfig):
    """港股券商配置"""

    # API密钥
    app_id: str = ""
    app_secret: str = ""

    # RSA私钥路径（用于富途等）
    rsa_key_path: str = ""

    # 交易市场
    market: str = "HK"  # HK, US

    # 环境
    environment: str = "REAL"  # REAL, SIMULATE

    @classmethod
    def futu(cls, app_id: str, app_secret: str, rsa_key_path: str = "", **kwargs) -> 'HKStockBrokerConfig':
        """富途证券配置"""
        return cls(
            broker_type=BrokerType.FUTU,
            app_id=app_id,
            app_secret=app_secret,
            rsa_key_path=rsa_key_path,
            api_host="127.0.0.1",
            api_port=11111,
            enabled=True,
            **kwargs
        )

    @classmethod
    def tiger(cls, app_id: str, app_secret: str, **kwargs) -> 'HKStockBrokerConfig':
        """老虎证券配置"""
        return cls(
            broker_type=BrokerType.TIGER,
            app_id=app_id,
            app_secret=app_secret,
            api_host="openapi.tigerbrokers.com",
            api_port=443,
            enabled=True,
            **kwargs
        )

    @classmethod
    def longbridge(cls, app_id: str, app_secret: str, **kwargs) -> 'HKStockBrokerConfig':
        """长桥证券配置"""
        return cls(
            broker_type=BrokerType.LONGBRIDGE,
            app_id=app_id,
            app_secret=app_secret,
            api_host="openapi.longbridgeapp.com",
            api_port=443,
            enabled=True,
            **kwargs
        )

    @classmethod
    def interactive_brokers(cls, account: str, password: str, **kwargs) -> 'HKStockBrokerConfig':
        """盈透证券配置"""
        return cls(
            broker_type=BrokerType.IB,
            account=account,
            password=password,
            api_host="127.0.0.1",
            api_port=7497,  # TWS端口
            enabled=True,
            **kwargs
        )
