"""
告警系统
支持多种通知渠道
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional, Callable
from enum import Enum
import threading
import queue
import logging
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """告警级别"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertChannel(Enum):
    """告警渠道"""
    LOG = "log"
    EMAIL = "email"
    WEBHOOK = "webhook"
    WECHAT = "wechat"
    DINGTALK = "dingtalk"
    SMS = "sms"


@dataclass
class Alert:
    """告警信息"""
    id: str
    level: AlertLevel
    title: str
    message: str
    source: str = ""
    code: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    acknowledged: bool = False

    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'level': self.level.value,
            'title': self.title,
            'message': self.message,
            'source': self.source,
            'code': self.code,
            'data': self.data,
            'timestamp': str(self.timestamp),
            'acknowledged': self.acknowledged
        }

    def __str__(self):
        return f"[{self.level.value.upper()}] {self.title}: {self.message}"


class AlertManager:
    """
    告警管理器
    管理告警的生成、分发和通知
    """

    def __init__(self):
        self._alerts: List[Alert] = []
        self._alert_queue: queue.Queue = queue.Queue()
        self._handlers: Dict[AlertChannel, Callable] = {}
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._alert_counter = 0

        # 告警配置
        self._channel_config: Dict[AlertChannel, Dict] = {}
        self._level_channels: Dict[AlertLevel, List[AlertChannel]] = {
            AlertLevel.INFO: [AlertChannel.LOG],
            AlertLevel.WARNING: [AlertChannel.LOG],
            AlertLevel.ERROR: [AlertChannel.LOG, AlertChannel.EMAIL],
            AlertLevel.CRITICAL: [AlertChannel.LOG, AlertChannel.EMAIL, AlertChannel.WEBHOOK]
        }

        # 注册默认处理器
        self._handlers[AlertChannel.LOG] = self._log_handler

    def configure_email(
        self,
        smtp_host: str,
        smtp_port: int,
        username: str,
        password: str,
        from_addr: str,
        to_addrs: List[str],
        use_ssl: bool = True
    ):
        """配置邮件通知"""
        self._channel_config[AlertChannel.EMAIL] = {
            'smtp_host': smtp_host,
            'smtp_port': smtp_port,
            'username': username,
            'password': password,
            'from_addr': from_addr,
            'to_addrs': to_addrs,
            'use_ssl': use_ssl
        }
        self._handlers[AlertChannel.EMAIL] = self._email_handler
        logger.info("邮件通知已配置")

    def configure_webhook(self, url: str, headers: Dict[str, str] = None):
        """配置Webhook通知"""
        self._channel_config[AlertChannel.WEBHOOK] = {
            'url': url,
            'headers': headers or {'Content-Type': 'application/json'}
        }
        self._handlers[AlertChannel.WEBHOOK] = self._webhook_handler
        logger.info("Webhook通知已配置")

    def configure_dingtalk(self, webhook_url: str, secret: str = ""):
        """配置钉钉机器人通知"""
        self._channel_config[AlertChannel.DINGTALK] = {
            'webhook_url': webhook_url,
            'secret': secret
        }
        self._handlers[AlertChannel.DINGTALK] = self._dingtalk_handler
        logger.info("钉钉通知已配置")

    def configure_wechat(self, corp_id: str, agent_id: str, secret: str, to_users: List[str]):
        """配置企业微信通知"""
        self._channel_config[AlertChannel.WECHAT] = {
            'corp_id': corp_id,
            'agent_id': agent_id,
            'secret': secret,
            'to_users': to_users
        }
        self._handlers[AlertChannel.WECHAT] = self._wechat_handler
        logger.info("企业微信通知已配置")

    def set_level_channels(self, level: AlertLevel, channels: List[AlertChannel]):
        """设置告警级别对应的通知渠道"""
        self._level_channels[level] = channels

    def start(self):
        """启动告警处理"""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._process_loop, daemon=True)
        self._thread.start()
        logger.info("告警管理器已启动")

    def stop(self):
        """停止告警处理"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("告警管理器已停止")

    def _process_loop(self):
        """告警处理循环"""
        while self._running:
            try:
                alert = self._alert_queue.get(timeout=1)
                self._dispatch(alert)
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"处理告警失败: {e}")

    def alert(
        self,
        level: AlertLevel,
        title: str,
        message: str,
        source: str = "",
        code: str = "",
        data: Dict[str, Any] = None
    ) -> Alert:
        """
        创建告警

        Args:
            level: 告警级别
            title: 告警标题
            message: 告警内容
            source: 告警来源
            code: 相关股票代码
            data: 附加数据

        Returns:
            告警对象
        """
        self._alert_counter += 1
        alert = Alert(
            id=f"ALT{self._alert_counter:06d}",
            level=level,
            title=title,
            message=message,
            source=source,
            code=code,
            data=data or {}
        )

        self._alerts.append(alert)
        self._alert_queue.put(alert)

        # 保留最近1000条告警
        if len(self._alerts) > 1000:
            self._alerts = self._alerts[-1000:]

        return alert

    def info(self, title: str, message: str, **kwargs) -> Alert:
        """信息告警"""
        return self.alert(AlertLevel.INFO, title, message, **kwargs)

    def warning(self, title: str, message: str, **kwargs) -> Alert:
        """警告告警"""
        return self.alert(AlertLevel.WARNING, title, message, **kwargs)

    def error(self, title: str, message: str, **kwargs) -> Alert:
        """错误告警"""
        return self.alert(AlertLevel.ERROR, title, message, **kwargs)

    def critical(self, title: str, message: str, **kwargs) -> Alert:
        """严重告警"""
        return self.alert(AlertLevel.CRITICAL, title, message, **kwargs)

    def _dispatch(self, alert: Alert):
        """分发告警"""
        channels = self._level_channels.get(alert.level, [AlertChannel.LOG])

        for channel in channels:
            handler = self._handlers.get(channel)
            if handler:
                try:
                    handler(alert)
                except Exception as e:
                    logger.error(f"告警处理失败 [{channel.value}]: {e}")

    def _log_handler(self, alert: Alert):
        """日志处理器"""
        level_map = {
            AlertLevel.INFO: logging.INFO,
            AlertLevel.WARNING: logging.WARNING,
            AlertLevel.ERROR: logging.ERROR,
            AlertLevel.CRITICAL: logging.CRITICAL
        }
        log_level = level_map.get(alert.level, logging.INFO)
        logger.log(log_level, f"[{alert.id}] {alert.title}: {alert.message}")

    def _email_handler(self, alert: Alert):
        """邮件处理器"""
        config = self._channel_config.get(AlertChannel.EMAIL)
        if not config:
            return

        try:
            msg = MIMEMultipart()
            msg['Subject'] = f"[{alert.level.value.upper()}] {alert.title}"
            msg['From'] = config['from_addr']
            msg['To'] = ', '.join(config['to_addrs'])

            body = f"""
告警ID: {alert.id}
告警级别: {alert.level.value}
告警时间: {alert.timestamp}
告警来源: {alert.source}
相关代码: {alert.code}

告警内容:
{alert.message}

附加数据:
{json.dumps(alert.data, indent=2, ensure_ascii=False)}
"""
            msg.attach(MIMEText(body, 'plain', 'utf-8'))

            if config['use_ssl']:
                server = smtplib.SMTP_SSL(config['smtp_host'], config['smtp_port'])
            else:
                server = smtplib.SMTP(config['smtp_host'], config['smtp_port'])
                server.starttls()

            server.login(config['username'], config['password'])
            server.sendmail(config['from_addr'], config['to_addrs'], msg.as_string())
            server.quit()

            logger.info(f"告警邮件已发送: {alert.id}")

        except Exception as e:
            logger.error(f"发送告警邮件失败: {e}")

    def _webhook_handler(self, alert: Alert):
        """Webhook处理器"""
        config = self._channel_config.get(AlertChannel.WEBHOOK)
        if not config:
            return

        try:
            import urllib.request

            payload = json.dumps(alert.to_dict()).encode('utf-8')
            req = urllib.request.Request(
                config['url'],
                data=payload,
                headers=config['headers']
            )
            with urllib.request.urlopen(req, timeout=10) as response:
                logger.info(f"Webhook告警已发送: {alert.id}, 响应: {response.status}")

        except Exception as e:
            logger.error(f"发送Webhook告警失败: {e}")

    def _dingtalk_handler(self, alert: Alert):
        """钉钉机器人处理器"""
        config = self._channel_config.get(AlertChannel.DINGTALK)
        if not config:
            return

        try:
            import urllib.request
            import time
            import hmac
            import hashlib
            import base64

            url = config['webhook_url']

            # 如果有签名密钥，添加签名
            if config.get('secret'):
                timestamp = str(round(time.time() * 1000))
                secret = config['secret']
                string_to_sign = f'{timestamp}\n{secret}'
                hmac_code = hmac.new(
                    secret.encode('utf-8'),
                    string_to_sign.encode('utf-8'),
                    digestmod=hashlib.sha256
                ).digest()
                sign = base64.b64encode(hmac_code).decode('utf-8')
                url = f"{url}&timestamp={timestamp}&sign={sign}"

            # 构建消息
            level_color = {
                AlertLevel.INFO: '#1890ff',
                AlertLevel.WARNING: '#faad14',
                AlertLevel.ERROR: '#f5222d',
                AlertLevel.CRITICAL: '#722ed1'
            }

            payload = {
                "msgtype": "markdown",
                "markdown": {
                    "title": alert.title,
                    "text": f"""### {alert.title}

**告警级别**: {alert.level.value}
**告警时间**: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
**告警来源**: {alert.source}
**相关代码**: {alert.code or '无'}

**告警内容**:
{alert.message}
"""
                }
            }

            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode('utf-8'),
                headers={'Content-Type': 'application/json'}
            )
            with urllib.request.urlopen(req, timeout=10) as response:
                result = json.loads(response.read().decode('utf-8'))
                if result.get('errcode') == 0:
                    logger.info(f"钉钉告警已发送: {alert.id}")
                else:
                    logger.error(f"钉钉告警发送失败: {result}")

        except Exception as e:
            logger.error(f"发送钉钉告警失败: {e}")

    def _wechat_handler(self, alert: Alert):
        """企业微信处理器"""
        config = self._channel_config.get(AlertChannel.WECHAT)
        if not config:
            return

        try:
            import urllib.request

            # 获取access_token
            token_url = f"https://qyapi.weixin.qq.com/cgi-bin/gettoken?corpid={config['corp_id']}&corpsecret={config['secret']}"
            with urllib.request.urlopen(token_url, timeout=10) as response:
                result = json.loads(response.read().decode('utf-8'))
                access_token = result.get('access_token')

            if not access_token:
                logger.error("获取企业微信access_token失败")
                return

            # 发送消息
            send_url = f"https://qyapi.weixin.qq.com/cgi-bin/message/send?access_token={access_token}"

            payload = {
                "touser": "|".join(config['to_users']),
                "msgtype": "text",
                "agentid": config['agent_id'],
                "text": {
                    "content": f"[{alert.level.value.upper()}] {alert.title}\n\n{alert.message}\n\n时间: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
                }
            }

            req = urllib.request.Request(
                send_url,
                data=json.dumps(payload).encode('utf-8'),
                headers={'Content-Type': 'application/json'}
            )
            with urllib.request.urlopen(req, timeout=10) as response:
                result = json.loads(response.read().decode('utf-8'))
                if result.get('errcode') == 0:
                    logger.info(f"企业微信告警已发送: {alert.id}")
                else:
                    logger.error(f"企业微信告警发送失败: {result}")

        except Exception as e:
            logger.error(f"发送企业微信告警失败: {e}")

    def get_alerts(
        self,
        level: Optional[AlertLevel] = None,
        limit: int = 100,
        acknowledged: Optional[bool] = None
    ) -> List[Alert]:
        """获取告警列表"""
        alerts = self._alerts.copy()

        if level:
            alerts = [a for a in alerts if a.level == level]
        if acknowledged is not None:
            alerts = [a for a in alerts if a.acknowledged == acknowledged]

        return alerts[-limit:]

    def acknowledge(self, alert_id: str) -> bool:
        """确认告警"""
        for alert in self._alerts:
            if alert.id == alert_id:
                alert.acknowledged = True
                return True
        return False

    def acknowledge_all(self):
        """确认所有告警"""
        for alert in self._alerts:
            alert.acknowledged = True

    def clear_alerts(self):
        """清除所有告警"""
        self._alerts = []
