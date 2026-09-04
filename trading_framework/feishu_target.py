"""主动推送的收件人解析

2026-09-04: 发现所有定时任务的飞书推送从未生效过。5 个推送点都读
FEISHU_USER_OPEN_ID，而 .env 里这一项是注释掉的(模板里它就带 # 前缀)。
push_feishu 在凭证缺失时只 log.error 然后 return —— 日志里有记录，但没人看，
对外表现是"任务跑完了、什么都没发生"。

smart_bot 不受影响，因为它是回复收到的消息，收件人来自消息本身。

这里让收件人可以回落到白名单 FEISHU_ALLOWED_OPEN_IDS 的第一个 ——
能给机器人发消息的人，本来就是该收到推送的人。
"""
import os


def resolve_open_id() -> str:
    uid = (os.environ.get("FEISHU_USER_OPEN_ID") or "").strip()
    if uid:
        return uid
    allowed = [x.strip() for x in
               os.environ.get("FEISHU_ALLOWED_OPEN_IDS", "").split(",") if x.strip()]
    return allowed[0] if allowed else ""
