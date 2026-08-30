#!/usr/bin/env python3
"""情绪哨兵 — 宏观政策 + 个股事件过滤

数据源:
  - ak.news_cctv(date) → 新闻联播
  - ak.stock_info_global_cls() → 财联社电报
  - ak.stock_news_em(symbol) → 东方财富个股新闻

分析:
  - Claude CLI 宏观情绪分析 (输入: 新闻 + 持仓)
  - Claude CLI 个股风险筛查 (输入: 候选股新闻)

用法:
    python news_sentinel.py --test-macro
    python news_sentinel.py --test-stocks SH600036 SH601318
"""
import json
import re
import subprocess
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from cli_paths import CLAUDE_BIN

PROJECT_DIR = Path(__file__).resolve().parent
CACHE_DIR = PROJECT_DIR / "experiment" / "news_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


class NewsSentinel:
    """新闻情绪分析 + 信号后处理"""

    def fetch_macro_news(self, target_date: str = None) -> dict:
        """获取宏观新闻: CCTV新闻联播 + 财联社电报

        Returns:
            {"cctv": [...], "cls": [...], "date": "2026-02-12"}
        """
        if target_date is None:
            target_date = date.today().isoformat()

        cache_file = CACHE_DIR / f"{target_date}_macro.json"
        if cache_file.exists():
            return json.loads(cache_file.read_text(encoding='utf-8'))

        import akshare as ak

        result = {"date": target_date, "cctv": [], "cls": []}

        # 1. CCTV 新闻联播
        try:
            df = ak.news_cctv(date=target_date.replace('-', ''))
            for _, row in df.iterrows():
                result["cctv"].append({
                    "title": str(row.get("title", "")),
                    "content": str(row.get("content", ""))[:300],
                })
            print(f"  [sentinel] CCTV 新闻: {len(result['cctv'])} 条")
        except Exception as e:
            print(f"  [sentinel] CCTV 获取失败: {e}")

        # 2. 财联社电报 (最新)
        try:
            df = ak.stock_info_global_cls()
            today_str = target_date
            for _, row in df.head(50).iterrows():
                title = str(row.get("标题", row.get("title", "")))
                content = str(row.get("内容", row.get("content", "")))[:200]
                result["cls"].append({
                    "title": title,
                    "content": content,
                })
            print(f"  [sentinel] 财联社电报: {len(result['cls'])} 条")
        except Exception as e:
            print(f"  [sentinel] 财联社获取失败: {e}")

        cache_file.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        return result

    def fetch_stock_news(self, stock_codes: list,
                         target_date: str = None) -> dict:
        """获取个股新闻 (东方财富)

        Args:
            stock_codes: Qlib 格式 ['SH600036', 'SZ000001']
        Returns:
            {"SH600036": [{"title": ..., "content": ...}], ...}
        """
        if target_date is None:
            target_date = date.today().isoformat()

        cache_file = CACHE_DIR / f"{target_date}_stocks.json"
        if cache_file.exists():
            return json.loads(cache_file.read_text(encoding='utf-8'))

        import akshare as ak

        result = {}
        cutoff = datetime.now() - timedelta(days=7)

        for code in stock_codes:
            # SH600036 → 600036
            symbol = code[2:]
            try:
                df = ak.stock_news_em(symbol=symbol)
                news_list = []
                for _, row in df.iterrows():
                    title = str(row.get("新闻标题", row.get("title", "")))
                    content = str(row.get("新闻内容", row.get("content", "")))[:200]
                    pub_time = str(row.get("发布时间", row.get("publish_time", "")))

                    # 只取 7 天内
                    try:
                        pub_dt = datetime.strptime(pub_time[:19], "%Y-%m-%d %H:%M:%S")
                        if pub_dt < cutoff:
                            continue
                    except (ValueError, TypeError):
                        pass

                    news_list.append({
                        "title": title,
                        "content": content,
                        "time": pub_time,
                    })
                    if len(news_list) >= 10:
                        break

                result[code] = news_list
                print(f"  [sentinel] {code}: {len(news_list)} 条新闻")
                time.sleep(0.5)
            except Exception as e:
                print(f"  [sentinel] {code} 新闻获取失败: {e}")
                result[code] = []

        cache_file.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        return result

    def analyze_macro(self, macro_news: dict, holdings: list) -> dict:
        """Claude 分析宏观情绪对持仓的影响

        Args:
            macro_news: fetch_macro_news() 结果
            holdings: [{"code": "SH600036", "name": "招商银行"}, ...]
        Returns:
            {"sentiment": "neutral", "summary": "...",
             "holdings_impact": [...], "topk_adjustment": 0}
        """
        cache_file = CACHE_DIR / f"{macro_news['date']}_macro_analysis.json"
        if cache_file.exists():
            return json.loads(cache_file.read_text(encoding='utf-8'))

        # 构建 prompt
        cctv_text = "\n".join(
            f"- {n['title']}" for n in macro_news.get("cctv", [])[:20])
        cls_text = "\n".join(
            f"- {n['title']}: {n['content'][:100]}"
            for n in macro_news.get("cls", [])[:30])
        holdings_text = "\n".join(
            f"- {h['code']} {h['name']}" for h in holdings)

        prompt = f"""你是一个宏观政策分析师。请分析今日新闻对 A 股市场和持仓的影响。

## 今日新闻联播
{cctv_text if cctv_text else '(无数据)'}

## 财联社电报 (最新)
{cls_text if cls_text else '(无数据)'}

## 当前持仓
{holdings_text if holdings_text else '(空仓)'}

## 输出要求
严格输出以下 JSON 格式（不要输出其他内容）：
```json
{{
    "sentiment": "bearish/neutral/bullish",
    "summary": "1-2句概述今日宏观环境",
    "holdings_impact": [
        {{"code": "SH600036", "name": "招商银行", "impact": "positive/neutral/negative", "reason": "影响原因"}}
    ],
    "topk_adjustment": 0
}}
```

规则:
- sentiment: bearish 表示利空(建议减仓), neutral 表示中性, bullish 表示利好(可加仓)
- topk_adjustment: bearish → -4, neutral → 0, bullish → +2
- holdings_impact: 只列出有明确影响的持仓，无影响的不列出
- 判断要保守，无明确信号就给 neutral"""

        result = self._call_claude(prompt)

        if result:
            cache_file.write_text(
                json.dumps(result, ensure_ascii=False, indent=2),
                encoding='utf-8')
        return result

    def screen_stock_risks(self, stock_news: dict,
                           stock_names: dict = None) -> dict:
        """Claude 筛查个股风险

        Args:
            stock_news: fetch_stock_news() 结果
            stock_names: {code: name} 映射
        Returns:
            {"flagged": [{"code": ..., "risk_level": "high", "reason": ...}],
             "safe": ["SH600036", ...]}
        """
        if not stock_news:
            return {"flagged": [], "safe": []}

        # 构建 prompt
        news_text = ""
        for code, news_list in stock_news.items():
            name = stock_names.get(code, code) if stock_names else code
            if not news_list:
                news_text += f"\n### {code} ({name})\n(无近期新闻)\n"
                continue
            news_text += f"\n### {code} ({name})\n"
            for n in news_list[:5]:
                news_text += f"- [{n.get('time', '')}] {n['title']}\n"

        prompt = f"""你是一个个股风险筛查分析师。请检查以下候选股近期新闻，识别重大负面风险。

## 候选股新闻
{news_text}

## 输出要求
严格输出以下 JSON 格式（不要输出其他内容）：
```json
{{
    "flagged": [
        {{"code": "SH601398", "risk_level": "high", "reason": "证监会立案调查"}}
    ],
    "safe": ["SH600036", "SH601318"]
}}
```

规则:
- 只标记 high risk: 被调查/ST风险/重大诉讼/制裁/退市风险/违规处罚/业绩暴雷/重大减持
- 一般性利空(如短期业绩不及预期)不算 high risk
- 判断要保守，宁可漏报不可误报
- safe 列出所有未被标记的股票代码"""

        result = self._call_claude(prompt)
        if not result:
            # Claude 失败时返回全部 safe
            return {"flagged": [], "safe": list(stock_news.keys())}
        return result

    def apply_sentiment_filter(self, base_signal: dict, macro: dict,
                               risks: dict) -> dict:
        """将情绪分析结果应用到基础信号上

        Args:
            base_signal: SignalGenerator.get_signal() 的输出
            macro: analyze_macro() 的输出
            risks: screen_stock_risks() 的输出
        Returns:
            修改后的信号 dict (新增 filtered/replacement/macro_sentiment 字段)
        """
        import copy
        result = copy.deepcopy(base_signal)

        # 1. 移除 high risk 股票
        flagged_codes = set(
            f['code'] for f in risks.get('flagged', [])
            if f.get('risk_level') == 'high')
        filtered_stocks = []
        for code in list(result.get('target_stocks', [])):
            if code in flagged_codes:
                filtered_stocks.append(code)
                result['target_stocks'].remove(code)

        # 2. TopK 调整
        topk_adj = macro.get('topk_adjustment', 0)
        original_topk = result.get('effective_topk', 16)
        new_topk = max(8, min(24, original_topk + topk_adj))
        result['effective_topk'] = new_topk

        # 3. 递补填满 (从 scores 中取下一名)
        scores = base_signal.get('scores', {})
        all_sorted = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        current_set = set(result['target_stocks'])
        replacements = []

        target_count = new_topk
        for code, score in all_sorted:
            if len(result['target_stocks']) >= target_count:
                break
            if code not in current_set and code not in flagged_codes:
                result['target_stocks'].append(code)
                result['scores'][code] = score
                current_set.add(code)
                replacements.append(code)

        # 截断到 new_topk
        result['target_stocks'] = result['target_stocks'][:new_topk]

        # 4. 附加元数据
        result['macro_sentiment'] = macro.get('sentiment', 'neutral')
        result['macro_summary'] = macro.get('summary', '')
        result['holdings_impact'] = macro.get('holdings_impact', [])
        result['filtered_stocks'] = filtered_stocks
        result['replacement_stocks'] = replacements
        result['topk_adjustment'] = topk_adj

        return result

    def _call_claude(self, prompt: str) -> dict:
        """调用 Claude CLI 并解析 JSON 输出"""
        try:
            cmd = [
                CLAUDE_BIN,
                '--print',
                '--dangerously-skip-permissions',
                '--output-format', 'text',
                '-p', prompt
            ]
            result = subprocess.run(
                cmd,
                capture_output=True, text=True,
                timeout=120,
                cwd=str(PROJECT_DIR)
            )
            output = result.stdout.strip()

            # 提取 JSON
            json_match = re.search(r'\{[\s\S]*\}', output)
            if json_match:
                return json.loads(json_match.group())
            else:
                print(f"  [sentinel] Claude 输出无法解析 JSON: {output[:200]}")
                return {}
        except subprocess.TimeoutExpired:
            print("  [sentinel] Claude 调用超时 (120s)")
            return {}
        except Exception as e:
            print(f"  [sentinel] Claude 调用失败: {e}")
            return {}


# ============ CLI ============

def _test_macro():
    """测试宏观新闻获取 + 分析"""
    sentinel = NewsSentinel()

    print("=== 获取宏观新闻 ===")
    macro = sentinel.fetch_macro_news()
    print(f"CCTV: {len(macro.get('cctv', []))} 条")
    print(f"财联社: {len(macro.get('cls', []))} 条")

    if macro.get('cctv'):
        print("\n--- CCTV 标题 ---")
        for n in macro['cctv'][:5]:
            print(f"  {n['title']}")

    # 模拟持仓
    holdings = [
        {"code": "SH600036", "name": "招商银行"},
        {"code": "SH601318", "name": "中国平安"},
    ]
    print("\n=== Claude 宏观分析 ===")
    analysis = sentinel.analyze_macro(macro, holdings)
    print(json.dumps(analysis, ensure_ascii=False, indent=2))


def _test_stocks(codes: list):
    """测试个股新闻获取 + 风险筛查"""
    sentinel = NewsSentinel()

    print(f"=== 获取个股新闻: {codes} ===")
    news = sentinel.fetch_stock_news(codes)
    for code, news_list in news.items():
        print(f"\n{code}: {len(news_list)} 条")
        for n in news_list[:3]:
            print(f"  - {n['title']}")

    print("\n=== Claude 风险筛查 ===")
    risks = sentinel.screen_stock_risks(news)
    print(json.dumps(risks, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='情绪哨兵')
    parser.add_argument('--test-macro', action='store_true',
                        help='测试宏观新闻获取+分析')
    parser.add_argument('--test-stocks', nargs='+', metavar='CODE',
                        help='测试个股新闻 (如 SH600036)')
    args = parser.parse_args()

    if args.test_macro:
        _test_macro()
    elif args.test_stocks:
        _test_stocks(args.test_stocks)
    else:
        parser.print_help()
