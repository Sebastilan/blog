"""Iterative optimizer: analyze eval reports → suggest prompt/config changes."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import yaml

from .api_caller import APICaller
from .logger_setup import get_logger

logger = get_logger("optimizer")

OPTIMIZER_PROMPT = """\
你是一个 AI 编码框架的优化专家。你的任务是分析一次运行的评估报告，找出薄弱环节，并给出具体的改进建议。

## 当前评估报告
```json
{report_json}
```

## 当前 prompt 模板
{prompt_templates_section}

## 当前配置
```yaml
{config_yaml}
```

{previous_section}

请分析以上数据，输出优化建议。将结果包裹在 <json> 标签中：

<json>
{{
  "analysis": {{
    "main_issues": ["问题1", "问题2", ...],
    "root_causes": ["原因1", "原因2", ...]
  }},
  "prompt_changes": [
    {{
      "file": "模板文件名，如 execute.txt",
      "change_type": "replace | append | prepend",
      "old": "要替换的内容（仅 replace 时需要）",
      "new": "新内容",
      "reason": "修改原因"
    }}
  ],
  "config_changes": [
    {{
      "key": "配置项名称",
      "old_value": "原值",
      "new_value": "新值",
      "reason": "修改原因"
    }}
  ],
  "expected_improvements": {{
    "指标名": "预期变化，如 55% → 70%"
  }}
}}
</json>
"""


class Optimizer:
    """Analyze evaluation reports and suggest improvements."""

    def __init__(
        self,
        api_caller: APICaller,
        project_dir: str,
        model_name: Optional[str] = None,
    ) -> None:
        self.api = api_caller
        self.project_dir = Path(project_dir)
        self.template_dir = self.project_dir / "prompt_templates"
        self.history_dir = self.project_dir / "optimization_history"
        self.history_dir.mkdir(exist_ok=True)
        self.model_name = model_name

    async def analyze(
        self,
        report: dict,
        config: dict,
        previous_iteration: Optional[dict] = None,
    ) -> dict:
        """Call LLM to analyze the report and produce optimization suggestions."""
        # Load current prompt templates
        templates_section = ""
        if self.template_dir.exists():
            for f in sorted(self.template_dir.glob("*.txt")):
                content = f.read_text(encoding="utf-8")
                templates_section += f"\n### {f.name}\n```\n{content}\n```\n"

        # Load current config
        config_path = self.project_dir / "config.yaml"
        config_yaml = config_path.read_text(encoding="utf-8") if config_path.exists() else yaml.dump(config)

        # Previous iteration context
        previous_section = ""
        if previous_iteration:
            previous_section = (
                "## 上一轮优化记录\n```json\n"
                + json.dumps(previous_iteration, ensure_ascii=False, indent=2)
                + "\n```\n"
            )

        prompt = OPTIMIZER_PROMPT.format(
            report_json=json.dumps(report, ensure_ascii=False, indent=2),
            prompt_templates_section=templates_section,
            config_yaml=config_yaml,
            previous_section=previous_section,
        )

        messages = [
            {"role": "system", "content": "你是 AI 编码框架优化专家。"},
            {"role": "user", "content": prompt},
        ]

        resp = await self.api.call(
            messages=messages,
            model_name=self.model_name,
            phase="optimize",
        )

        text = resp["choices"][0]["message"].get("content", "")

        # Parse suggestion JSON
        import re
        m = re.search(r"<json>\s*(.*?)\s*</json>", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass

        logger.warning("Failed to parse optimizer response, returning raw text")
        return {"raw_response": text}

    def apply_suggestions(self, suggestions: dict) -> list[str]:
        """Apply prompt and config changes. Returns list of applied changes."""
        applied = []

        # Apply prompt changes
        for change in suggestions.get("prompt_changes", []):
            fname = change.get("file", "")
            path = self.template_dir / fname
            if not path.exists():
                logger.warning("Template not found: %s", fname)
                continue

            content = path.read_text(encoding="utf-8")
            change_type = change.get("change_type", "append")

            if change_type == "replace" and change.get("old"):
                if change["old"] in content:
                    content = content.replace(change["old"], change.get("new", ""))
                    path.write_text(content, encoding="utf-8")
                    applied.append(f"prompt/{fname}: replaced")
                else:
                    logger.warning("Old text not found in %s", fname)
            elif change_type == "append":
                content += "\n" + change.get("new", "")
                path.write_text(content, encoding="utf-8")
                applied.append(f"prompt/{fname}: appended")
            elif change_type == "prepend":
                content = change.get("new", "") + "\n" + content
                path.write_text(content, encoding="utf-8")
                applied.append(f"prompt/{fname}: prepended")

        # Apply config changes
        config_path = self.project_dir / "config.yaml"
        if config_path.exists() and suggestions.get("config_changes"):
            config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
            for change in suggestions["config_changes"]:
                key = change.get("key", "")
                new_val = change.get("new_value")
                if key and new_val is not None:
                    # Try to convert type
                    try:
                        if isinstance(config.get(key), int):
                            new_val = int(new_val)
                        elif isinstance(config.get(key), float):
                            new_val = float(new_val)
                        elif isinstance(config.get(key), bool):
                            new_val = str(new_val).lower() in ("true", "1", "yes")
                    except (ValueError, TypeError):
                        pass
                    config[key] = new_val
                    applied.append(f"config/{key}: {change.get('old_value')} → {new_val}")
            config_path.write_text(yaml.dump(config, default_flow_style=False), encoding="utf-8")

        return applied

    def save_iteration(
        self,
        before_report: dict,
        after_report: Optional[dict],
        suggestions: dict,
        applied: list[str],
    ) -> None:
        """Save a complete iteration record."""
        existing = list(self.history_dir.glob("iteration_*.json"))
        num = len(existing) + 1

        def _key_metrics(r: dict) -> dict:
            return {
                "run_id": r.get("run_id", ""),
                "first_pass_rate": r.get("quality", {}).get("first_pass_rate", 0),
                "final_pass_rate": r.get("quality", {}).get("final_pass_rate", 0),
                "parse_failures": r.get("process", {}).get("parse_failures", 0),
                "total_cost_usd": r.get("efficiency", {}).get("estimated_cost_usd", 0),
                "avg_agent_steps": r.get("efficiency", {}).get("avg_agent_steps", 0),
            }

        record = {
            "iteration": num,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "before": _key_metrics(before_report),
            "changes_applied": applied,
            "suggestions": suggestions,
        }

        if after_report:
            bm = _key_metrics(before_report)
            am = _key_metrics(after_report)
            record["after"] = am
            record["improvement"] = {
                k: round(am[k] - bm[k], 4) if isinstance(am[k], (int, float)) else "?"
                for k in bm if k != "run_id"
            }
            record["verdict"] = (
                "improved" if am.get("final_pass_rate", 0) >= bm.get("final_pass_rate", 0)
                else "regressed"
            )

        path = self.history_dir / f"iteration_{num:03d}.json"
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("Saved iteration %d to %s", num, path)

    def load_history(self) -> list[dict]:
        records = []
        for f in sorted(self.history_dir.glob("iteration_*.json")):
            records.append(json.loads(f.read_text(encoding="utf-8")))
        return records

    def print_comparison(self, before: dict, after: dict) -> str:
        lines = [
            "",
            "  Iteration Comparison",
            "  " + "-" * 40,
        ]
        keys = ["first_pass_rate", "final_pass_rate", "parse_failures", "total_cost_usd", "avg_agent_steps"]
        b = before.get("quality", {}); b.update(before.get("efficiency", {})); b.update(before.get("process", {}))
        a = after.get("quality", {}); a.update(after.get("efficiency", {})); a.update(after.get("process", {}))
        for k in keys:
            bv = b.get(k, "?")
            av = a.get(k, "?")
            arrow = "↑" if isinstance(av, (int, float)) and isinstance(bv, (int, float)) and av > bv else "↓" if isinstance(av, (int, float)) and isinstance(bv, (int, float)) and av < bv else "="
            lines.append(f"  {k:25s}  {bv} → {av}  {arrow}")
        lines.append("")
        return "\n".join(lines)
