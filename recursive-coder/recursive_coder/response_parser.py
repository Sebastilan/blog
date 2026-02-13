"""Parse structured JSON from model responses (judge & backtrack phases)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from .logger_setup import get_logger

logger = get_logger("parser")


@dataclass
class JudgeResult:
    """Parsed result of a judge-phase API call."""

    can_verify: bool = False

    # if can_verify == True (leaf task planning)
    verification_description: str = ""
    expected_output: str = ""
    verification_command: str = ""
    compare_mode: str = "contains"
    data_port: dict = field(default_factory=dict)
    implementation_hint: str = ""
    execution_plan: list[str] = field(default_factory=list)   # step-by-step plan
    interface: dict = field(default_factory=dict)              # input/output contract

    # if can_verify == False (decomposition planning)
    subtasks: list[dict] = field(default_factory=list)
    decomposition_reason: str = ""
    interface_contract: str = ""  # shared contract between sibling tasks

    # parsing meta
    raw_response: str = ""
    parse_success: bool = False


@dataclass
class BacktrackResult:
    subtasks: list[dict] = field(default_factory=list)
    decomposition_reason: str = ""
    interface_contract: str = ""  # updated contract for re-decomposition
    changes_from_previous: str = ""
    raw_response: str = ""
    parse_success: bool = False


def _extract_json_block(text: str) -> Optional[dict]:
    """Try to extract a JSON object from <json>...</json> tags, then fenced blocks."""
    # Strategy 1: <json>...</json> tags
    m = re.search(r"<json>\s*(.*?)\s*</json>", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # Strategy 2: ```json ... ``` fenced block
    m = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # Strategy 3: try each '{' as a potential JSON start
    for i, ch in enumerate(text):
        if ch == "{":
            depth = 1
            for j in range(i + 1, len(text)):
                if text[j] == "{":
                    depth += 1
                elif text[j] == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(text[i : j + 1])
                        except json.JSONDecodeError:
                            break  # try next '{'

    return None


def parse_judge_response(text: str) -> JudgeResult:
    result = JudgeResult(raw_response=text)
    data = _extract_json_block(text)
    if data is None:
        logger.warning("Failed to extract JSON from judge response")
        return result

    result.parse_success = True
    result.can_verify = data.get("can_verify", False)

    if result.can_verify:
        v = data.get("verification", {})
        result.verification_description = v.get("description", "")
        result.expected_output = v.get("expected_output", "")
        result.verification_command = v.get("command", "")
        result.compare_mode = v.get("compare_mode", "contains")
        result.data_port = data.get("data_port", {})
        result.implementation_hint = data.get("implementation_hint", "")
        result.execution_plan = data.get("execution_plan", [])
        result.interface = data.get("interface", {})
    else:
        result.subtasks = data.get("subtasks", [])
        result.decomposition_reason = data.get("decomposition_reason", "")
        result.interface_contract = data.get("interface_contract", "")

    return result


def parse_backtrack_response(text: str) -> BacktrackResult:
    result = BacktrackResult(raw_response=text)
    data = _extract_json_block(text)
    if data is None:
        logger.warning("Failed to extract JSON from backtrack response")
        return result

    result.parse_success = True
    result.subtasks = data.get("subtasks", [])
    result.decomposition_reason = data.get("decomposition_reason", "")
    result.interface_contract = data.get("interface_contract", "")
    result.changes_from_previous = data.get("changes_from_previous", "")
    return result
