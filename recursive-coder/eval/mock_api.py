"""Mock API caller that simulates LLM responses for offline evaluation.

Each test level has scripted responses that exercise specific capabilities:
- L1: Direct leaf execution (no decomposition)
- L2: Data-driven leaf with file I/O
- L3: Judge decomposes into 2 subtasks, each executed
- L4: Judge decomposes into 3 subtasks with dependencies
- L5: Judge decomposes, one subtask fails, triggers backtrack + retry
"""

from __future__ import annotations

import json
import time
from typing import Any, Optional

from recursive_coder.api_caller import APICaller


class MockAPICaller(APICaller):
    """Drop-in replacement for APICaller that returns scripted responses."""

    def __init__(self, scenario: str = "L1", **kwargs):
        super().__init__(**kwargs)
        self.scenario = scenario
        self._call_log: list[dict] = []
        self._agent_step_counters: dict[str, int] = {}  # task_id → step count

    async def call(
        self,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
        model_name: Optional[str] = None,
        task_node_id: str = "",
        phase: str = "",
    ) -> dict:
        self.call_count += 1
        t0 = time.monotonic()

        # Log the call
        self._call_log.append({
            "call_num": self.call_count,
            "task_node_id": task_node_id,
            "phase": phase,
            "has_tools": tools is not None,
            "msg_count": len(messages),
        })

        # Route to scenario handler
        handler = getattr(self, f"_handle_{self.scenario}", None)
        if handler is None:
            handler = self._handle_default

        response = handler(messages, tools, task_node_id, phase)

        latency = int((time.monotonic() - t0) * 1000)
        self.latencies.append(latency)
        usage = response.get("usage", {})
        self.total_input_tokens += usage.get("prompt_tokens", 0)
        self.total_output_tokens += usage.get("completion_tokens", 0)

        if self.persistence:
            self.persistence.save_api_call({
                "model": "mock",
                "task_node_id": task_node_id,
                "phase": phase,
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
                "latency_ms": latency,
            })

        return response

    def _make_response(self, content: str = "", tool_calls: list | None = None) -> dict:
        msg: dict[str, Any] = {"role": "assistant"}
        if content:
            msg["content"] = content
        if tool_calls:
            msg["tool_calls"] = tool_calls
        return {
            "choices": [{"message": msg}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50},
        }

    def _make_tool_call(self, tc_id: str, name: str, args: dict) -> dict:
        return {
            "id": tc_id,
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)},
        }

    def _get_step(self, task_id: str) -> int:
        count = self._agent_step_counters.get(task_id, 0) + 1
        self._agent_step_counters[task_id] = count
        return count

    # ── L1: echo_number — no decomposition, just write & run ──

    def _handle_L1(self, messages, tools, task_id, phase):
        if phase == "judge":
            return self._make_response(content="""
分析完了，这是一个简单任务。
<json>
{
  "can_verify": true,
  "verification": {
    "description": "Run solution.py and check output is 42",
    "command": "python solution.py",
    "expected_output": "42",
    "compare_mode": "contains"
  },
  "data_port": {"output_files": ["solution.py"]},
  "implementation_hint": "Just print(42)"
}
</json>
""")
        elif phase == "agent":
            step = self._get_step(task_id)
            if step == 1:
                return self._make_response(
                    content="I'll write solution.py",
                    tool_calls=[
                        self._make_tool_call("tc1", "write_file", {
                            "path": "solution.py",
                            "content": "print(42)\n",
                        }),
                    ],
                )
            elif step == 2:
                return self._make_response(
                    content="Done! Let me verify.",
                    tool_calls=[
                        self._make_tool_call("tc2", "task_done", {
                            "success": True,
                            "summary": "Created solution.py that prints 42",
                        }),
                    ],
                )
            return self._make_response(
                tool_calls=[self._make_tool_call("tc_done", "task_done", {
                    "success": True, "summary": "done",
                })]
            )
        return self._handle_default(messages, tools, task_id, phase)

    # ── L2: sum_from_file — leaf with file I/O ──

    def _handle_L2(self, messages, tools, task_id, phase):
        if phase == "judge":
            return self._make_response(content="""
<json>
{
  "can_verify": true,
  "verification": {
    "description": "Run sum.py to sum numbers from data/numbers.txt",
    "command": "python sum.py",
    "expected_output": "100",
    "compare_mode": "contains"
  },
  "data_port": {
    "input_files": ["data/numbers.txt"],
    "output_files": ["sum.py"]
  },
  "implementation_hint": "Read file, sum lines, print result"
}
</json>
""")
        elif phase == "agent":
            step = self._get_step(task_id)
            if step == 1:
                return self._make_response(
                    content="Reading input file first",
                    tool_calls=[
                        self._make_tool_call("tc1", "read_file", {"path": "data/numbers.txt"}),
                    ],
                )
            elif step == 2:
                return self._make_response(
                    content="Now writing sum.py",
                    tool_calls=[
                        self._make_tool_call("tc2", "write_file", {
                            "path": "sum.py",
                            "content": (
                                "with open('data/numbers.txt') as f:\n"
                                "    total = sum(int(line.strip()) for line in f if line.strip())\n"
                                "print(total)\n"
                            ),
                        }),
                    ],
                )
            elif step == 3:
                return self._make_response(
                    content="Let me test it",
                    tool_calls=[
                        self._make_tool_call("tc3", "shell", {"command": "python sum.py"}),
                    ],
                )
            elif step == 4:
                return self._make_response(
                    tool_calls=[self._make_tool_call("tc4", "task_done", {
                        "success": True, "summary": "sum.py prints 100",
                    })]
                )
            return self._make_response(
                tool_calls=[self._make_tool_call("tc_done", "task_done", {
                    "success": True, "summary": "done",
                })]
            )
        return self._handle_default(messages, tools, task_id, phase)

    # ── L3: word_count — decomposes into 2 subtasks ──

    def _handle_L3(self, messages, tools, task_id, phase):
        if phase == "judge":
            # Check if this is root or a subtask
            user_msg = messages[-1]["content"] if messages else ""
            if "sorted by count descending" in user_msg.lower() or ("top 5 most frequent" in user_msg.lower() and "count_words" not in user_msg.lower()):
                # Root task — decompose
                return self._make_response(content="""
<json>
{
  "can_verify": false,
  "subtasks": [
    {
      "description": "Write a Python script count_words.py that reads data/article.txt, counts word frequencies (case-insensitive, strip punctuation), and saves counts to output/word_counts.json",
      "data_port": {"input_files": ["data/article.txt"], "output_files": ["output/word_counts.json"]},
      "dependencies": []
    },
    {
      "description": "Write a script top_words.py that reads output/word_counts.json, finds top 5 words, writes to output/top_words.txt and prints to stdout",
      "data_port": {"input_files": ["output/word_counts.json"], "output_files": ["output/top_words.txt"]},
      "dependencies": [0]
    }
  ],
  "decomposition_reason": "Separating counting from formatting allows independent testing"
}
</json>
""")
            else:
                # Subtask — can verify directly
                # Check "top_words" first (more specific) before checking "count_words"
                if "top_words" in user_msg.lower() or ("top 5" in user_msg.lower() and "count_words" not in user_msg.lower()):
                    return self._make_response(content="""
<json>
{
  "can_verify": true,
  "verification": {
    "description": "Run top_words.py and check output contains 'the'",
    "command": "python top_words.py",
    "expected_output": "the",
    "compare_mode": "contains"
  },
  "data_port": {"input_files": ["output/word_counts.json"], "output_files": ["output/top_words.txt"]}
}
</json>
""")
                else:
                    return self._make_response(content="""
<json>
{
  "can_verify": true,
  "verification": {
    "description": "Run count_words.py",
    "command": "python count_words.py",
    "expected_output": "",
    "compare_mode": "returncode"
  },
  "data_port": {"input_files": ["data/article.txt"], "output_files": ["output/word_counts.json"]}
}
</json>
""")

        elif phase == "agent":
            step = self._get_step(task_id)
            user_msg = messages[1]["content"] if len(messages) > 1 else ""

            if "top_words" in user_msg.lower() or ("top 5" in user_msg.lower() and "count_words" not in user_msg.lower()):
                # Agent for top_words subtask
                if step == 1:
                    return self._make_response(
                        content="Writing top_words.py",
                        tool_calls=[self._make_tool_call("tc1", "write_file", {
                            "path": "top_words.py",
                            "content": (
                                "import json\n"
                                "with open('output/word_counts.json') as f:\n"
                                "    counts = json.load(f)\n"
                                "top5 = sorted(counts.items(), key=lambda x: -x[1])[:5]\n"
                                "import os; os.makedirs('output', exist_ok=True)\n"
                                "lines = []\n"
                                "for word, count in top5:\n"
                                "    line = f'{word}: {count}'\n"
                                "    lines.append(line)\n"
                                "    print(line)\n"
                                "with open('output/top_words.txt', 'w') as f:\n"
                                "    f.write('\\n'.join(lines) + '\\n')\n"
                            ),
                        })]
                    )
                elif step == 2:
                    return self._make_response(
                        tool_calls=[self._make_tool_call("tc2", "task_done", {
                            "success": True, "summary": "top_words.py written",
                        })]
                    )
            elif "count_words" in user_msg.lower() or "word_counts" in user_msg.lower():
                # Agent for count_words subtask
                if step == 1:
                    return self._make_response(
                        content="Writing count_words.py",
                        tool_calls=[self._make_tool_call("tc1", "write_file", {
                            "path": "count_words.py",
                            "content": (
                                "import json, re\n"
                                "from collections import Counter\n"
                                "with open('data/article.txt') as f:\n"
                                "    text = f.read().lower()\n"
                                "words = re.findall(r'[a-z]+', text)\n"
                                "counts = dict(Counter(words))\n"
                                "import os; os.makedirs('output', exist_ok=True)\n"
                                "with open('output/word_counts.json', 'w') as f:\n"
                                "    json.dump(counts, f, indent=2)\n"
                                "print('Counted', len(counts), 'unique words')\n"
                            ),
                        })]
                    )
                elif step == 2:
                    return self._make_response(
                        tool_calls=[self._make_tool_call("tc2", "task_done", {
                            "success": True, "summary": "count_words.py written",
                        })]
                    )

            return self._make_response(
                tool_calls=[self._make_tool_call("tc_done", "task_done", {
                    "success": True, "summary": "done",
                })]
            )

        # integration phase
        return self._handle_default(messages, tools, task_id, phase)

    # ── L4: csv_pipeline — 3 subtasks with deps ──

    def _handle_L4(self, messages, tools, task_id, phase):
        if phase == "judge":
            user_msg = messages[-1]["content"] if messages else ""
            if "CSV" in user_msg and ("Parse" in user_msg or "Process" in user_msg):
                # Root task — decompose into 3
                return self._make_response(content="""
<json>
{
  "can_verify": false,
  "subtasks": [
    {
      "description": "Write parse_csv.py: parse data/sales.csv into structured data, filter region=='North', save to output/north_sales.csv",
      "data_port": {"input_files": ["data/sales.csv"], "output_files": ["output/north_sales.csv"]},
      "dependencies": []
    },
    {
      "description": "Write aggregate.py: read output/north_sales.csv, sum the amount column, write total to output/north_total.txt and print it",
      "data_port": {"input_files": ["output/north_sales.csv"], "output_files": ["output/north_total.txt"]},
      "dependencies": [0]
    }
  ],
  "decomposition_reason": "Separate filtering from aggregation for independent testing"
}
</json>
""")
            else:
                # Subtask — can verify
                if "parse" in user_msg.lower() or "filter" in user_msg.lower():
                    return self._make_response(content="""
<json>
{
  "can_verify": true,
  "verification": {"description": "Run parse_csv.py", "command": "python parse_csv.py", "expected_output": "", "compare_mode": "returncode"},
  "data_port": {"input_files": ["data/sales.csv"], "output_files": ["output/north_sales.csv"]}
}
</json>
""")
                else:
                    return self._make_response(content="""
<json>
{
  "can_verify": true,
  "verification": {"description": "Run aggregate.py", "command": "python aggregate.py", "expected_output": "350", "compare_mode": "contains"},
  "data_port": {"input_files": ["output/north_sales.csv"], "output_files": ["output/north_total.txt"]}
}
</json>
""")

        elif phase == "agent":
            step = self._get_step(task_id)
            user_msg = messages[1]["content"] if len(messages) > 1 else ""

            if "parse" in user_msg.lower() or "filter" in user_msg.lower():
                if step == 1:
                    return self._make_response(
                        tool_calls=[self._make_tool_call("tc1", "write_file", {
                            "path": "parse_csv.py",
                            "content": (
                                "import csv, os\n"
                                "os.makedirs('output', exist_ok=True)\n"
                                "with open('data/sales.csv') as fin, open('output/north_sales.csv', 'w', newline='') as fout:\n"
                                "    reader = csv.DictReader(fin)\n"
                                "    writer = csv.DictWriter(fout, fieldnames=['product', 'region', 'amount'])\n"
                                "    writer.writeheader()\n"
                                "    for row in reader:\n"
                                "        if row['region'].strip() == 'North':\n"
                                "            writer.writerow(row)\n"
                                "print('Filtered North sales')\n"
                            ),
                        })]
                    )
                elif step == 2:
                    return self._make_response(
                        tool_calls=[self._make_tool_call("tc2", "task_done", {
                            "success": True, "summary": "parse_csv.py written",
                        })]
                    )
            else:
                if step == 1:
                    return self._make_response(
                        tool_calls=[self._make_tool_call("tc1", "write_file", {
                            "path": "aggregate.py",
                            "content": (
                                "import csv, os\n"
                                "os.makedirs('output', exist_ok=True)\n"
                                "total = 0\n"
                                "with open('output/north_sales.csv') as f:\n"
                                "    reader = csv.DictReader(f)\n"
                                "    for row in reader:\n"
                                "        total += int(row['amount'])\n"
                                "print(total)\n"
                                "with open('output/north_total.txt', 'w') as f:\n"
                                "    f.write(str(total) + '\\n')\n"
                            ),
                        })]
                    )
                elif step == 2:
                    return self._make_response(
                        tool_calls=[self._make_tool_call("tc2", "task_done", {
                            "success": True, "summary": "aggregate.py written",
                        })]
                    )

            return self._make_response(
                tool_calls=[self._make_tool_call("tc_done", "task_done", {
                    "success": True, "summary": "done",
                })]
            )

        return self._handle_default(messages, tools, task_id, phase)

    # ── L5: expression_eval — decompose, one fails, backtrack ──

    def _handle_L5(self, messages, tools, task_id, phase):
        if phase == "judge":
            user_msg = messages[-1]["content"] if messages else ""
            # Root task has "Build a Python program" — subtasks don't
            is_root = "build a python program" in user_msg.lower() or "do not use eval" in user_msg.lower()
            if is_root:
                return self._make_response(content="""
<json>
{
  "can_verify": false,
  "subtasks": [
    {
      "description": "Write tokenizer.py: tokenize arithmetic expressions from data/expressions.txt into tokens list, save to output/tokens.json",
      "data_port": {"input_files": ["data/expressions.txt"], "output_files": ["output/tokens.json"]},
      "dependencies": []
    },
    {
      "description": "Write eval_expr.py: read output/tokens.json, parse and evaluate each expression, write results to output/results.txt and print each result",
      "data_port": {"input_files": ["output/tokens.json"], "output_files": ["output/results.txt"]},
      "dependencies": [0]
    }
  ],
  "decomposition_reason": "Separate tokenization from evaluation for independent testing"
}
</json>
""")
            else:
                # Check eval_expr FIRST — its description also contains "token"
                if "eval_expr" in user_msg.lower() or "parse and evaluate" in user_msg.lower():
                    return self._make_response(content="""
<json>
{
  "can_verify": true,
  "verification": {"description": "Run eval_expr.py", "command": "python eval_expr.py", "expected_output": "5", "compare_mode": "contains"},
  "data_port": {"input_files": ["output/tokens.json"], "output_files": ["output/results.txt"]}
}
</json>
""")
                else:
                    return self._make_response(content="""
<json>
{
  "can_verify": true,
  "verification": {"description": "Run tokenizer.py", "command": "python tokenizer.py", "expected_output": "", "compare_mode": "returncode"},
  "data_port": {"input_files": ["data/expressions.txt"], "output_files": ["output/tokens.json"]}
}
</json>
""")

        elif phase == "agent":
            step = self._get_step(task_id)
            user_msg = messages[1]["content"] if len(messages) > 1 else ""

            # Check for eval_expr subtask FIRST (more specific) — it also contains "token"
            if "eval_expr" in user_msg.lower() or ("parse and evaluate" in user_msg.lower()):
                if step == 1:
                    return self._make_response(
                        tool_calls=[self._make_tool_call("tc1", "write_file", {
                            "path": "eval_expr.py",
                            "content": (
                                "import json, os\n"
                                "os.makedirs('output', exist_ok=True)\n\n"
                                "class Parser:\n"
                                "    def __init__(self, tokens):\n"
                                "        self.tokens = tokens\n"
                                "        self.pos = 0\n"
                                "    def peek(self):\n"
                                "        return self.tokens[self.pos] if self.pos < len(self.tokens) else None\n"
                                "    def consume(self):\n"
                                "        t = self.tokens[self.pos]\n"
                                "        self.pos += 1\n"
                                "        return t\n"
                                "    def parse_expr(self):\n"
                                "        result = self.parse_term()\n"
                                "        while self.peek() in ('+', '-'):\n"
                                "            op = self.consume()\n"
                                "            right = self.parse_term()\n"
                                "            result = result + right if op == '+' else result - right\n"
                                "        return result\n"
                                "    def parse_term(self):\n"
                                "        result = self.parse_factor()\n"
                                "        while self.peek() in ('*', '//', '/'):\n"
                                "            op = self.consume()\n"
                                "            right = self.parse_factor()\n"
                                "            if op == '*': result = result * right\n"
                                "            elif op == '//': result = result // right\n"
                                "            else: result = result // right\n"
                                "        return result\n"
                                "    def parse_factor(self):\n"
                                "        if self.peek() == '(':\n"
                                "            self.consume()\n"
                                "            result = self.parse_expr()\n"
                                "            self.consume()  # ')'\n"
                                "            return result\n"
                                "        return int(self.consume())\n\n"
                                "with open('output/tokens.json') as f:\n"
                                "    data = json.load(f)\n"
                                "results = []\n"
                                "for item in data:\n"
                                "    p = Parser(item['tokens'])\n"
                                "    val = p.parse_expr()\n"
                                "    results.append(str(val))\n"
                                "    print(val)\n"
                                "with open('output/results.txt', 'w') as f:\n"
                                "    f.write('\\n'.join(results) + '\\n')\n"
                            ),
                        })]
                    )
                elif step == 2:
                    return self._make_response(
                        tool_calls=[self._make_tool_call("tc2", "task_done", {
                            "success": True, "summary": "eval_expr.py written",
                        })]
                    )
            elif "tokenize" in user_msg.lower() or "tokenizer" in user_msg.lower():
                if step == 1:
                    return self._make_response(
                        tool_calls=[self._make_tool_call("tc1", "write_file", {
                            "path": "tokenizer.py",
                            "content": (
                                "import json, re, os\n"
                                "os.makedirs('output', exist_ok=True)\n"
                                "all_tokens = []\n"
                                "with open('data/expressions.txt') as f:\n"
                                "    for line in f:\n"
                                "        line = line.strip()\n"
                                "        if not line: continue\n"
                                "        toks = re.findall(r'\\d+|//|[+\\-*/()/]', line)\n"
                                "        all_tokens.append({'expr': line, 'tokens': toks})\n"
                                "with open('output/tokens.json', 'w') as f:\n"
                                "    json.dump(all_tokens, f, indent=2)\n"
                                "print(f'Tokenized {len(all_tokens)} expressions')\n"
                            ),
                        })]
                    )
                elif step == 2:
                    return self._make_response(
                        tool_calls=[self._make_tool_call("tc2", "task_done", {
                            "success": True, "summary": "tokenizer.py written",
                        })]
                    )

            return self._make_response(
                tool_calls=[self._make_tool_call("tc_done", "task_done", {
                    "success": True, "summary": "done",
                })]
            )

        elif phase == "backtrack":
            return self._make_response(content="""
<json>
{
  "subtasks": [
    {"description": "Rewrite combined eval_expr.py that tokenizes and evaluates in one script", "dependencies": []},
    {"description": "Test with all expressions", "dependencies": [0]}
  ],
  "decomposition_reason": "Combining steps to avoid file format issues"
}
</json>
""")

        return self._handle_default(messages, tools, task_id, phase)

    # ── Default fallback (integration verify, etc.) ──

    def _handle_default(self, messages, tools, task_id, phase):
        if tools:
            # Integration verify agent — just succeed
            step = self._get_step(task_id)
            if step <= 1:
                return self._make_response(
                    content="Integration looks good.",
                    tool_calls=[self._make_tool_call("tc_done", "task_done", {
                        "success": True,
                        "summary": "Integration verified successfully",
                    })]
                )
            return self._make_response(
                tool_calls=[self._make_tool_call("tc_fallback", "task_done", {
                    "success": True, "summary": "done",
                })]
            )
        return self._make_response(content="OK")
