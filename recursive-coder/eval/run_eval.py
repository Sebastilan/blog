"""
End-to-end evaluation suite: 5 levels from simple to complex.

Each test case targets specific capabilities of the recursive-coder framework:

L1: echo_number    — Agent writes & runs trivial code, no decomposition
                     Tests: agent loop, tool use, exact verification
L2: sum_from_file  — Read data file, compute result, verify output
                     Tests: data pipeline, file I/O in agent, verification
L3: word_count     — Should decompose into read→count→sort steps
                     Tests: judge decomposition, subtask execution, integration
L4: csv_pipeline   — Multi-step: parse CSV → filter → aggregate → output
                     Tests: data flow between subtasks, dependency ordering
L5: expression_eval— Build expression evaluator (tokenize→parse→compute)
                     Tests: deeper recursion, potential backtracking

Usage:
    DEEPSEEK_API_KEY=sk-xxx python eval/run_eval.py          # run all
    DEEPSEEK_API_KEY=sk-xxx python eval/run_eval.py --level 1 # run only L1
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# Make sure recursive_coder is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from recursive_coder.api_caller import APICaller
from recursive_coder.evaluator import Evaluator
from recursive_coder.executor import Executor
from recursive_coder.logger_setup import setup_logging, get_logger
from recursive_coder.models import DataPort, TaskStatus
from recursive_coder.persistence import Persistence
from recursive_coder.processor import RecursiveProcessor
from recursive_coder.prompt_builder import PromptBuilder


logger = get_logger("eval")


@dataclass
class TestCase:
    level: int
    name: str
    description: str               # what we ask the framework to do
    task_text: str                  # the actual task description for the processor
    data_port: DataPort             # root data port
    setup_files: dict[str, str]     # files to create in workspace before run
    expected_check: str             # how to verify success
    config_overrides: dict = field(default_factory=dict)
    max_api_calls: int = 50


# ── Test case definitions ──────────────────────────────────────────────────

CASES: list[TestCase] = [
    # ── L1: Trivial leaf task ──
    TestCase(
        level=1,
        name="echo_number",
        description="Write a Python script that prints 42. No decomposition needed.",
        task_text="Write a Python script called solution.py that prints the number 42.",
        data_port=DataPort(
            input_description="No input needed. The program should simply print 42.",
        ),
        setup_files={},
        expected_check="exact:42",
        config_overrides={"max_depth": 2, "max_retries": 2, "max_agent_steps": 15},
        max_api_calls=20,
    ),

    # ── L2: Data-driven verification ──
    TestCase(
        level=2,
        name="sum_from_file",
        description="Read numbers from a file and print their sum.",
        task_text=(
            "Read the file data/numbers.txt which contains one integer per line. "
            "Write a Python script sum.py that reads the file and prints the sum of all numbers."
        ),
        data_port=DataPort(
            input_description="A text file with one integer per line",
            input_files=["data/numbers.txt"],
        ),
        setup_files={
            "data/numbers.txt": "10\n20\n30\n40\n",
        },
        expected_check="contains:100",
        config_overrides={"max_depth": 2, "max_retries": 2, "max_agent_steps": 15},
        max_api_calls=30,
    ),

    # ── L3: Should trigger decomposition ──
    TestCase(
        level=3,
        name="word_count",
        description="Count word frequencies and output sorted result. May decompose.",
        task_text=(
            "Given the text file data/article.txt, write a program that:\n"
            "1. Reads the file\n"
            "2. Counts the frequency of each word (case-insensitive, strip punctuation)\n"
            "3. Outputs the top 5 most frequent words with counts, one per line, "
            "in format 'word: count', sorted by count descending.\n"
            "Save the result to output/top_words.txt and also print it to stdout."
        ),
        data_port=DataPort(
            input_description="A text file containing an English article",
            input_files=["data/article.txt"],
            output_files=["output/top_words.txt"],
        ),
        setup_files={
            "data/article.txt": (
                "The quick brown fox jumps over the lazy dog. "
                "The dog barked at the fox. The fox ran away. "
                "The dog chased the fox through the park. "
                "The quick fox jumped over the fence. "
                "A lazy dog slept in the sun. The dog is a good dog."
            ),
        },
        expected_check="contains:the",  # "the" should be the most frequent word
        config_overrides={"max_depth": 3, "max_retries": 2, "max_agent_steps": 20},
        max_api_calls=60,
    ),

    # ── L4: Multi-step data pipeline ──
    TestCase(
        level=4,
        name="csv_pipeline",
        description="Parse CSV, filter rows, compute aggregate. Tests data flow.",
        task_text=(
            "Process the CSV file data/sales.csv:\n"
            "1. Parse the CSV (columns: product, region, amount)\n"
            "2. Filter rows where region == 'North'\n"
            "3. Sum the 'amount' column for the filtered rows\n"
            "4. Write the total to output/north_total.txt\n"
            "5. Print the total to stdout.\n"
            "Expected output for the given data: 350"
        ),
        data_port=DataPort(
            input_description="A CSV file with columns: product, region, amount",
            input_files=["data/sales.csv"],
            output_files=["output/north_total.txt"],
        ),
        setup_files={
            "data/sales.csv": (
                "product,region,amount\n"
                "Widget,North,100\n"
                "Gadget,South,200\n"
                "Widget,North,150\n"
                "Doohickey,East,50\n"
                "Gadget,North,100\n"
                "Widget,South,75\n"
            ),
        },
        expected_check="contains:350",
        config_overrides={"max_depth": 4, "max_retries": 2, "max_agent_steps": 25},
        max_api_calls=80,
    ),

    # ── L5: Complex — expression evaluator ──
    TestCase(
        level=5,
        name="expression_eval",
        description="Build an arithmetic expression evaluator. Deeper decomposition.",
        task_text=(
            "Build a Python program eval_expr.py that evaluates simple arithmetic expressions.\n"
            "The program should:\n"
            "1. Read expressions from data/expressions.txt (one per line)\n"
            "2. Evaluate each expression (support +, -, *, / and parentheses)\n"
            "3. Write the results to output/results.txt, one result per line\n"
            "4. Print each result to stdout\n"
            "The expressions use integers only. Division should use integer division (//).\n"
            "Do NOT use eval() for security reasons. Implement a proper parser."
        ),
        data_port=DataPort(
            input_description="A text file with one arithmetic expression per line",
            input_files=["data/expressions.txt"],
            output_files=["output/results.txt"],
        ),
        setup_files={
            "data/expressions.txt": (
                "2 + 3\n"
                "10 - 4 * 2\n"
                "(1 + 2) * (3 + 4)\n"
                "100 // 3\n"
            ),
        },
        expected_check="contains:5",  # first expression should eval to 5
        config_overrides={"max_depth": 5, "max_retries": 3, "max_agent_steps": 30},
        max_api_calls=120,
    ),
]


# ── Evaluation runner ──────────────────────────────────────────────────────

@dataclass
class EvalResult:
    level: int
    name: str
    success: bool
    root_status: str
    api_calls: int
    total_tokens: int
    duration_s: float
    tree_depth: int
    leaf_count: int
    first_pass_rate: float
    final_pass_rate: float
    error: str = ""
    tree_repr: str = ""


async def run_case(case: TestCase, eval_dir: Path, use_mock: bool = False) -> EvalResult:
    """Run a single test case and return its evaluation result."""
    ws = eval_dir / f"L{case.level}_{case.name}"
    output_dir = ws / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup files in workspace
    for rel_path, content in case.setup_files.items():
        fpath = output_dir / rel_path
        fpath.parent.mkdir(parents=True, exist_ok=True)
        fpath.write_text(content, encoding="utf-8")

    setup_logging(str(ws), verbose=True)

    persistence = Persistence(str(ws))

    if use_mock:
        from eval.mock_api import MockAPICaller
        api = MockAPICaller(scenario=f"L{case.level}", persistence=persistence)
    else:
        api = APICaller(default_model="deepseek-v3", persistence=persistence)

    executor = Executor(
        workspace_dir=str(output_dir),
        timeout=case.config_overrides.get("command_timeout", 30),
        output_truncate=10000,
    )
    prompt_builder = PromptBuilder()
    evaluator = Evaluator(str(ws))

    config = {
        "default_model": "deepseek-v3",
        "max_total_api_calls": case.max_api_calls,
        **case.config_overrides,
    }

    processor = RecursiveProcessor(
        api_caller=api,
        prompt_builder=prompt_builder,
        executor=executor,
        persistence=persistence,
        config=config,
    )

    t0 = time.time()
    error = ""
    tree = None

    try:
        tree = await processor.run(case.task_text, data_port=case.data_port)
    except Exception as exc:
        error = str(exc)
        logger.error("Case L%d %s failed: %s", case.level, case.name, error)

    duration = time.time() - t0

    # Gather results
    if tree is None:
        return EvalResult(
            level=case.level, name=case.name, success=False,
            root_status="error", api_calls=api.call_count,
            total_tokens=api.total_input_tokens + api.total_output_tokens,
            duration_s=round(duration, 1), tree_depth=0, leaf_count=0,
            first_pass_rate=0, final_pass_rate=0, error=error,
        )

    # Generate evaluation report
    report = evaluator.generate_report(
        tree=tree, api_stats=api.get_stats(), config=config,
        backtrack_count=processor.backtrack_count,
    )
    persistence.save_report(report)

    root = tree.nodes.get(tree.root_id)
    root_status = root.status.value if root else "?"
    task_passed = root_status == "passed"

    # Additional output check
    output_ok = True
    if case.expected_check.startswith("exact:"):
        expected = case.expected_check[6:]
        # Check if any file or verification_result contains the expected output
        output_ok = _check_output(tree, expected, output_dir, "exact")
    elif case.expected_check.startswith("contains:"):
        expected = case.expected_check[9:]
        output_ok = _check_output(tree, expected, output_dir, "contains")

    # Count tree metrics
    leaves = [n for n in tree.nodes.values() if not n.children and n.id != tree.root_id]
    max_depth = max((n.depth for n in tree.nodes.values()), default=0)

    return EvalResult(
        level=case.level,
        name=case.name,
        success=task_passed and output_ok,
        root_status=root_status,
        api_calls=api.call_count,
        total_tokens=api.total_input_tokens + api.total_output_tokens,
        duration_s=round(duration, 1),
        tree_depth=max_depth,
        leaf_count=len(leaves),
        first_pass_rate=report.get("quality", {}).get("first_pass_rate", 0),
        final_pass_rate=report.get("quality", {}).get("final_pass_rate", 0),
        error=error,
        tree_repr=tree.print_tree(),
    )


def _check_output(tree, expected: str, output_dir: Path, mode: str) -> bool:
    """Check if the expected string appears in task outputs."""
    # Check verification results in leaf nodes
    for node in tree.nodes.values():
        vr = node.verification_result
        if vr:
            if mode == "exact" and expected.strip() == vr.strip():
                return True
            if mode == "contains" and expected in vr:
                return True

    # Check output files
    for f in output_dir.rglob("*"):
        if f.is_file() and f.suffix in (".txt", ".csv", ".json", ".py"):
            try:
                content = f.read_text(encoding="utf-8", errors="replace")
                if mode == "exact" and expected.strip() == content.strip():
                    return True
                if mode == "contains" and expected in content:
                    return True
            except Exception:
                pass

    return False


# ── Report formatting ──────────────────────────────────────────────────────

def print_report(results: list[EvalResult]) -> str:
    lines = []
    lines.append("")
    lines.append("=" * 72)
    lines.append("  RECURSIVE-CODER END-TO-END EVALUATION REPORT")
    lines.append("=" * 72)
    lines.append("")

    passed = sum(1 for r in results if r.success)
    total = len(results)
    lines.append(f"  Overall: {passed}/{total} passed")
    lines.append("")

    lines.append(f"  {'Level':<6} {'Name':<20} {'Status':<10} {'API':<6} {'Tokens':<10} {'Time':<8} {'Depth':<6} {'Leaves':<7} {'FPR':<6} {'FinalPR'}")
    lines.append("  " + "-" * 95)

    for r in results:
        status = "PASS" if r.success else "FAIL"
        lines.append(
            f"  L{r.level:<5} {r.name:<20} {status:<10} {r.api_calls:<6} "
            f"{r.total_tokens:<10} {r.duration_s:<8} {r.tree_depth:<6} "
            f"{r.leaf_count:<7} {r.first_pass_rate:<6} {r.final_pass_rate}"
        )

    lines.append("")

    # Detailed results
    for r in results:
        lines.append(f"  --- L{r.level}: {r.name} ---")
        lines.append(f"  Root status: {r.root_status}")
        if r.error:
            lines.append(f"  Error: {r.error[:200]}")
        if r.tree_repr:
            for tl in r.tree_repr.split("\n"):
                lines.append(f"    {tl}")
        lines.append("")

    lines.append("=" * 72)

    report = "\n".join(lines)
    return report


# ── Main ───────────────────────────────────────────────────────────────────

async def main(levels: list[int] | None = None, use_mock: bool = False):
    # Ensure API key (not needed for mock mode)
    if not use_mock and not os.environ.get("DEEPSEEK_API_KEY"):
        os.environ["DEEPSEEK_API_KEY"] = "sk-1abb4628c5ed45bf8b933ce86299e866"

    mode_label = "MOCK" if use_mock else "LIVE"
    eval_dir = Path("eval_runs") / f"{time.strftime('%Y%m%d_%H%M%S')}_{mode_label}"
    eval_dir.mkdir(parents=True, exist_ok=True)

    cases = CASES
    if levels:
        cases = [c for c in CASES if c.level in levels]

    print(f"\nRunning {len(cases)} test cases ({mode_label} mode)...")
    print(f"Results will be saved to: {eval_dir}\n")

    results: list[EvalResult] = []
    for case in cases:
        print(f"  [{time.strftime('%H:%M:%S')}] L{case.level}: {case.name} — {case.description}")
        result = await run_case(case, eval_dir, use_mock=use_mock)
        results.append(result)
        status = "PASS" if result.success else "FAIL"
        print(f"  [{time.strftime('%H:%M:%S')}] L{case.level}: {status} "
              f"(api={result.api_calls}, tokens={result.total_tokens}, "
              f"time={result.duration_s}s)")
        print()

    report = print_report(results)
    print(report)

    # Save report
    report_path = eval_dir / "eval_report.txt"
    report_path.write_text(report, encoding="utf-8")

    # Save structured results
    json_path = eval_dir / "eval_results.json"
    json_data = []
    for r in results:
        json_data.append({
            "level": r.level, "name": r.name, "success": r.success,
            "root_status": r.root_status, "api_calls": r.api_calls,
            "total_tokens": r.total_tokens, "duration_s": r.duration_s,
            "tree_depth": r.tree_depth, "leaf_count": r.leaf_count,
            "first_pass_rate": r.first_pass_rate,
            "final_pass_rate": r.final_pass_rate,
            "error": r.error,
        })
    json_path.write_text(json.dumps(json_data, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nResults saved to: {eval_dir}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run end-to-end evaluation")
    parser.add_argument("--level", type=int, nargs="+", help="Only run specific levels (1-5)")
    parser.add_argument("--mock", action="store_true", help="Use mock API (no network needed)")
    args = parser.parse_args()
    asyncio.run(main(args.level, use_mock=args.mock))
