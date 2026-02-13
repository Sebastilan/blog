"""CLI entry point for recursive-coder."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

import yaml

from .api_caller import APICaller
from .evaluator import Evaluator
from .executor import Executor
from .logger_setup import setup_logging, get_logger
from .models import DataPort
from .optimizer import Optimizer
from .persistence import Persistence
from .processor import RecursiveProcessor
from .prompt_builder import PromptBuilder

logger = get_logger("cli")


def _load_config(project_dir: Path) -> dict:
    config_path = project_dir / "config.yaml"
    if config_path.exists():
        return yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    return {}


def _make_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# ── Subcommands ──────────────────────────────────────────────────────────────

async def cmd_run(args: argparse.Namespace) -> None:
    project_dir = Path(args.project_dir).resolve()
    config = _load_config(project_dir)

    # CLI overrides
    if args.model:
        config["default_model"] = args.model
    if args.max_depth:
        config["max_depth"] = args.max_depth
    if args.max_retries:
        config["max_retries"] = args.max_retries
    if args.max_calls:
        config["max_total_api_calls"] = args.max_calls

    run_id = _make_run_id()
    ws = Path(args.workspace) / run_id
    ws.mkdir(parents=True, exist_ok=True)

    setup_logging(str(ws), verbose=args.verbose)

    persistence = Persistence(str(ws))
    api = APICaller(
        default_model=config.get("default_model", "deepseek-v3"),
        persistence=persistence,
    )
    executor = Executor(
        workspace_dir=str(ws / "output"),
        timeout=config.get("command_timeout", 60),
        output_truncate=config.get("output_truncate", 10000),
        blacklist=config.get("command_blacklist"),
    )
    prompt_builder = PromptBuilder(project_dir / "prompt_templates")
    evaluator = Evaluator(str(ws))

    processor = RecursiveProcessor(
        api_caller=api,
        prompt_builder=prompt_builder,
        executor=executor,
        persistence=persistence,
        config=config,
    )

    # Build data port if user provided data files
    data_port = DataPort()
    if args.data:
        data_port.input_files = [args.data]
        data_port.input_description = f"Test data file: {args.data}"

    # Run
    evaluator.record_event("run_start", detail=args.task)
    tree = await processor.run(args.task, data_port=data_port)
    evaluator.record_event("run_end", detail=tree.nodes[tree.root_id].status.value if tree.root_id else "?")

    # Generate report
    report = evaluator.generate_report(
        tree=tree,
        api_stats=api.get_stats(),
        config=config,
        backtrack_count=processor.backtrack_count,
    )
    persistence.save_report(report)

    summary = evaluator.print_summary(report)
    print(summary)
    print(f"  Workspace: {ws}")
    print(f"  Tree:\n{tree.print_tree()}")


async def cmd_iterate(args: argparse.Namespace) -> None:
    project_dir = Path(args.project_dir).resolve()
    config = _load_config(project_dir)
    if args.model:
        config["default_model"] = args.model

    rounds = args.rounds or config.get("optimization_rounds", 3)
    auto = args.auto or config.get("auto_apply_optimization", False)

    api_for_optimizer = APICaller(
        default_model=config.get("optimizer_model") or config.get("default_model", "deepseek-v3"),
    )
    optimizer = Optimizer(api_for_optimizer, str(project_dir), model_name=config.get("optimizer_model"))

    prev_report = None
    prev_iteration = None

    for round_num in range(1, rounds + 1):
        print(f"\n{'='*60}")
        print(f"  Iteration {round_num}/{rounds}")
        print(f"{'='*60}\n")

        # Run
        run_args = argparse.Namespace(
            task=args.task,
            model=args.model,
            workspace=args.workspace,
            project_dir=args.project_dir,
            data=args.data,
            max_depth=None,
            max_retries=None,
            max_calls=None,
            verbose=args.verbose,
        )
        await cmd_run(run_args)

        # Load the latest report
        ws_path = Path(args.workspace)
        run_dirs = sorted(ws_path.iterdir()) if ws_path.exists() else []
        if not run_dirs:
            print("No run found, stopping iteration.")
            break

        latest_ws = run_dirs[-1]
        report_path = latest_ws / "evaluation_report.json"
        if not report_path.exists():
            print("No report found, stopping iteration.")
            break

        current_report = json.loads(report_path.read_text(encoding="utf-8"))

        # Compare with previous
        if prev_report:
            comparison = optimizer.print_comparison(prev_report, current_report)
            print(comparison)

        # Optimize (except last round)
        if round_num < rounds:
            print("  Analyzing and generating optimization suggestions...")
            suggestions = await optimizer.analyze(
                report=current_report,
                config=config,
                previous_iteration=prev_iteration,
            )

            if "raw_response" in suggestions:
                print("  Optimizer returned unstructured response:")
                print(f"  {suggestions['raw_response'][:500]}")
            else:
                # Show suggestions
                analysis = suggestions.get("analysis", {})
                print("\n  Issues found:")
                for issue in analysis.get("main_issues", []):
                    print(f"    - {issue}")
                print("\n  Suggested changes:")
                for pc in suggestions.get("prompt_changes", []):
                    print(f"    prompt/{pc.get('file')}: {pc.get('reason')}")
                for cc in suggestions.get("config_changes", []):
                    print(f"    config/{cc.get('key')}: {cc.get('reason')}")

                if auto:
                    applied = optimizer.apply_suggestions(suggestions)
                    print(f"\n  Auto-applied {len(applied)} changes.")
                    for a in applied:
                        print(f"    {a}")
                else:
                    print("\n  Apply these changes? [y/n] ", end="")
                    answer = input().strip().lower()
                    if answer in ("y", "yes"):
                        applied = optimizer.apply_suggestions(suggestions)
                        print(f"  Applied {len(applied)} changes.")
                    else:
                        applied = []
                        print("  Skipped.")

                # Save iteration record
                optimizer.save_iteration(
                    before_report=current_report,
                    after_report=None,  # will be filled next round
                    suggestions=suggestions,
                    applied=applied if 'applied' in dir() else [],
                )

                # Reload config for next round
                config = _load_config(project_dir)

            prev_iteration = suggestions

        prev_report = current_report

    print("\nIteration complete.")


async def cmd_report(args: argparse.Namespace) -> None:
    report_path = Path(args.run_dir) / "evaluation_report.json"
    if not report_path.exists():
        print(f"Report not found: {report_path}")
        return
    report = json.loads(report_path.read_text(encoding="utf-8"))
    evaluator = Evaluator(args.run_dir)
    print(evaluator.print_summary(report))


async def cmd_history(args: argparse.Namespace) -> None:
    project_dir = Path(args.project_dir).resolve()
    history_dir = project_dir / "optimization_history"
    if not history_dir.exists():
        print("No optimization history found.")
        return
    for f in sorted(history_dir.glob("iteration_*.json")):
        data = json.loads(f.read_text(encoding="utf-8"))
        verdict = data.get("verdict", "?")
        before = data.get("before", {})
        after = data.get("after", {})
        print(f"  Iteration {data.get('iteration', '?')}: {verdict}")
        if after:
            print(f"    first_pass: {before.get('first_pass_rate')} → {after.get('first_pass_rate')}")
            print(f"    final_pass: {before.get('final_pass_rate')} → {after.get('final_pass_rate')}")
        print()


# ── Main ─────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="recursive-coder",
        description="Recursive Task Decomposition AI Coding Framework",
    )
    parser.add_argument("--project-dir", default=".", help="Project root (default: .)")
    sub = parser.add_subparsers(dest="command")

    # run
    p_run = sub.add_parser("run", help="Run a single task")
    p_run.add_argument("task", help="Task description")
    p_run.add_argument("--model", help="Model to use")
    p_run.add_argument("--workspace", default="./workspace", help="Workspace directory")
    p_run.add_argument("--data", help="Input data file path (relative to workspace)")
    p_run.add_argument("--max-depth", type=int)
    p_run.add_argument("--max-retries", type=int)
    p_run.add_argument("--max-calls", type=int)
    p_run.add_argument("--verbose", action="store_true")

    # iterate
    p_iter = sub.add_parser("iterate", help="Run iterative optimization")
    p_iter.add_argument("task", help="Task description")
    p_iter.add_argument("--model", help="Model to use")
    p_iter.add_argument("--workspace", default="./workspace")
    p_iter.add_argument("--data", help="Input data file path")
    p_iter.add_argument("--rounds", type=int, help="Number of iteration rounds")
    p_iter.add_argument("--auto", action="store_true", help="Auto-apply optimizations")
    p_iter.add_argument("--verbose", action="store_true")

    # report
    p_report = sub.add_parser("report", help="View evaluation report")
    p_report.add_argument("run_dir", help="Path to run workspace directory")

    # history
    sub.add_parser("history", help="View optimization history")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "run":
        asyncio.run(cmd_run(args))
    elif args.command == "iterate":
        asyncio.run(cmd_iterate(args))
    elif args.command == "report":
        asyncio.run(cmd_report(args))
    elif args.command == "history":
        asyncio.run(cmd_history(args))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
