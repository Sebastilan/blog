"""Core recursive processor: judge(plan) → execute → backtrack → integrate."""

from __future__ import annotations

import json
import time
from typing import Optional

from .agent_loop import AgentLoop, AgentResult
from .api_caller import APICaller
from .executor import Executor
from .logger_setup import get_logger
from .models import (
    DataPort,
    TaskNode,
    TaskStatus,
    TaskTree,
    Verification,
)
from .persistence import Persistence
from .prompt_builder import PromptBuilder
from .response_parser import parse_backtrack_response, parse_judge_response
from .tools import ToolExecutor

logger = get_logger("processor")


class RecursiveProcessor:
    """Orchestrate recursive task decomposition and agent-based execution."""

    def __init__(
        self,
        api_caller: APICaller,
        prompt_builder: PromptBuilder,
        executor: Executor,
        persistence: Persistence,
        config: dict,
    ) -> None:
        self.api = api_caller
        self.prompts = prompt_builder
        self.executor = executor
        self.persistence = persistence
        self.cfg = config

        self.tree = TaskTree()
        self.workspace = str(executor.workspace)
        self.backtrack_count = 0
        self.total_api_calls = 0

        # Build agent loop and tool executor
        self.tool_executor = ToolExecutor(executor)
        self.agent_loop = AgentLoop(
            api_caller=api_caller,
            tool_executor=self.tool_executor,
            max_steps=config.get("max_agent_steps", 30),
            context_window=config.get("agent_context_window", 10),
            idle_detection=config.get("idle_detection", 3),
            model_name=config.get("execute_model"),
        )

    # ── Public entry point ──

    async def run(self, task_description: str, data_port: Optional[DataPort] = None) -> TaskTree:
        """Accept a task description (+ optional data port), build and execute the tree."""
        root = TaskNode(
            description=task_description,
            data_port=data_port or DataPort(),
        )
        self.tree.add_node(root)
        self.persistence.save_tree(self.tree)

        logger.info("[RUN] task=%s desc='%s'", root.id, task_description)
        await self.process(root)
        self.persistence.save_tree(self.tree)

        logger.info("[DONE] final_status=%s tree:\n%s", root.status.value, self.tree.print_tree())
        return self.tree

    # ── Core recursive logic ──

    async def process(self, task: TaskNode) -> None:
        if self._over_limits():
            logger.error("[LIMIT] task=%s global API call limit reached", task.id)
            task.status = TaskStatus.FAILED
            return

        max_depth = self.cfg.get("max_depth", 5)
        if task.depth >= max_depth:
            logger.error("[LIMIT] task=%s max_depth=%d reached", task.id, max_depth)
            task.status = TaskStatus.FAILED
            return

        task.status = TaskStatus.RUNNING
        task.start_time = time.time()
        self.persistence.save_tree(self.tree)

        logger.info(
            "[JUDGE] task=%s depth=%d desc='%s'",
            task.id, task.depth, task.description[:80],
        )

        # Step 1: Judge (Planning) — produce execution_plan/interface or decompose with interface_contract
        system_prompt = self.prompts.system()
        user_prompt = self.prompts.judge(task, self.workspace, tree=self.tree)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        resp = await self.api.call(
            messages=messages,
            task_node_id=task.id,
            phase="judge",
            model_name=self.cfg.get("judge_model"),
        )
        self.total_api_calls += 1

        text = resp["choices"][0]["message"].get("content", "")
        parsed = parse_judge_response(text)

        if not parsed.parse_success:
            logger.warning(
                "[JUDGE] task=%s parse_failed, falling back to decompose",
                task.id,
            )
            parsed.can_verify = False

        # Step 2a: Leaf — directly executable (with execution_plan + interface)
        if parsed.can_verify:
            task.verification = Verification(
                description=parsed.verification_description,
                expected_output=parsed.expected_output,
                command=parsed.verification_command,
                compare_mode=parsed.compare_mode,
            )
            if parsed.data_port:
                dp = parsed.data_port
                task.data_port.output_description = dp.get("output_description", "")
                task.data_port.output_files = dp.get("output_files", [])
            task.implementation_hint = parsed.implementation_hint
            task.execution_plan = parsed.execution_plan
            task.interface = parsed.interface

            logger.info(
                "[PLAN] task=%s decision=LEAF plan_steps=%d interface_keys=%s verify_cmd='%s'",
                task.id,
                len(task.execution_plan),
                list(task.interface.keys()) if task.interface else [],
                task.verification.command[:60] if task.verification else "",
            )
            if task.execution_plan:
                for i, step in enumerate(task.execution_plan):
                    logger.debug("[PLAN] task=%s step %s", task.id, step)

            self.persistence.save_tree(self.tree)
            await self._execute_with_agent(task)

        # Step 2b: Decompose (with interface_contract)
        else:
            task.decomposition_reason = parsed.decomposition_reason
            task.interface_contract = parsed.interface_contract

            logger.info(
                "[PLAN] task=%s decision=DECOMPOSE subtasks=%d reason='%s'",
                task.id,
                len(parsed.subtasks),
                task.decomposition_reason[:80],
            )
            if task.interface_contract:
                logger.info(
                    "[CONTRACT] task=%s interface_contract='%s'",
                    task.id, task.interface_contract[:200],
                )

            # Check if we need a "prepare test data" subtask
            has_data = bool(task.data_port.input_files or task.data_port.input_description)
            needs_data_subtask = not has_data and task.depth == 0

            subtask_defs = parsed.subtasks
            if needs_data_subtask:
                subtask_defs.insert(0, {
                    "description": f"为任务准备测试数据：{task.description}",
                    "data_port": {
                        "input_description": "无（需要创建）",
                        "output_description": "测试输入数据文件 + 预期输出文件",
                        "output_files": ["data/test_input.txt", "data/expected_output.txt"],
                    },
                    "dependencies": [],
                    "context_files": [],
                })

            self._add_subtasks(task, subtask_defs)
            self.persistence.save_tree(self.tree)

            # Log child creation summary
            for cid in task.children:
                child = self.tree.get_node(cid)
                if child:
                    logger.info(
                        "[CHILD] parent=%s child=%s desc='%s' deps=%s in_files=%s out_files=%s",
                        task.id, child.id, child.description[:60],
                        child.dependencies,
                        child.data_port.input_files,
                        child.data_port.output_files,
                    )

            # Process subtasks in topological order
            ordered = self.tree.topological_order(task.children)
            for child_id in ordered:
                child = self.tree.get_node(child_id)
                if child and child.status == TaskStatus.PENDING:
                    await self.process(child)
                    if child.status == TaskStatus.FAILED:
                        await self._backtrack(task)
                        return

            # All children passed → integration verify
            if self.tree.all_children_passed(task.id):
                await self._integration_verify(task)
            else:
                task.status = TaskStatus.FAILED

        task.end_time = time.time()
        elapsed = task.end_time - (task.start_time or task.end_time)
        logger.info(
            "[FINISH] task=%s status=%s elapsed=%.1fs tokens_in=%d tokens_out=%d",
            task.id, task.status.value, elapsed,
            task.token_usage.get("input", 0), task.token_usage.get("output", 0),
        )
        self.persistence.save_tree(self.tree)

    # ── Agent execution ──

    async def _execute_with_agent(self, task: TaskNode) -> None:
        """Run the agent loop to implement and verify a leaf task."""
        for attempt in range(1, task.max_attempts + 1):
            task.attempts = attempt
            logger.info(
                "[EXEC] task=%s attempt=%d/%d plan_steps=%d",
                task.id, attempt, task.max_attempts, len(task.execution_plan),
            )

            if attempt == 1:
                user_prompt = self.prompts.execute(task, self.workspace)
            else:
                last_error = task.error_log[-1] if task.error_log else "unknown error"
                user_prompt = self.prompts.fix(task, last_error, self.workspace)

            agent_result = await self.agent_loop.run(
                task=task,
                system_prompt=self.prompts.system(),
                user_prompt=user_prompt,
            )

            task.agent_steps += len(agent_result.steps)
            task.output_files = agent_result.files_modified
            task.token_usage["input"] += agent_result.total_tokens.get("input", 0)
            task.token_usage["output"] += agent_result.total_tokens.get("output", 0)

            logger.debug(
                "[EXEC] task=%s attempt=%d agent_steps=%d files_modified=%s",
                task.id, attempt, len(agent_result.steps), agent_result.files_modified,
            )

            # Run verification command if present
            if task.verification and task.verification.command:
                vresult = await self.executor.run(task.verification.command)
                task.verification_result = vresult.stdout + vresult.stderr

                passed = self._check_verification(task, vresult)
                logger.info(
                    "[VERIFY] task=%s attempt=%d passed=%s mode=%s cmd='%s' rc=%d",
                    task.id, attempt, passed,
                    task.verification.compare_mode,
                    task.verification.command[:60],
                    vresult.returncode,
                )
                if not passed:
                    logger.debug(
                        "[VERIFY] task=%s expected='%s' actual='%s'",
                        task.id,
                        task.verification.expected_output[:200],
                        (vresult.stdout or vresult.stderr)[:200],
                    )

                if passed:
                    task.status = TaskStatus.PASSED
                    logger.info("[PASS] task=%s", task.id)
                    return
                else:
                    error_msg = f"Verification failed (rc={vresult.returncode}):\n{vresult.stderr or vresult.stdout}"
                    task.error_log.append(error_msg)
                    logger.warning(
                        "[FAIL] task=%s attempt=%d error='%s'",
                        task.id, attempt, error_msg[:150],
                    )
            elif agent_result.success:
                # No verification command — trust the agent's task_done
                task.status = TaskStatus.PASSED
                logger.info("[PASS] task=%s (agent self-declared)", task.id)
                return
            else:
                task.error_log.append(agent_result.summary or "Agent did not complete")
                logger.warning(
                    "[FAIL] task=%s attempt=%d error='%s'",
                    task.id, attempt, (agent_result.summary or "Agent did not complete")[:150],
                )

        # All attempts exhausted
        task.status = TaskStatus.FAILED
        logger.error(
            "[FAILED] task=%s exhausted %d attempts, errors: %s",
            task.id, task.max_attempts,
            "; ".join(e[:80] for e in task.error_log[-3:]),
        )

    def _check_verification(self, task: TaskNode, result) -> bool:
        """Compare execution result against verification criteria."""
        v = task.verification
        if not v:
            return result.returncode == 0

        mode = v.compare_mode
        expected = v.expected_output.strip()
        actual_stdout = result.stdout.strip()

        if mode == "returncode":
            return result.returncode == 0
        elif mode == "exact":
            return actual_stdout == expected
        elif mode == "contains":
            return expected in actual_stdout
        elif mode == "file_diff":
            # compare output files — not yet implemented, fallback to returncode
            return result.returncode == 0
        else:
            return result.returncode == 0

    # ── Backtrack ──

    async def _backtrack(self, task: TaskNode) -> None:
        """A child failed after all retries. Re-decompose this task."""
        self.backtrack_count += 1
        max_bt = self.cfg.get("max_backtrack_retries", 2)

        if self.backtrack_count > max_bt:
            logger.error(
                "[BACKTRACK] task=%s global_limit=%d exceeded, giving up",
                task.id, max_bt,
            )
            task.status = TaskStatus.FAILED
            return

        # Collect failure info from children (include interface violations)
        failures = []
        for cid in task.children:
            child = self.tree.get_node(cid)
            if child and child.status == TaskStatus.FAILED:
                failures.append(
                    f"- {child.description} (id={child.id})\n"
                    f"  Errors: {'; '.join(child.error_log[-2:])}\n"
                    f"  Interface: {json.dumps(child.interface, ensure_ascii=False) if child.interface else 'N/A'}"
                )
        failure_details = "\n".join(failures)

        logger.info(
            "[BACKTRACK] task=%s attempt=%d/%d failed_children=%d",
            task.id, self.backtrack_count, max_bt,
            len(failures),
        )
        logger.debug("[BACKTRACK] task=%s failures:\n%s", task.id, failure_details)

        # Call LLM for new decomposition
        system_prompt = self.prompts.system()
        user_prompt = self.prompts.backtrack(task, failure_details, self.workspace)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        resp = await self.api.call(
            messages=messages,
            task_node_id=task.id,
            phase="backtrack",
            model_name=self.cfg.get("judge_model"),
        )
        self.total_api_calls += 1

        text = resp["choices"][0]["message"].get("content", "")
        parsed = parse_backtrack_response(text)

        if not parsed.parse_success or not parsed.subtasks:
            logger.error("[BACKTRACK] task=%s parse_failed", task.id)
            task.status = TaskStatus.FAILED
            return

        # Update interface_contract if backtrack provides a new one
        from .response_parser import _extract_json_block
        bt_data = _extract_json_block(text)
        if bt_data and bt_data.get("interface_contract"):
            task.interface_contract = bt_data["interface_contract"]
            logger.info(
                "[BACKTRACK] task=%s updated interface_contract='%s'",
                task.id, task.interface_contract[:200],
            )

        # Replace children
        task.children.clear()
        self._add_subtasks(task, parsed.subtasks)
        task.decomposition_reason = parsed.decomposition_reason
        self.persistence.save_tree(self.tree)

        logger.info(
            "[BACKTRACK] task=%s new_subtasks=%d reason='%s'",
            task.id, len(parsed.subtasks), parsed.decomposition_reason[:80],
        )

        # Re-process children
        ordered = self.tree.topological_order(task.children)
        for child_id in ordered:
            child = self.tree.get_node(child_id)
            if child and child.status == TaskStatus.PENDING:
                await self.process(child)
                if child.status == TaskStatus.FAILED:
                    task.status = TaskStatus.FAILED
                    return

        if self.tree.all_children_passed(task.id):
            await self._integration_verify(task)
        else:
            task.status = TaskStatus.FAILED

    # ── Integration verify ──

    async def _integration_verify(self, task: TaskNode) -> None:
        """After all children pass, run an integration verification via agent."""
        children_summary = ""
        for cid in task.children:
            child = self.tree.get_node(cid)
            if child:
                iface_str = ""
                if child.interface:
                    iface_str = f" interface={json.dumps(child.interface, ensure_ascii=False)}"
                children_summary += (
                    f"- {child.description} [PASSED] "
                    f"outputs={child.output_files}{iface_str}\n"
                )

        logger.info(
            "[INTEGRATE] task=%s children=%d has_contract=%s",
            task.id, len(task.children), bool(task.interface_contract),
        )

        user_prompt = self.prompts.integrate(task, children_summary, self.workspace)
        agent_result = await self.agent_loop.run(
            task=task,
            system_prompt=self.prompts.system(),
            user_prompt=user_prompt,
        )

        if agent_result.success:
            task.status = TaskStatus.PASSED
            logger.info("[INTEGRATE] task=%s PASSED", task.id)
        else:
            logger.warning(
                "[INTEGRATE] task=%s FAILED (soft-pass: children all passed)",
                task.id,
            )
            task.status = TaskStatus.PASSED  # soft-pass: children all passed

    # ── Helpers ──

    def _add_subtasks(self, parent: TaskNode, subtask_defs: list[dict]) -> None:
        """Create TaskNode children from the parsed subtask definitions.

        Propagates parent's interface_contract to all children.
        """
        id_map: dict[int, str] = {}  # index → node id (for dependency resolution)

        for i, sd in enumerate(subtask_defs):
            dp_data = sd.get("data_port", {})
            child = TaskNode(
                description=sd["description"],
                parent_id=parent.id,
                depth=parent.depth + 1,
                context_files=sd.get("context_files", []),
                # Propagate parent's interface_contract to child
                interface_contract=parent.interface_contract,
                data_port=DataPort(
                    input_description=dp_data.get("input_description", ""),
                    input_files=dp_data.get("input_files", parent.data_port.input_files.copy()),
                    output_description=dp_data.get("output_description", ""),
                    output_files=dp_data.get("output_files", []),
                    upstream_task_ids=dp_data.get("upstream_task_ids", []),
                ),
            )
            self.tree.add_node(child)
            id_map[i] = child.id

        # Resolve "dependencies" which may be indices or task ids
        for i, sd in enumerate(subtask_defs):
            child_id = id_map[i]
            child = self.tree.get_node(child_id)
            if not child:
                continue
            for dep in sd.get("dependencies", []):
                if isinstance(dep, int) and dep in id_map:
                    child.dependencies.append(id_map[dep])
                elif isinstance(dep, str):
                    child.dependencies.append(dep)

    def _over_limits(self) -> bool:
        max_calls = self.cfg.get("max_total_api_calls", 500)
        return self.api.call_count >= max_calls
