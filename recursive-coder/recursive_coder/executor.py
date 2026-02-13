"""Shell execution layer with safety controls."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path

from .logger_setup import get_logger

logger = get_logger("executor")

# Default blacklist patterns — checked with `in` against the raw command string.
DEFAULT_BLACKLIST = [
    "rm -rf /",
    "sudo ",
    "reboot",
    "shutdown",
    "systemctl",
    "mkfs",
    "dd if=",
    ":(){",        # fork bomb
    "> /dev/sd",
]


@dataclass
class ExecutionResult:
    command: str
    returncode: int = -1
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    duration_ms: int = 0
    blocked: bool = False
    block_reason: str = ""


class Executor:
    """Run shell commands inside a workspace with timeout & safety."""

    def __init__(
        self,
        workspace_dir: str,
        timeout: int = 60,
        output_truncate: int = 10_000,
        blacklist: list[str] | None = None,
    ) -> None:
        self.workspace = Path(workspace_dir)
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.output_truncate = output_truncate
        self.blacklist = blacklist or list(DEFAULT_BLACKLIST)

    def _check_safety(self, command: str) -> tuple[bool, str]:
        for pattern in self.blacklist:
            if pattern in command:
                return False, f"blocked by blacklist: {pattern!r}"
        return True, ""

    def _truncate(self, text: str) -> str:
        if len(text) > self.output_truncate:
            return text[: self.output_truncate] + f"\n... [truncated, {len(text)} chars total]"
        return text

    async def run(self, command: str, timeout: int | None = None) -> ExecutionResult:
        safe, reason = self._check_safety(command)
        if not safe:
            logger.warning("Command blocked: %s (%s)", command, reason)
            return ExecutionResult(
                command=command, blocked=True, block_reason=reason,
            )

        t0 = time.monotonic()
        effective_timeout = timeout or self.timeout
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.workspace),
            )
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(), timeout=effective_timeout,
            )
            elapsed = int((time.monotonic() - t0) * 1000)
            return ExecutionResult(
                command=command,
                returncode=proc.returncode or 0,
                stdout=self._truncate(stdout_bytes.decode(errors="replace")),
                stderr=self._truncate(stderr_bytes.decode(errors="replace")),
                duration_ms=elapsed,
            )
        except asyncio.TimeoutError:
            elapsed = int((time.monotonic() - t0) * 1000)
            logger.warning("Command timed out after %ds: %s", effective_timeout, command)
            try:
                proc.kill()  # type: ignore[possibly-undefined]
            except Exception:
                pass
            return ExecutionResult(
                command=command, timed_out=True, duration_ms=elapsed,
            )
        except Exception as exc:
            elapsed = int((time.monotonic() - t0) * 1000)
            return ExecutionResult(
                command=command,
                returncode=-1,
                stderr=str(exc),
                duration_ms=elapsed,
            )

    async def write_file(self, rel_path: str, content: str) -> str:
        """Write content to a file inside the workspace. Returns abs path."""
        target = (self.workspace / rel_path).resolve()
        if not str(target).startswith(str(self.workspace.resolve())):
            return f"ERROR: path escapes workspace: {rel_path}"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        logger.info("Wrote file: %s (%d bytes)", rel_path, len(content))
        return f"OK: wrote {rel_path}"

    async def read_file(self, rel_path: str) -> str:
        target = (self.workspace / rel_path).resolve()
        if not str(target).startswith(str(self.workspace.resolve())):
            return f"ERROR: path escapes workspace: {rel_path}"
        if not target.exists():
            return f"ERROR: file not found: {rel_path}"
        text = target.read_text(encoding="utf-8", errors="replace")
        return self._truncate(text)

    async def list_dir(self, rel_path: str = ".") -> str:
        target = (self.workspace / rel_path).resolve()
        if not str(target).startswith(str(self.workspace.resolve())):
            return f"ERROR: path escapes workspace: {rel_path}"
        if not target.is_dir():
            return f"ERROR: not a directory: {rel_path}"
        entries = sorted(target.iterdir())
        lines = []
        for e in entries:
            kind = "d" if e.is_dir() else "f"
            lines.append(f"[{kind}] {e.relative_to(self.workspace)}")
        return "\n".join(lines) if lines else "(empty directory)"
