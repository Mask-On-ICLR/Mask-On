from __future__ import annotations
import ast
import re
import shutil
import subprocess
import tempfile
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

@dataclass(frozen=True, slots=True)
class DownstreamExample:
    """Prompt, reference answer, and task-specific scoring fields."""

    id: str
    task: str
    prompt: str
    reference: str
    metadata: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "task": self.task,
            "prompt": self.prompt,
            "reference": self.reference,
            "metadata": dict(self.metadata),
        }


class TaskScorer(ABC):
    """Interface for scoring one generated response."""

    @abstractmethod
    def score(self, generated: str, example: DownstreamExample) -> dict[str, Any]:
        """Return a JSON-compatible score record."""


@dataclass(frozen=True, slots=True)
class SandboxResult:
    passed: bool
    status: str
    returncode: int | None
    stderr: str


class BubblewrapPythonSandbox:
    """Execute MBPP code in a sandbox with read-only host mounts and network isolation."""

    def __init__(
        self,
        *,
        executable: str = "bwrap",
        python: str = "/usr/bin/python3",
        timeout_seconds: float = 4.0,
        cpu_seconds: int = 2,
        max_code_bytes: int = 64_000,
    ) -> None:
        if cpu_seconds < 1:
            raise ValueError("sandbox cpu_seconds must be positive")
        if timeout_seconds <= cpu_seconds:
            raise ValueError("sandbox wall timeout must exceed the CPU limit")
        self._executable = executable
        self._python = python
        self._timeout_seconds = timeout_seconds
        self._cpu_seconds = cpu_seconds
        self._max_code_bytes = max_code_bytes

    def run(self, code: str, imports: Sequence[str], tests: Sequence[str]) -> SandboxResult:
        if shutil.which(self._executable) is None:
            return SandboxResult(False, "sandbox_unavailable", None, "bubblewrap not found")
        if len(code.encode("utf-8")) > self._max_code_bytes:
            return SandboxResult(False, "code_too_large", None, "generated code exceeds limit")
        try:
            ast.parse(code)
        except SyntaxError as exc:
            return SandboxResult(False, "syntax_error", None, str(exc))
        except (MemoryError, RecursionError) as exc:
            return SandboxResult(
                False,
                "parser_resource_error",
                None,
                type(exc).__name__,
            )
        wrapper = self._wrapper(code, imports, tests)
        with tempfile.TemporaryDirectory(prefix="mask-on-mbpp-") as directory:
            runner = Path(directory) / "runner.py"
            runner.write_text(wrapper, encoding="utf-8")
            command = [
                self._executable,
                "--unshare-all",
                "--die-with-parent",
                "--new-session",
                "--ro-bind",
                "/usr",
                "/usr",
                "--ro-bind",
                "/lib",
                "/lib",
                "--ro-bind",
                "/lib64",
                "/lib64",
                "--proc",
                "/proc",
                "--dev",
                "/dev",
                "--tmpfs",
                "/tmp",
                "--ro-bind",
                str(runner),
                "/runner.py",
                self._python,
                "-I",
                "/runner.py",
            ]
            try:
                completed = subprocess.run(
                    command,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=self._timeout_seconds,
                    env={"PATH": "/usr/bin", "PYTHONHASHSEED": "0"},
                )
            except subprocess.TimeoutExpired:
                return SandboxResult(False, "timeout", None, "sandbox timeout")
        stderr = completed.stderr[-2000:]
        return SandboxResult(
            passed=completed.returncode == 0,
            status="passed" if completed.returncode == 0 else "failed_tests_or_runtime",
            returncode=completed.returncode,
            stderr=stderr,
        )

    def _wrapper(self, code: str, imports: Sequence[str], tests: Sequence[str]) -> str:
        prelude = f"""import resource
resource.setrlimit(resource.RLIMIT_CPU, ({self._cpu_seconds}, {self._cpu_seconds}))
resource.setrlimit(resource.RLIMIT_AS, (536870912, 536870912))
resource.setrlimit(resource.RLIMIT_FSIZE, (1048576, 1048576))
resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))
resource.setrlimit(resource.RLIMIT_NPROC, (32, 32))
"""
        return "\n".join((prelude, *imports, code, *tests, "print('PASS')", ""))


class MbppPassAtOneScorer(TaskScorer):
    def __init__(self, sandbox: BubblewrapPythonSandbox | None = None) -> None:
        self._sandbox = sandbox or BubblewrapPythonSandbox()

    def score(self, generated: str, example: DownstreamExample) -> dict[str, Any]:
        code = self._extract_code(generated)
        result = self._sandbox.run(
            code,
            example.metadata["test_imports"],
            example.metadata["test_list"],
        )
        return {
            "score": float(result.passed),
            "sandbox_status": result.status,
            "sandbox_returncode": result.returncode,
            "sandbox_stderr": result.stderr,
        }

    @staticmethod
    def _extract_code(generated: str) -> str:
        fenced = re.search(r"```(?:python)?\s*(.*?)```", generated, flags=re.DOTALL | re.IGNORECASE)
        return (fenced.group(1) if fenced else generated).strip()
