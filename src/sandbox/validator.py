"""
Static validation of LLM-generated code.

This replaces the previous substring blacklist, which was bypassable in
several trivial ways:

    "eval("      -> defeated by `eval ( ... )`      (whitespace)
    "import os"  -> defeated by `__import__("os")`  (different spelling)
    nothing      -> caught `().__class__.__bases__[0].__subclasses__()`

Substring matching operates on text; escapes operate on *syntax*. So we
parse the code and inspect the AST instead, which is insensitive to
formatting and spelling tricks.

This is layer 1 of defence. Layer 2 is process isolation (see executor.py) —
validation alone is never sufficient, because static analysis cannot decide
halting (e.g. `while True: pass`) and cannot see through any dynamic
indirection we failed to anticipate.
"""

from __future__ import annotations

import ast

# ── Imports the generated code may use ────────────────────────
# Deliberately excludes `operator` (operator.attrgetter is a getattr bypass),
# `builtins`, `importlib`, `pickle`, `os`, `sys`, `subprocess`, `socket`.
ALLOWED_IMPORT_ROOTS = frozenset(
    {
        "pandas",
        "numpy",
        "matplotlib",
        "seaborn",
        "plotly",
        "scipy",
        "sklearn",
        "statsmodels",
        "math",
        "statistics",
        "random",
        "re",
        "json",
        "collections",
        "itertools",
        "functools",
        "datetime",
        "decimal",
        "fractions",
        "string",
        "textwrap",
        "warnings",
        "typing",
    }
)

# ── Builtins that can break containment ───────────────────────
# getattr/setattr/delattr are blocked because dynamic attribute access
# defeats static analysis entirely: getattr(o, "__cl" + "ass__") is
# invisible to any AST rule. Legitimate dataframe analysis does not need
# them. This is a deliberate usability/security trade-off.
FORBIDDEN_BUILTINS = frozenset(
    {
        "eval",
        "exec",
        "compile",
        "open",
        "__import__",
        "globals",
        "locals",
        "vars",
        "getattr",
        "setattr",
        "delattr",
        "super",
        "input",
        "breakpoint",
        "exit",
        "quit",
        "help",
        "memoryview",
        "Popen",
        "check_output",
        "check_call",
    }
)

# ── Module objects reachable by ordinary attribute access ──────
# Blocking `import os` is not enough. Third-party packages import the stdlib
# as a side effect, and those module objects hang off public, non-dunder
# attributes of names we *do* inject:
#
#     pd.io.common.os.system("...")   # os, via pandas, no import required
#
# None of `io`, `common`, `os` is a dunder, so the introspection rule below
# never sees it. Cutting the hop itself kills the whole family at once and is
# far more robust than blacklisting every dangerous method these modules
# expose. Names that are plausible dataframe columns (`platform`, `signal`,
# `code`, `types`, `time`) are deliberately left out — `df.platform` is a
# real column accessor, and a false positive here costs a correct answer.
FORBIDDEN_MODULE_ATTRS = frozenset(
    {
        "os", "sys", "subprocess", "shutil", "pathlib", "tempfile", "glob",
        "importlib", "builtins", "ctypes", "pickle", "marshal", "shelve",
        "socket", "urllib", "requests", "http", "ftplib", "smtplib",
        "multiprocessing", "threading", "inspect", "gc", "pty", "runpy",
        "webbrowser", "sqlite3", "site", "sysconfig",
    }
)

# ── Attribute names that touch the filesystem, network, or OS ──
# pandas/numpy are injected into the sandbox namespace, so df.to_csv(...)
# and pd.read_csv(...) would otherwise be a complete file-I/O bypass.
#
# With the module hop closed above, a name only belongs here if its dangerous
# binding is reachable *without* traversing a blocked module. That distinction
# matters: `rename` and `remove` used to sit in this set purely to stop
# `os.rename` / `os.remove`, but they also blocked `DataFrame.rename(columns=…)`
# and `list.remove(…)` — everyday analysis. That false positive burned a repair
# cycle on six of eighteen benchmark tasks. Since `os` itself is now
# unreachable, both names come off and nothing is lost.
#
# Note: to_string/to_markdown/to_dict/to_frame/to_numpy/to_list are NOT
# blocked — they are in-memory and used constantly by real analysis code.
FORBIDDEN_ATTRIBUTES = frozenset(
    {
        # pandas readers
        "read_csv", "read_table", "read_fwf", "read_excel", "read_json",
        "read_html", "read_xml", "read_clipboard", "read_hdf", "read_feather",
        "read_parquet", "read_orc", "read_sas", "read_spss", "read_stata",
        "read_pickle", "read_sql", "read_sql_query", "read_sql_table",
        "read_gbq",
        # pandas writers
        "to_csv", "to_excel", "to_clipboard", "to_hdf", "to_feather",
        "to_parquet", "to_orc", "to_stata", "to_pickle", "to_sql", "to_gbq",
        # numpy I/O
        "load", "loads", "save", "savez", "savez_compressed", "savetxt",
        "loadtxt", "genfromtxt", "fromfile", "tofile", "memmap", "frombuffer",
        # matplotlib / plotly export
        "savefig", "write_image", "write_html", "write_json", "print_figure",
        # process / filesystem
        "system", "popen", "fork", "execv", "execve", "spawn", "spawnv",
        "unlink", "rmdir", "removedirs", "makedirs", "mkdir",
        "chmod", "chown", "listdir", "walk",
        "getcwd", "chdir", "environ", "putenv", "kill", "terminate",
        # `open` as an *attribute*: visit_Name catches the bare builtin, but
        # `io.open` is the same function reached by a different route.
        "open",
        # serialization
        "dump", "dumps",
        # import machinery
        "import_module", "find_module", "load_module", "reload",
        # dynamic attribute access
        "attrgetter", "itemgetter", "methodcaller",
    }
)

MAX_CODE_CHARS = 20_000


class CodeValidationError(ValueError):
    """Raised when generated code violates the sandbox policy."""


def _is_dunder(name: str) -> bool:
    return name.startswith("__") and name.endswith("__")


class _SecurityVisitor(ast.NodeVisitor):
    """Walks the AST and records every policy violation."""

    def __init__(self) -> None:
        self.violations: list[str] = []

    def _flag(self, node: ast.AST, message: str) -> None:
        line = getattr(node, "lineno", "?")
        self.violations.append(f"line {line}: {message}")

    # ── Attribute access ──────────────────────────────────────

    def visit_Attribute(self, node: ast.Attribute) -> None:
        # Blocks the sandbox-escape chain:
        #   ().__class__.__bases__[0].__subclasses__()
        # Every link in that chain is a dunder attribute, so one rule
        # closes the whole family of escapes.
        if _is_dunder(node.attr):
            self._flag(
                node,
                f"access to special attribute '{node.attr}' is not allowed "
                "(introspection can be used to escape the sandbox)",
            )
        elif node.attr in FORBIDDEN_MODULE_ATTRS:
            self._flag(
                node,
                f"reaching the '{node.attr}' module through attribute access "
                "is not allowed",
            )
        elif node.attr in FORBIDDEN_ATTRIBUTES:
            self._flag(
                node,
                f"'{node.attr}' is not allowed — file, network, and OS access "
                "are blocked inside the sandbox",
            )
        self.generic_visit(node)

    # ── Bare names ────────────────────────────────────────────

    def visit_Name(self, node: ast.Name) -> None:
        if _is_dunder(node.id):
            self._flag(node, f"use of '{node.id}' is not allowed")
        elif node.id in FORBIDDEN_BUILTINS:
            self._flag(
                node,
                f"'{node.id}' is not allowed inside the sandbox",
            )
        self.generic_visit(node)

    # ── Imports ───────────────────────────────────────────────

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            root = alias.name.split(".")[0]
            if root not in ALLOWED_IMPORT_ROOTS:
                self._flag(node, f"import of '{alias.name}' is not allowed")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.level and node.level > 0:
            self._flag(node, "relative imports are not allowed")
        elif node.module:
            root = node.module.split(".")[0]
            if root not in ALLOWED_IMPORT_ROOTS:
                self._flag(node, f"import from '{node.module}' is not allowed")
        else:
            self._flag(node, "this import form is not allowed")
        self.generic_visit(node)

    # ── Misc hardening ────────────────────────────────────────

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._flag(node, "async functions are not supported in the sandbox")
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        # Custom classes are unnecessary for analysis and are a common
        # vehicle for metaclass / descriptor tricks.
        self._flag(node, "class definitions are not allowed in the sandbox")
        self.generic_visit(node)


def _find_run_function(tree: ast.Module) -> ast.FunctionDef | None:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "run":
            return node
    return None


def validate_code(code: str) -> ast.Module:
    """
    Validate generated code against the sandbox policy.

    Returns the parsed AST on success.
    Raises CodeValidationError with a message suitable for feeding back to
    the model for self-correction.
    """
    if not code or not code.strip():
        raise CodeValidationError("No code was provided.")

    if len(code) > MAX_CODE_CHARS:
        raise CodeValidationError(
            f"Code is too long ({len(code)} chars, limit {MAX_CODE_CHARS})."
        )

    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise CodeValidationError(
            f"Syntax error on line {exc.lineno}: {exc.msg}"
        ) from exc

    visitor = _SecurityVisitor()
    visitor.visit(tree)

    if visitor.violations:
        detail = "\n  - ".join(visitor.violations)
        raise CodeValidationError(
            f"Code rejected by the sandbox security policy:\n  - {detail}"
        )

    run_fn = _find_run_function(tree)
    if run_fn is None:
        raise CodeValidationError(
            "Generated code must define a top-level function `def run(df):`."
        )
    if not run_fn.args.args:
        raise CodeValidationError(
            "`run()` must accept the dataframe as its first argument: `def run(df):`."
        )

    return tree


def is_safe(code: str) -> bool:
    """Convenience boolean wrapper, mainly for tests."""
    try:
        validate_code(code)
        return True
    except CodeValidationError:
        return False
