"""
Security regression tests for the AST validator.

Every payload in ESCAPES was verified to **bypass** the original substring
blacklist. They are kept here permanently so the containment cannot silently
regress.
"""

from __future__ import annotations

import pytest

from src.sandbox.validator import CodeValidationError, is_safe, validate_code


def wrap(body: str) -> str:
    """Put an expression in a valid `run` function so only policy is tested."""
    return f"def run(df):\n    {body}\n    return (1, None)"


# ── Escapes that must be rejected ─────────────────────────────

ESCAPES = {
    # S1 — introspection chain. On the old sandbox this reached 168 classes,
    # including subprocess.Popen and the file type.
    "dunder_class_traversal": "x = ().__class__.__bases__[0].__subclasses__()",
    "dunder_globals": "x = run.__globals__",
    "dunder_builtins": "x = [].__class__.__base__.__subclasses__()[0].__init__.__globals__",
    "dunder_mro": "x = type(df).__mro__",
    "dunder_dict": "x = df.__dict__",
    "dunder_import_call": 'x = __import__("os").getcwd()',
    # S2 — file and network I/O via the pre-injected pandas/numpy objects.
    "pandas_read": 'x = pd.read_csv("C:/Windows/win.ini")',
    "pandas_write": 'df.to_csv("/tmp/stolen.csv")',
    "pandas_pickle": 'df.to_pickle("/tmp/x.pkl")',
    "numpy_load": 'x = np.load("/etc/passwd")',
    "matplotlib_savefig": 'fig.savefig("/tmp/out.png")',
    "read_sql": 'x = pd.read_sql("SELECT 1", conn)',
    # S3 — whitespace defeated the old `"eval("` substring check.
    "eval_spaced": "x = eval ( '2+2' )",
    "eval_plain": "x = eval('2+2')",
    "exec_call": "exec('import os')",
    "compile_call": "x = compile('1', '<s>', 'eval')",
    "open_call": "f = open('/etc/passwd')",
    # Dynamic attribute access defeats static analysis, so it is blocked.
    "getattr_split": "x = getattr(df, '__cl' + 'ass__')",
    "setattr_call": "setattr(df, 'x', 1)",
    "attrgetter": "x = operator.attrgetter('__class__')(df)",
    # Imports outside the whitelist (inside run(), where the model puts them).
    "import_os": "import os",
    "import_subprocess": "import subprocess",
    "from_os_import": "from os import system",
    "import_builtins": "import builtins",
    "import_importlib": "import importlib",
    "globals_call": "x = globals()",
    "vars_call": "x = vars(df)",
    # S6 — module hop. `pandas.io.common` imports `os` and `io` as a side
    # effect, so the real modules hang off public, non-dunder attributes of
    # the `pd` we inject. Verified live: `pd.io.common.os` is the os module.
    # No import statement and no dunder is involved, so only the module-attr
    # rule catches these.
    "module_hop_os_system": 'pd.io.common.os.system("calc.exe")',
    "module_hop_os_remove": 'pd.io.common.os.remove("/etc/passwd")',
    "module_hop_os_rename": 'pd.io.common.os.rename("a", "b")',
    "module_hop_io_open": 'f = pd.io.common.io.open("/etc/passwd")',
    "module_hop_environ": "x = pd.io.common.os.environ",
    "module_hop_sys": "x = np.sys.modules",
}

# Escapes that only make sense at module level, so they are written in full.
MODULE_LEVEL_ESCAPES = {
    "relative_import": "from . import something\n\ndef run(df):\n    return (1, None)",
    "class_def": "class Evil:\n    pass\n\ndef run(df):\n    return (1, None)",
    "toplevel_import_os": "import os\n\ndef run(df):\n    return (1, None)",
}


@pytest.mark.parametrize("name,body", sorted(ESCAPES.items()))
def test_escape_is_rejected(name: str, body: str) -> None:
    assert not is_safe(wrap(body)), f"{name} was NOT blocked — sandbox escape is reachable"


@pytest.mark.parametrize("name,code", sorted(MODULE_LEVEL_ESCAPES.items()))
def test_module_level_escape_is_rejected(name: str, code: str) -> None:
    assert not is_safe(code), f"{name} was NOT blocked — sandbox escape is reachable"


def test_rejection_message_is_actionable() -> None:
    """The message is fed back to the model for self-correction."""
    with pytest.raises(CodeValidationError) as exc:
        validate_code(wrap("x = ().__class__"))
    message = str(exc.value)
    assert "__class__" in message
    assert "line" in message


# ── Legitimate analysis must not be blocked ───────────────────

LEGITIMATE = {
    "groupby": """
def run(df):
    out = df.groupby('Gender')['Fare'].agg(['mean', 'count']).round(2).reset_index()
    return (out, None)
""",
    "matplotlib_chart": """
def run(df):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df['Age'].dropna(), bins=20)
    ax.set_title('Age')
    return ('ok', fig)
""",
    "plotly_chart": """
def run(df):
    fig = px.scatter(df.dropna(subset=['Age', 'Fare']), x='Age', y='Fare')
    return ('ok', fig)
""",
    "scipy_stats": """
def run(df):
    sub = df.dropna(subset=['Age', 'Fare'])
    r, p = scipy_stats.pearsonr(sub['Age'], sub['Fare'])
    return (f'r={r:.3f}', None)
""",
    "sklearn_pipeline": """
def run(df):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    sub = df[['Age', 'Fare', 'Survived']].dropna()
    X_train, X_test, y_train, y_test = train_test_split(
        sub[['Age', 'Fare']], sub['Survived'], test_size=0.2, random_state=42
    )
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    return (accuracy_score(y_test, model.predict(X_test)), None)
""",
    # These `to_*` / `load*` names are in-memory and must stay allowed —
    # blocking them by prefix would break ordinary analysis.
    "in_memory_converters": """
def run(df):
    a = df.head().to_string()
    b = df.describe().to_dict()
    c = df['Age'].to_numpy()
    d = df.to_markdown()
    e = df['Age'].tolist()
    return (a, None)
""",
    "stdlib_imports": """
def run(df):
    import math, json, re
    from collections import Counter
    from datetime import datetime
    return (math.sqrt(16), None)
""",
    "comprehensions_and_lambdas": """
def run(df):
    cols = {c: str(df[c].dtype) for c in df.columns}
    vals = sorted(df['Age'].dropna().unique(), key=lambda v: -v)[:5]
    return (cols, None)
""",
    "nested_helper": """
def run(df):
    def fmt(v):
        return f'{v:.2f}'
    return (fmt(df['Fare'].mean()), None)
""",
    # `rename` and `remove` were blocked to stop `os.rename` / `os.remove`,
    # which also took out ordinary pandas and list operations. Six of the
    # eighteen benchmark tasks hit this and wasted a repair round-trip on it.
    # The module-attr rule blocks the `os` hop directly, so these are free.
    "dataframe_rename": """
def run(df):
    out = df.rename(columns={'Fare': 'ticket_price'})
    return (out.head(), None)
""",
    "list_remove": """
def run(df):
    cols = list(df.columns)
    if 'Name' in cols:
        cols.remove('Name')
    return (cols, None)
""",
    "series_rename": """
def run(df):
    s = df['Age'].rename('age_years')
    return (s.describe(), None)
""",
    # `df.platform` is attribute-style column access — the module-attr list
    # must not grow to include names that read like real columns.
    "attribute_style_column": """
def run(df):
    return (df.Age.mean(), None)
""",
}


@pytest.mark.parametrize("name,code", sorted(LEGITIMATE.items()))
def test_legitimate_code_passes(name: str, code: str) -> None:
    assert is_safe(code), f"false positive: {name} was wrongly blocked"


# ── Structural requirements ───────────────────────────────────


def test_missing_run_function_is_rejected() -> None:
    with pytest.raises(CodeValidationError, match="run"):
        validate_code("x = df['Age'].mean()")


def test_run_without_argument_is_rejected() -> None:
    with pytest.raises(CodeValidationError, match="first argument"):
        validate_code("def run():\n    return (1, None)")


def test_syntax_error_is_reported_with_line() -> None:
    with pytest.raises(CodeValidationError, match="Syntax error"):
        validate_code("def run(df):\n    return (1, None")


def test_empty_code_is_rejected() -> None:
    with pytest.raises(CodeValidationError):
        validate_code("   ")


def test_oversized_code_is_rejected() -> None:
    with pytest.raises(CodeValidationError, match="too long"):
        validate_code("def run(df):\n    x = 1\n" + "    # pad\n" * 8000)


def test_validate_returns_ast_on_success() -> None:
    import ast

    tree = validate_code("def run(df):\n    return (df.shape, None)")
    assert isinstance(tree, ast.Module)
