"""AST mutation engine for real library source code (Cosmic Ray / MutPy style operators).

The engine parses the *actual* source file of a module, enumerates single-point
mutations inside a chosen set of functions, and can rebuild a private copy of
the module with exactly one mutation applied.  Nothing is hand-written: the
mutant population is determined by the operator catalogue and the source code.
"""
from __future__ import annotations

import ast
import copy
import importlib
import re
import signal
import sys
import types
from contextlib import contextmanager
from dataclasses import dataclass, asdict

# ----------------------------------------------------------------------------
# Operator catalogue
# ----------------------------------------------------------------------------
CMP_SWAPS = {
    ast.Lt: [(ast.LtE, "boundary"), (ast.Gt, "relational")],
    ast.LtE: [(ast.Lt, "boundary"), (ast.GtE, "relational")],
    ast.Gt: [(ast.GtE, "boundary"), (ast.Lt, "relational")],
    ast.GtE: [(ast.Gt, "boundary"), (ast.LtE, "relational")],
    ast.Eq: [(ast.NotEq, "relational")],
    ast.NotEq: [(ast.Eq, "relational")],
    ast.Is: [(ast.IsNot, "relational")],
    ast.IsNot: [(ast.Is, "relational")],
    ast.In: [(ast.NotIn, "relational")],
    ast.NotIn: [(ast.In, "relational")],
}
BIN_SWAPS = {
    ast.Add: [ast.Sub], ast.Sub: [ast.Add], ast.Mult: [ast.Div], ast.Div: [ast.Mult],
    ast.FloorDiv: [ast.Div], ast.Pow: [ast.Mult], ast.Mod: [ast.Mult],
}
# call-name swaps: name -> [(new_name, category)]
CALL_SWAPS = {
    "mean": [("median", "reduction")], "median": [("mean", "reduction")],
    "nanmean": [("nanmedian", "reduction"), ("mean", "nan")],
    "nanmedian": [("nanmean", "reduction"), ("median", "nan")],
    "nanmin": [("nanmax", "reduction"), ("min", "nan")], "nanmax": [("nanmin", "reduction"), ("max", "nan")],
    "nansum": [("sum", "nan")], "nanstd": [("std", "nan")],
    "min": [("max", "reduction")], "max": [("min", "reduction")],
    "argmin": [("argmax", "reduction")], "argmax": [("argmin", "reduction")],
    "sum": [("mean", "reduction")],
    "isnan": [("isinf", "nan")], "isfinite": [("isnan", "nan")],
    "all": [("any", "logical")], "any": [("all", "logical")],
    "sqrt": [("square", "arithmetic")],
}
STR_SWAPS = {
    "mean": "median", "median": "mean", "most_frequent": "mean",
    "uniform": "distance", "distance": "uniform",
    "ascending": "descending", "descending": "ascending",
    "allow-nan": True,
}
KW_CATEGORY = [
    (r"dtype", "dtype"), (r"kind|order|stable", "tie"), (r"^axis$", "axis"),
    (r"copy|writeable", "copy"), (r"finite|sparse|min_samples|ensure", "validation"),
    (r"tol|eps", "tolerance"), (r"weight", "semantic"),
]
TOL_KW = re.compile(r"tol|eps|thresh|min_value|max_value|alpha|lambda", re.I)

CATEGORIES = [
    "arithmetic", "relational", "boundary", "logical", "branch", "constant", "tolerance",
    "semantic", "reduction", "nan", "dtype", "tie", "axis", "copy", "validation", "kwarg", "statement",
]


def _name_of_call(node: ast.Call):
    f = node.func
    if isinstance(f, ast.Attribute):
        return f.attr
    if isinstance(f, ast.Name):
        return f.id
    return None


def _np_attr(name: str):
    return ast.Attribute(value=ast.Name(id="np", ctx=ast.Load()), attr=name, ctx=ast.Load())


def _build_parents(tree):
    parents = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    return parents


def _tolerance_context(node, parents):
    p = parents.get(node)
    if isinstance(p, ast.keyword) and p.arg and TOL_KW.search(p.arg):
        return True
    if isinstance(p, (ast.Compare, ast.BinOp)):
        for sib in ast.iter_child_nodes(p):
            if isinstance(sib, ast.Name) and TOL_KW.search(sib.id):
                return True
            if isinstance(sib, ast.Attribute) and TOL_KW.search(sib.attr):
                return True
    return False


def generate(node, parents):
    """All single-point mutants of `node`: list of (operator, category, new_node)."""
    out = []
    if isinstance(node, ast.Compare):
        for i, op in enumerate(node.ops):
            for new_cls, cat in CMP_SWAPS.get(type(op), []):
                new = copy.copy(node)
                new.ops = list(node.ops)
                new.ops[i] = new_cls()
                out.append((f"Cmp[{type(op).__name__}->{new_cls.__name__}]", cat, new))
    elif isinstance(node, (ast.BinOp, ast.AugAssign)):
        for new_cls in BIN_SWAPS.get(type(node.op), []):
            new = copy.copy(node)
            new.op = new_cls()
            out.append((f"Bin[{type(node.op).__name__}->{new_cls.__name__}]", "arithmetic", new))
    elif isinstance(node, ast.UnaryOp):
        if isinstance(node.op, ast.USub):
            out.append(("Unary[remove-]", "arithmetic", node.operand))
        elif isinstance(node.op, ast.Not):
            out.append(("Unary[remove-not]", "logical", node.operand))
        elif isinstance(node.op, ast.Invert):
            out.append(("Unary[remove~]", "logical", node.operand))
    elif isinstance(node, ast.BoolOp):
        new = copy.copy(node)
        new.op = ast.Or() if isinstance(node.op, ast.And) else ast.And()
        out.append((f"Bool[{type(node.op).__name__}->{type(new.op).__name__}]", "logical", new))
    elif isinstance(node, ast.Constant):
        v = node.value
        if isinstance(v, bool):
            out.append(("Const[bool-flip]", "logical", ast.Constant(value=not v)))
        elif isinstance(v, int):
            out.append(("Const[int+1]", "constant", ast.Constant(value=v + 1)))
            out.append(("Const[int-1]", "constant", ast.Constant(value=v - 1)))
        elif isinstance(v, float):
            cat = "tolerance" if (abs(v) < 0.1 or _tolerance_context(node, parents)) else "constant"
            if v == 0.0:
                out.append(("Const[float0->1]", cat, ast.Constant(value=1.0)))
            else:
                out.append(("Const[float*10]", cat, ast.Constant(value=v * 10.0)))
                out.append(("Const[float/10]", cat, ast.Constant(value=v / 10.0)))
        elif isinstance(v, str) and v in STR_SWAPS:
            repl = STR_SWAPS[v]
            cat = "validation" if v == "allow-nan" else "semantic"
            out.append((f"Str[{v}->{repl}]", cat, ast.Constant(value=repl)))
    elif isinstance(node, ast.Name) and node.id == "FLOAT_DTYPES":
        out.append(("Dtype[FLOAT_DTYPES->float32-only]", "dtype",
                    ast.Tuple(elts=[_np_attr("float32")], ctx=ast.Load())))
    elif isinstance(node, ast.Attribute) and node.attr == "float64" and isinstance(node.value, ast.Name):
        new = copy.copy(node)
        new.attr = "float32"
        out.append(("Dtype[float64->float32]", "dtype", new))
    elif isinstance(node, ast.Call):
        name = _name_of_call(node)
        for new_name, cat in CALL_SWAPS.get(name, []):
            new = copy.copy(node)
            new.func = copy.copy(node.func)
            if isinstance(new.func, ast.Attribute):
                new.func.attr = new_name
            else:
                new.func.id = new_name
            out.append((f"Call[{name}->{new_name}]", cat, new))
        for i, kw in enumerate(node.keywords):
            if kw.arg is None:
                continue
            cat = "kwarg"
            for pat, c in KW_CATEGORY:
                if re.search(pat, kw.arg):
                    cat = c
                    break
            new = copy.copy(node)
            new.keywords = [k for j, k in enumerate(node.keywords) if j != i]
            out.append((f"Kwarg[drop {kw.arg}]", cat, new))
    elif isinstance(node, (ast.If, ast.While, ast.IfExp)):
        new = copy.copy(node)
        new.test = ast.UnaryOp(op=ast.Not(), operand=node.test)
        out.append((f"Branch[negate {type(node).__name__}]", "branch", new))
    elif isinstance(node, (ast.Assign, ast.AugAssign, ast.Expr)):
        is_docstring = isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
        if not is_docstring:
            try:
                text = ast.unparse(node)
            except Exception:  # pragma: no cover
                text = ""
            cat = "validation" if re.search(r"valid|check|ensure", text) else "statement"
            out.append(("Stmt[delete]", cat, ast.Pass()))
    return out


# ----------------------------------------------------------------------------
# Mutant enumeration
# ----------------------------------------------------------------------------
@dataclass
class Mutant:
    mutant_id: str
    target: str
    module_name: str
    func: str
    node_index: int
    variant: int
    operator: str
    category: str
    lineno: int
    end_lineno: int
    stmt_start: int
    stmt_end: int
    original: str
    replacement: str
    stmt_text: str = ""

    def asdict(self):
        return asdict(self)


def _find_functions(tree, qualnames, strict=True):
    found = {}
    wanted = set(qualnames)

    def visit(body, prefix):
        for n in body:
            if isinstance(n, ast.ClassDef):
                visit(n.body, prefix + n.name + ".")
            elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                q = prefix + n.name
                if q in wanted:
                    found[q] = n
    visit(tree.body, "")
    missing = wanted - set(found)
    if missing:
        if strict:
            raise KeyError(f"functions not found in module: {sorted(missing)}")
        print(f"[mutate] WARNING: functions not found in this library version and skipped: {sorted(missing)}")
    return found


def _skip_subtrees(fnode):
    """Nodes that are not worth mutating: error messages, docstrings, f-strings, warnings."""
    skip = set()
    body = fnode.body
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        for n in ast.walk(body[0]):
            skip.add(n)
    for n in ast.walk(fnode):
        if isinstance(n, (ast.Raise, ast.JoinedStr)):
            for m in ast.walk(n):
                skip.add(m)
        elif isinstance(n, ast.Expr) and isinstance(n.value, ast.Call):
            nm = ast.unparse(n.value.func)
            if nm.endswith("warn") or nm.endswith("warnings.warn"):
                for m in ast.walk(n):
                    skip.add(m)
    return skip


def _clip(s, k=110):
    s = " ".join(s.split())
    return s if len(s) <= k else s[: k - 3] + "..."


def _stmt_text(st):
    try:
        if isinstance(st, (ast.If, ast.While)):
            return _clip(("if " if isinstance(st, ast.If) else "while ") + ast.unparse(st.test), 200)
        if isinstance(st, (ast.For, ast.AsyncFor)):
            return _clip(f"for {ast.unparse(st.target)} in {ast.unparse(st.iter)}", 200)
        if isinstance(st, (ast.With, ast.AsyncWith)):
            return _clip("with " + ", ".join(ast.unparse(i) for i in st.items), 200)
        if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Try)):
            return type(st).__name__
        return _clip(ast.unparse(st), 200)
    except Exception:  # pragma: no cover
        return ""


def enumerate_mutants(source: str, module_name: str, target: str, funcs, strict=False):
    tree = ast.parse(source)
    parents = _build_parents(tree)
    nodes = list(ast.walk(tree))
    index = {n: i for i, n in enumerate(nodes)}
    fnodes = _find_functions(tree, funcs, strict=strict)
    mutants, seen = [], set()
    for qual, fnode in fnodes.items():
        skip = _skip_subtrees(fnode)
        for stmt in fnode.body:
            for node in ast.walk(stmt):
                if node in skip or not hasattr(node, "lineno"):
                    continue
                cands = generate(node, parents)
                if not cands:
                    continue
                st = node
                while st is not None and not isinstance(st, ast.stmt):
                    st = parents.get(st)
                s0 = getattr(st, "lineno", node.lineno)
                s1 = getattr(st, "end_lineno", node.lineno)
                orig = _clip(ast.unparse(node))
                stext = _stmt_text(st) if st is not None else ""
                for v, (op, cat, new) in enumerate(cands):
                    try:
                        repl = _clip(ast.unparse(new))
                    except Exception:
                        continue
                    key = (index[node], repl)
                    if key in seen or repl == orig:
                        continue
                    seen.add(key)
                    mutants.append(Mutant(
                        mutant_id=f"{target}#{len(mutants):04d}", target=target, module_name=module_name,
                        func=qual, node_index=index[node], variant=v, operator=op, category=cat,
                        lineno=node.lineno, end_lineno=getattr(node, "end_lineno", node.lineno),
                        stmt_start=s0, stmt_end=s1, original=orig, replacement=repl, stmt_text=stext))
    return mutants


def build_mutant_code(source: str, mutant: Mutant, filename: str):
    tree = ast.parse(source)
    parents = _build_parents(tree)
    nodes = list(ast.walk(tree))
    node = nodes[mutant.node_index]
    op, _cat, new = generate(node, parents)[mutant.variant]
    assert op == mutant.operator, (op, mutant.operator)
    ast.copy_location(new, node)
    parent = parents[node]
    for field, value in ast.iter_fields(parent):
        if value is node:
            setattr(parent, field, new)
            break
        if isinstance(value, list):
            for i, item in enumerate(value):
                if item is node:
                    value[i] = new
                    break
            else:
                continue
            break
    ast.fix_missing_locations(tree)
    return compile(tree, filename, "exec", dont_inherit=True)


def load_module_copy(module_name: str, code, file_path: str):
    """Execute `code` as a private copy of module `module_name` (relative imports keep working)."""
    mod = types.ModuleType(module_name)
    mod.__dict__["__file__"] = file_path
    mod.__dict__["__package__"] = module_name.rpartition(".")[0]
    exec(code, mod.__dict__)
    return mod


def module_source(module_name: str):
    mod = importlib.import_module(module_name)
    with open(mod.__file__, "r", encoding="utf-8") as fh:
        return fh.read(), mod.__file__


class MutantTimeout(Exception):
    pass


@contextmanager
def time_limit(seconds: float):
    """Best-effort wall-clock guard (POSIX main thread only; a no-op elsewhere)."""
    if not hasattr(signal, "setitimer") or seconds is None:
        yield
        return

    def handler(signum, frame):
        raise MutantTimeout(f"timeout after {seconds}s")

    try:
        old = signal.signal(signal.SIGALRM, handler)
    except ValueError:  # not in main thread
        yield
        return
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)
