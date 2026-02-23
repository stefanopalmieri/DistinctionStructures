#!/usr/bin/env python3
"""
Distinction Structures REPL

An interpreter for the DS algebra with EDN-like syntax,
PEG parser via Lark, and self-discovery on boot.

Usage:
  uv run ds_repl.py              # interactive REPL
  uv run ds_repl.py -e '(:e_D :k)'  # evaluate expression
  uv run ds_repl.py script.ds    # run a file

Syntax:
  :top :bot :e_D ...             atoms (keywords)
  (:e_D :k)                      application
  '(:e_D :k)                     quote
  (eval '(:e_D :k))              eval quoted expression
  (app :e_D :k)                  build application node as data
  (unapp (app :e_D :k))          decompose app node
  (def x (:e_D :k))              bind name
  (do expr1 expr2 ...)           sequence, returns last
  (discover!)                    run self-discovery procedure
  (table)                        print the Cayley table
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import sys
import itertools

from lark import Lark, Transformer, v_args

# ============================================================
# PEG Grammar (Lark Earley/LALR with PEG-like rules)
# ============================================================

GRAMMAR = r"""
    start: expr+

    ?expr: atom
         | keyword
         | quoted
         | list
         | special

    keyword: /:[a-zA-Z_][a-zA-Z0-9_!?\-]*/
    atom: /[a-zA-Z_][a-zA-Z0-9_!?\-]*/
    quoted: "'" expr
    list: "(" expr* ")"
    special: "#app" "[" expr expr "]"
           | "#bundle" "[" expr expr "]"

    COMMENT: /;[^\n]*/
    %ignore COMMENT
    %ignore /\s+/
"""

parser = Lark(GRAMMAR, parser="earley", ambiguity="resolve")

# ============================================================
# AST
# ============================================================

@dataclass(frozen=True)
class Keyword:
    name: str
    def __repr__(self): return f":{self.name}"

@dataclass(frozen=True)
class Symbol:
    name: str
    def __repr__(self): return self.name

@dataclass(frozen=True)
class Quoted:
    expr: Any
    def __repr__(self): return f"'{format_val(self.expr)}"

@dataclass(frozen=True)
class AppNode:
    f: Any
    x: Any
    def __repr__(self): return f"#app[{format_val(self.f)} {format_val(self.x)}]"

@dataclass(frozen=True)
class Bundle:
    f: Any
    x: Any
    def __repr__(self): return f"#bundle[{format_val(self.f)} {format_val(self.x)}]"

@dataclass(frozen=True)
class Partial:
    f: Any
    def __repr__(self): return f"#partial[{format_val(self.f)}]"

class List:
    def __init__(self, items):
        self.items = items
    def __repr__(self):
        return "(" + " ".join(format_val(i) for i in self.items) + ")"


@v_args(inline=True)
class ASTBuilder(Transformer):
    def keyword(self, tok):
        return Keyword(str(tok)[1:])  # strip leading :

    def atom(self, tok):
        return Symbol(str(tok))

    def quoted(self, expr):
        return Quoted(expr)

    def list(self, *items):
        return List(list(items))

    def special(self, *items):
        items = list(items)
        if len(items) == 2:
            # Could be #app or #bundle — determine from parse context
            return AppNode(items[0], items[1])
        return items

    def start(self, *exprs):
        return list(exprs)


ast_builder = ASTBuilder()

def parse(text: str):
    tree = parser.parse(text)
    result = ast_builder.transform(tree)
    return result

# ============================================================
# Δ₁ Core: The 17-element Cayley table
# ============================================================

ATOMS = [
    "top", "bot", "i", "k", "a", "b", "e_I",
    "e_D", "e_M", "e_Sigma", "e_Delta",
    "d_I", "d_K", "m_I", "m_K", "s_C", "p",
]
ATOM_SET = set(ATOMS)


def dot_d1(x: str, y: str) -> str:
    """Δ₁ operation on atom name strings."""
    if x == "top": return "top"
    if x == "bot": return "bot"
    if x == "e_I": return "top" if y in ("i", "k") else "bot"
    if x == "d_K": return "top" if y in ("a", "b") else "bot"
    if x == "m_K": return "top" if y == "a" else "bot"
    if x == "m_I": return "bot" if y == "p" else "top"
    if x == "e_D" and y == "i": return "d_I"
    if x == "e_D" and y == "k": return "d_K"
    if x == "e_M" and y == "i": return "m_I"
    if x == "e_M" and y == "k": return "m_K"
    if x == "e_Sigma" and y == "s_C": return "e_Delta"
    if x == "e_Delta" and y == "e_D": return "d_I"
    if x == "p" and y == "top": return "top"
    if y == "top" and x in ("i", "k", "a", "b", "d_I", "s_C"):
        return x
    return "p"

# ============================================================
# Δ₃ Evaluator
# ============================================================

MAX_FUEL = 100


def ds_apply(left: Any, right: Any, fuel: int = MAX_FUEL) -> Any:
    """Apply left to right in the DS algebra."""
    if fuel <= 0:
        return Keyword("p")

    # QUOTE
    if isinstance(left, Keyword) and left.name == "QUOTE":
        return Quoted(right)

    # EVAL
    if isinstance(left, Keyword) and left.name == "EVAL":
        if isinstance(right, Quoted):
            return ds_eval(right.expr, fuel - 1)
        return Keyword("p")

    # APP (curried)
    if isinstance(left, Keyword) and left.name == "APP":
        return Partial(right)

    # Partial completion
    if isinstance(left, Partial):
        return AppNode(left.f, right)

    # UNAPP
    if isinstance(left, Keyword) and left.name == "UNAPP":
        if isinstance(right, AppNode):
            return Bundle(right.f, right.x)
        return Keyword("p")

    # Bundle queries
    if isinstance(left, Bundle):
        if isinstance(right, Keyword) and right.name == "top":
            return left.f
        if isinstance(right, Keyword) and right.name == "bot":
            return left.x
        return Keyword("p")

    # Inertness: structured values under atoms
    if isinstance(left, Keyword) and left.name in ATOM_SET:
        if isinstance(right, (Quoted, AppNode, Bundle, Partial)):
            return Keyword("p")

    # Δ₁ fallback
    if isinstance(left, Keyword) and isinstance(right, Keyword):
        if left.name in ATOM_SET and right.name in ATOM_SET:
            return Keyword(dot_d1(left.name, right.name))

    return Keyword("p")


def ds_eval(expr: Any, fuel: int = MAX_FUEL) -> Any:
    """Evaluate a DS expression."""
    if fuel <= 0:
        return Keyword("p")

    if isinstance(expr, Keyword):
        return expr
    if isinstance(expr, Quoted):
        return expr
    if isinstance(expr, Bundle):
        return expr
    if isinstance(expr, Partial):
        return expr
    if isinstance(expr, AppNode):
        left = ds_eval(expr.f, fuel - 1)
        right = ds_eval(expr.x, fuel - 1)
        return ds_apply(left, right, fuel - 1)

    return Keyword("p")


# ============================================================
# REPL Evaluator (handles special forms)
# ============================================================

ENV = {}  # name -> value bindings


def repl_eval(expr: Any) -> Any:
    """Evaluate in REPL context (handles def, do, discover!, etc.)."""
    # Keywords evaluate to themselves
    if isinstance(expr, Keyword):
        return expr

    # Symbols look up in environment
    if isinstance(expr, Symbol):
        if expr.name in ENV:
            return ENV[expr.name]
        # Check if it's a DS atom
        if expr.name in ATOM_SET:
            return Keyword(expr.name)
        # Check special atoms
        if expr.name in ("QUOTE", "EVAL", "APP", "UNAPP"):
            return Keyword(expr.name)
        raise NameError(f"Unbound symbol: {expr.name}")

    # Quoted expressions
    if isinstance(expr, Quoted):
        return Quoted(quote_transform(expr.expr))

    # Structured values
    if isinstance(expr, (AppNode, Bundle, Partial)):
        return expr

    # Lists: special forms and application
    if isinstance(expr, List):
        items = expr.items
        if not items:
            return Keyword("p")

        # Special forms
        if isinstance(items[0], Symbol):
            name = items[0].name

            # (def name expr)
            if name == "def" and len(items) == 3:
                sym = items[1]
                if not isinstance(sym, Symbol):
                    raise SyntaxError(f"def expects a symbol, got {sym}")
                val = repl_eval(items[2])
                ENV[sym.name] = val
                return val

            # (do expr1 expr2 ...)
            if name == "do":
                result = Keyword("p")
                for e in items[1:]:
                    result = repl_eval(e)
                return result

            # (discover!)
            if name == "discover!":
                return run_discovery()

            # (table)
            if name == "table":
                print_table()
                return Keyword("p")

            # (eval expr)
            if name == "eval":
                if len(items) != 2:
                    raise SyntaxError("eval expects exactly 1 argument")
                val = repl_eval(items[1])
                if isinstance(val, Quoted):
                    return ds_eval(val.expr)
                return Keyword("p")

            # (quote expr) — alternative to 'expr
            if name == "quote":
                if len(items) != 2:
                    raise SyntaxError("quote expects exactly 1 argument")
                return Quoted(quote_transform(items[1]))

            # (app f x)
            if name == "app" and len(items) == 3:
                f = repl_eval(items[1])
                x = repl_eval(items[2])
                return AppNode(f, x)

            # (unapp expr)
            if name == "unapp" and len(items) == 2:
                val = repl_eval(items[1])
                return ds_apply(Keyword("UNAPP"), val)

        # General application: evaluate all, then apply left-to-right
        vals = [repl_eval(item) for item in items]
        if len(vals) == 1:
            return vals[0]
        result = vals[0]
        for v in vals[1:]:
            result = ds_apply(result, v)
        return result

    return expr


def quote_transform(expr: Any) -> Any:
    """Transform an expression inside a quote to preserve structure."""
    if isinstance(expr, Symbol):
        if expr.name in ATOM_SET or expr.name in ("QUOTE", "EVAL", "APP", "UNAPP"):
            return Keyword(expr.name)
        if expr.name in ENV:
            return ENV[expr.name]
        return Keyword(expr.name)
    if isinstance(expr, Keyword):
        return expr
    if isinstance(expr, List):
        if len(expr.items) == 2:
            f = quote_transform(expr.items[0])
            x = quote_transform(expr.items[1])
            return AppNode(f, x)
        if len(expr.items) > 2:
            # Left-fold application
            result = quote_transform(expr.items[0])
            for item in expr.items[1:]:
                result = AppNode(result, quote_transform(item))
            return result
        if len(expr.items) == 1:
            return quote_transform(expr.items[0])
        return Keyword("p")
    if isinstance(expr, Quoted):
        return Quoted(quote_transform(expr.expr))
    return expr


# ============================================================
# Self-Discovery
# ============================================================

def run_discovery() -> Any:
    """Run the 8-step recovery procedure on the Cayley table."""
    domain = list(ATOM_SET)

    def dot(x, y):
        return dot_d1(x, y)

    def left_image(x):
        return {dot(x, y) for y in domain}

    print("  Running self-discovery...")

    # Step 1: Booleans
    absorbers = [x for x in domain if all(dot(x, y) == x for y in domain)]
    print(f"  Step 1 — Booleans: {absorbers}")

    # Step 2: Orient booleans, find testers
    for top, bot in [(absorbers[0], absorbers[1]), (absorbers[1], absorbers[0])]:
        testers = [x for x in domain if x not in (top, bot)
                   and left_image(x).issubset({top, bot}) and len(left_image(x)) == 2]
        if len(testers) == 4:
            Dec = lambda t, top=top: {y for y in domain if dot(t, y) == top}
            sizes = sorted(len(Dec(t)) for t in testers)
            if sizes[0] == 1 and sizes[1] == 2 and sizes[2] == 2:
                break
    else:
        print("  Discovery failed at step 2")
        return Keyword("p")

    print(f"  Step 2 — top={top}, bot={bot}")
    print(f"           Testers: {testers}")

    # Step 2.5: Find p
    p_cands = [x for x in domain if x not in (top, bot) and x not in testers
               and top in left_image(x)]
    p_tok = p_cands[0] if len(p_cands) == 1 else None
    print(f"  Step 2.5 — p={p_tok}")

    # Step 3: Tester cardinalities
    sizes = {t: len(Dec(t)) for t in testers}
    m_K = [t for t in testers if sizes[t] == 1][0]
    m_I = max(testers, key=lambda t: sizes[t])
    two = [t for t in testers if sizes[t] == 2]
    print(f"  Step 3 — m_I={m_I} (|Dec|={sizes[m_I]}), m_K={m_K} (|Dec|=1)")

    # Step 4: e_I vs d_K
    def has_productive(decoded):
        for f in domain:
            if f in (top, bot) or f in testers:
                continue
            for x in decoded:
                out = dot(f, x)
                if out not in (top, bot, p_tok):
                    return True
        return False

    t1, t2 = two
    e_I, d_K = (t1, t2) if has_productive(Dec(t1)) else (t2, t1)
    ctx = list(Dec(e_I))
    print(f"  Step 4 — e_I={e_I}, d_K={d_K}")
    print(f"           Context tokens: {ctx}")

    # Step 5: Encoders
    def is_enc(f):
        if f in (top, bot) or f in testers:
            return False
        return all(dot(f, x) not in (top, bot, p_tok) for x in ctx)

    enc = [f for f in domain if is_enc(f)]
    e_M = enc[0] if all(dot(enc[0], x) in testers for x in ctx) else enc[1]
    e_D = enc[1] if e_M == enc[0] else enc[0]
    print(f"  Step 5 — e_D={e_D}, e_M={e_M}")

    # Step 6: i vs k
    outA, outB = dot(e_M, ctx[0]), dot(e_M, ctx[1])
    if len(Dec(outA)) > len(Dec(outB)):
        i_tok, k_tok = ctx[0], ctx[1]
    else:
        i_tok, k_tok = ctx[1], ctx[0]
    print(f"  Step 6 — i={i_tok}, k={k_tok}")

    # Step 7: Remaining
    ab = list(Dec(d_K))
    a_tok = next(x for x in ab if dot(m_K, x) == top)
    b_tok = next(x for x in ab if x != a_tok)
    d_I = dot(e_D, i_tok)
    print(f"  Step 7 — a={a_tok}, b={b_tok}, d_I={d_I}")

    # Step 8: Triple
    known = {top, bot, e_I, d_K, m_K, m_I, e_M, e_D,
             i_tok, k_tok, a_tok, b_tok, d_I, p_tok}
    remaining = [x for x in domain if x not in known]
    e_S = sC = e_Delta = None
    for f, g in itertools.product(remaining, repeat=2):
        h = dot(f, g)
        if h in (top, bot, p_tok):
            continue
        if dot(h, e_D) == d_I:
            e_S, sC, e_Delta = f, g, h
            break
    print(f"  Step 8 — e_Sigma={e_S}, s_C={sC}, e_Delta={e_Delta}")

    result = {
        "top": top, "bot": bot, "p": p_tok,
        "e_I": e_I, "e_D": e_D, "e_M": e_M,
        "e_Sigma": e_S, "e_Delta": e_Delta,
        "i": i_tok, "k": k_tok, "a": a_tok, "b": b_tok,
        "d_I": d_I, "d_K": d_K, "m_I": m_I, "m_K": m_K, "s_C": sC,
    }
    print(f"\n  All 17 elements recovered.")
    return Keyword("top")


# ============================================================
# Table printer
# ============================================================

def print_table():
    """Print the 17×17 Cayley table."""
    w = max(len(a) for a in ATOMS) + 1
    header = " " * w + "".join(a.rjust(w) for a in ATOMS)
    print(header)
    print(" " * w + "-" * (w * len(ATOMS)))
    for x in ATOMS:
        row = x.rjust(w) + "".join(dot_d1(x, y).rjust(w) for y in ATOMS)
        print(row)


# ============================================================
# Formatter
# ============================================================

def format_val(v: Any) -> str:
    if isinstance(v, Keyword):
        return f":{v.name}"
    if isinstance(v, Quoted):
        return f"'{format_val(v.expr)}"
    if isinstance(v, AppNode):
        return f"#app[{format_val(v.f)} {format_val(v.x)}]"
    if isinstance(v, Bundle):
        return f"#bundle[{format_val(v.f)} {format_val(v.x)}]"
    if isinstance(v, Partial):
        return f"#partial[{format_val(v.f)}]"
    if isinstance(v, List):
        return "(" + " ".join(format_val(i) for i in v.items) + ")"
    if isinstance(v, Symbol):
        return v.name
    return str(v)


# ============================================================
# REPL
# ============================================================

def repl():
    """Interactive REPL."""
    print("Distinction Structures REPL")
    print("  21 atoms: 17 from Δ₁ + QUOTE, EVAL, APP, UNAPP")
    print("  Type (discover!) to run self-discovery")
    print("  Type (table) to see the Cayley table")
    print("  Ctrl-D to exit\n")

    while True:
        try:
            line = input("ds> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not line or line.startswith(";"):
            continue

        try:
            exprs = parse(line)
            for expr in exprs:
                result = repl_eval(expr)
                print(f"=> {format_val(result)}")
        except Exception as e:
            print(f"Error: {e}")


def eval_string(text: str):
    """Evaluate a string of expressions."""
    exprs = parse(text)
    result = None
    for expr in exprs:
        result = repl_eval(expr)
    return result


def eval_file(path: str):
    """Evaluate a file."""
    with open(path) as f:
        text = f.read()
    exprs = parse(text)
    for expr in exprs:
        result = repl_eval(expr)
        print(f"=> {format_val(result)}")


# ============================================================
# Main
# ============================================================

def main():
    if len(sys.argv) > 1:
        if sys.argv[1] == "-e":
            result = eval_string(" ".join(sys.argv[2:]))
            print(format_val(result))
        else:
            eval_file(sys.argv[1])
    else:
        repl()


if __name__ == "__main__":
    main()
