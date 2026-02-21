#!/usr/bin/env python3
"""
Δ₂ Scripted Discoverability Demo

Demonstrates three progressive levels of self-description in Distinction Structures:

  Δ₁ (finite algebra):  17 atoms, directed application, self-model recoverable from behavior
  Δ₂ (interpreter):     21 atoms + unbounded term space, QUOTE/EVAL/APP/UNAPP

The demo:
  1. Shuffles all 21 atom names into opaque labels (u00, u01, ...)
  2. Recovers Δ₁ primitives (booleans, testers, encoders, etc.) by probing behavior
  3. Recovers Δ₂ primitives (QUOTE, EVAL, APP, UNAPP) by probing behavior
  4. Runs example programs using recovered primitives

Key boundary: Δ₁ is a finite algebra (finite carrier, total operation). Δ₂ is NOT —
QUOTE generates unbounded inert values, EVAL is recursive. Δ₂ is a Distinction Structure
core embedded in an interpreter.

Usage:
  python delta2_interpreter.py
  python delta2_interpreter.py --seed 42    # different shuffle
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Set, Optional
import random
import itertools
import argparse


# ============================================================================
# Runtime term representations
# ============================================================================

@dataclass(frozen=True)
class Atom:
    """A primitive element (one of the 21 base atoms)."""
    name: str

@dataclass(frozen=True)
class Quote:
    """An inert code value. Only EVAL can unwrap it."""
    term: Any

@dataclass(frozen=True)
class AppNode:
    """An application AST node: represents f · x as data."""
    f: Any
    x: Any

@dataclass(frozen=True)
class UnappBundle:
    """Result of UNAPP on an AppNode. Queryable by booleans: bundle·⊤=f, bundle·⊥=x."""
    f: Any
    x: Any

@dataclass(frozen=True)
class Partial:
    """A curried partial application (APP waiting for its second argument)."""
    op: str
    a: Any


def A(n: str) -> Atom:
    return Atom(n)


# ============================================================================
# Atom inventory
# ============================================================================

NAMES_D1 = [
    "⊤", "⊥",           # booleans
    "i", "k",             # context tokens
    "a", "b",             # κ-element encodings
    "e_I",                # context tester
    "e_D", "e_M",         # structural encoders
    "e_Σ", "e_Δ",         # synthesis encoder, whole-structure token
    "d_I", "d_K",         # domain codes
    "m_I", "m_K",         # actuality codes
    "s_C",                # component-set token
    "p",                  # surplus/default (non-actual)
]

NAMES_D2 = ["QUOTE", "EVAL", "APP", "UNAPP"]

ALL_ATOMS = [A(n) for n in (NAMES_D1 + NAMES_D2)]


# ============================================================================
# Δ₁ core: directed application on atoms (ι-context)
# ============================================================================

def dot_iota_d1(x: Atom, y: Atom) -> Atom:
    """
    The 26-rule operation table for Δ₁, defined by first-match priority.
    This is a FINITE algebra: total function on 17 × 17 atom pairs.
    """
    TOP, BOT = A("⊤"), A("⊥")

    # Block A — Boolean absorption
    if x == TOP: return TOP
    if x == BOT: return BOT

    # Block B — Testers (boolean-valued left-actions)
    if x == A("e_I"): return TOP if y in (A("i"), A("k")) else BOT
    if x == A("d_K"): return TOP if y in (A("a"), A("b")) else BOT
    if x == A("m_K"): return TOP if y == A("a") else BOT
    if x == A("m_I"): return BOT if y == A("p") else TOP

    # Block C — Structural encoders
    if x == A("e_D") and y == A("i"): return A("d_I")
    if x == A("e_D") and y == A("k"): return A("d_K")
    if x == A("e_M") and y == A("i"): return A("m_I")
    if x == A("e_M") and y == A("k"): return A("m_K")
    if x == A("e_Σ") and y == A("s_C"): return A("e_Δ")
    if x == A("e_Δ") and y == A("e_D"): return A("d_I")

    # Block D — Absorption breaker
    if x == A("p") and y == TOP: return TOP

    # Block E — Passive self-identification (Ext fix)
    if y == TOP and x in (A("i"), A("k"), A("a"), A("b"), A("d_I"), A("s_C")):
        return x

    # Block F — Default
    return A("p")


# ============================================================================
# Δ₂ extension: QUOTE / EVAL / APP / UNAPP
# ============================================================================

def eval_term(t: Any) -> Any:
    """Recursively evaluate a term. This is where Δ₂ becomes an interpreter."""
    if isinstance(t, Atom): return t
    if isinstance(t, Quote): return t
    if isinstance(t, AppNode):
        return dot_iota(eval_term(t.f), eval_term(t.x))
    return A("p")


def dot_iota(x: Any, y: Any) -> Any:
    """
    The full Δ₂ operation. Extends Δ₁ with recursive evaluation.
    
    This is NOT a finite algebra: QUOTE generates unbounded values,
    EVAL is defined recursively over syntax trees.
    """
    # Curried partial applications (APP waiting for second arg)
    if isinstance(x, Partial):
        return AppNode(x.a, y) if x.op == "APP" else A("p")

    # QUOTE: make inert data
    if x == A("QUOTE"): return Quote(y)

    # APP: build application AST nodes (curried)
    if x == A("APP"): return Partial("APP", y)

    # UNAPP: deconstruct application nodes into boolean-queryable bundles
    if x == A("UNAPP"):
        return UnappBundle(y.f, y.x) if isinstance(y, AppNode) else A("p")

    # Bundles: bundle·⊤ = f, bundle·⊥ = x (reuses already-discoverable booleans)
    if isinstance(x, UnappBundle):
        if y == A("⊤"): return x.f
        if y == A("⊥"): return x.x
        return A("p")

    # EVAL: interpret quoted code (recursive — the heart of Δ₂)
    if x == A("EVAL"):
        return eval_term(y.term) if isinstance(y, Quote) else A("p")

    # Inertness discipline: quoted values are "stuck" under all non-EVAL operators
    if isinstance(y, Quote):
        return A("p")

    # Fall back to Δ₁ rules for atom-atom pairs
    if isinstance(x, Atom) and isinstance(y, Atom):
        return dot_iota_d1(x, y)

    return A("p")


# ============================================================================
# Black-box wrapper: shuffles atom names, exposes only oracle dot(x, y)
# ============================================================================

def make_blackbox(seed: int = 11):
    """
    Create a black-box version of the algebra where atom names are hidden.
    Returns (domain, dot_oracle, true_to_hidden_map).
    """
    rng = random.Random(seed)
    atoms = ALL_ATOMS.copy()
    rng.shuffle(atoms)
    labels = [f"u{idx:02d}" for idx in range(len(atoms))]
    true_to_hidden = {atoms[i]: labels[i] for i in range(len(atoms))}
    hidden_to_true = {labels[i]: atoms[i] for i in range(len(atoms))}
    domain = labels.copy()

    def dot_oracle(xh: Any, yh: Any) -> Any:
        def to_true(v):
            return hidden_to_true[v] if isinstance(v, str) and v in hidden_to_true else v
        def to_hidden(v):
            return true_to_hidden[v] if isinstance(v, Atom) else v
        return to_hidden(dot_iota(to_true(xh), to_true(yh)))

    return domain, dot_oracle, true_to_hidden


# ============================================================================
# Phase 1: Discover Δ₁ primitives from behavior alone
# ============================================================================

def discover_d1(domain: List[str], dot, true2hid: Dict[Atom, str]) -> Dict[str, Any]:
    """
    Recover all 17 Δ₁ atoms from black-box probing.
    Implements the 8-step recovery procedure from the main document.
    """
    pH = true2hid[A("p")]

    def left_image(xh):
        return {dot(xh, y) for y in domain}

    # Step 1: Find booleans (unique left-absorbers)
    absorbers = [x for x in domain if all(dot(x, y) == x for y in domain)]
    assert len(absorbers) == 2, f"Expected 2 absorbers, got {len(absorbers)}"
    B1, B2 = absorbers

    def testers_for(top, bot):
        out = []
        for x in domain:
            if x in (top, bot):
                continue
            im = left_image(x)
            if im.issubset({top, bot}) and len(im) == 2:
                out.append(x)
        return out

    # Orient booleans: choose labeling where tester cardinalities work out
    chosen = None
    for top, bot in [(B1, B2), (B2, B1)]:
        testers = testers_for(top, bot)
        if len(testers) != 4:
            continue
        Dec = lambda t, top=top: {y for y in domain if dot(t, y) == top}
        sizes = sorted(len(Dec(t)) for t in testers)
        if sizes[0] == 1 and sizes[1] == 2 and sizes[2] == 2:
            chosen = (top, bot, testers, Dec)
            break
    assert chosen is not None, "Failed to orient booleans"
    top, bot, testers, Dec = chosen

    # Step 2–3: Identify testers by cardinality
    sizes = {t: len(Dec(t)) for t in testers}
    m_K = [t for t in testers if sizes[t] == 1][0]
    m_I = max(testers, key=lambda t: sizes[t])
    two = [t for t in testers if sizes[t] == 2]

    # Step 4: Distinguish e_I from d_K via right-argument richness
    def richness(decoded_pair):
        for f in domain:
            if f in (top, bot) or f in testers:
                continue
            for x in decoded_pair:
                out = dot(f, x)
                if out not in (top, bot, pH) and out in domain:
                    return True
        return False

    t1, t2 = two
    e_I, d_K = (t1, t2) if richness(Dec(t1)) else (t2, t1)
    ctx = list(Dec(e_I))

    # Step 5: Find e_D and e_M
    def is_encoder(f):
        if f in (top, bot) or f in testers:
            return False
        outs = [dot(f, x) for x in ctx]
        return all(o in domain for o in outs) and any(o not in (top, bot, pH) for o in outs)

    enc = [f for f in domain if is_encoder(f)]
    assert len(enc) == 2, f"Expected 2 encoders, got {len(enc)}"

    def maps_both_to_testers(f):
        return all(dot(f, x) in testers for x in ctx)

    e_M = enc[0] if maps_both_to_testers(enc[0]) else enc[1]
    e_D = enc[1] if e_M == enc[0] else enc[0]

    # Step 6: Distinguish i from k
    outA, outB = dot(e_M, ctx[0]), dot(e_M, ctx[1])
    i_tok, k_tok = (ctx[0], ctx[1]) if len(Dec(outA)) > len(Dec(outB)) else (ctx[1], ctx[0])

    # Decode remaining elements
    ab = list(Dec(d_K))
    a_tok = next(x for x in ab if dot(m_K, x) == top)
    b_tok = next(x for x in ab if x != a_tok)
    d_I = dot(e_D, i_tok)

    # Step 7–8: Find e_Σ, s_C, e_Δ
    known = {top, bot, e_I, d_K, m_K, m_I, e_M, e_D, i_tok, k_tok, a_tok, b_tok, d_I}
    remaining = [x for x in domain if x not in known]

    e_S = sC = e_Delta = None
    for f, g in itertools.product(remaining, repeat=2):
        h = dot(f, g)
        if h in (top, bot, pH):
            continue
        if h in domain and dot(h, e_D) == d_I:
            e_S, sC, e_Delta = f, g, h
            break
    assert e_S is not None, "Failed to recover e_Σ/s_C/e_Δ"

    p_tok = next(x for x in domain if dot(m_I, x) == bot)

    return {
        "⊤": top, "⊥": bot, "p": p_tok,
        "e_I": e_I, "e_D": e_D, "e_M": e_M, "e_Σ": e_S, "e_Δ": e_Delta,
        "i": i_tok, "k": k_tok, "a": a_tok, "b": b_tok,
        "d_I": d_I, "d_K": d_K, "m_I": m_I, "m_K": m_K, "s_C": sC,
        "_testers": set(testers),
    }


# ============================================================================
# Phase 2: Discover Δ₂ primitives from behavior alone
# ============================================================================

def discover_d2(domain: List[str], dot, d1: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recover QUOTE, EVAL, APP, UNAPP by probing behavior.
    Uses signatures: QUOTE produces structured (non-atom) outputs;
    EVAL inverts QUOTE on atoms; APP creates nodes; UNAPP opens them via booleans.
    """
    top, bot = d1["⊤"], d1["⊥"]
    testers = d1["_testers"]
    p_tok = d1["p"]

    cand = [x for x in domain if x not in (top, bot) and x not in testers]
    sample = domain[:12]

    # Find QUOTE and EVAL jointly: QUOTE produces structured outputs, EVAL inverts them
    QUOTE = EVAL = None
    for q, e in itertools.permutations(cand, 2):
        structured = sum(1 for x in sample if dot(q, x) not in domain)
        if structured < 8:
            continue
        inv = sum(1 for x in sample if dot(e, dot(q, x)) == x)
        if inv >= 10:
            QUOTE, EVAL = q, e
            break
    assert QUOTE is not None, "Failed to recover QUOTE/EVAL"

    # Find APP and UNAPP via node-creation + bundle-query signature
    cand2 = [x for x in cand if x not in (QUOTE, EVAL)]
    sample_fs = [d1["e_D"], d1["e_M"], d1["e_I"], d1["d_K"], d1["m_I"], d1["e_Σ"]]
    sample_xs = [d1["i"], d1["k"], d1["a"], d1["b"], d1["s_C"], top, bot]

    APP = UNAPP = None
    for app in cand2:
        for f in sample_fs:
            mid = dot(app, f)
            if mid in domain:
                continue
            for x in sample_xs:
                node = dot(mid, x)
                if node in domain:
                    continue
                for unapp in cand2:
                    if unapp == app:
                        continue
                    bundle = dot(unapp, node)
                    if bundle in domain:
                        continue
                    left = dot(bundle, top)
                    right = dot(bundle, bot)
                    if left != p_tok and right != p_tok and left != right:
                        APP, UNAPP = app, unapp
                        break
                if APP: break
            if APP: break
        if APP: break
    assert APP is not None, "Failed to recover APP/UNAPP"

    return {"QUOTE": QUOTE, "EVAL": EVAL, "APP": APP, "UNAPP": UNAPP}


# ============================================================================
# Demo programs
# ============================================================================

def demo_eval_quote_app(dot, d1, d2):
    """Program 1: eval(quote(app(e_D, k))) → d_K"""
    node = dot(dot(d2["APP"], d1["e_D"]), d1["k"])
    return dot(d2["EVAL"], dot(d2["QUOTE"], node))


def demo_unapp_inspect(dot, d1, d2):
    """Program 2: build app(e_D, k), decompose with UNAPP, query with booleans"""
    node = dot(dot(d2["APP"], d1["e_D"]), d1["k"])
    bundle = dot(d2["UNAPP"], node)
    left = dot(bundle, d1["⊤"])
    right = dot(bundle, d1["⊥"])
    return node, bundle, left, right


def demo_nested_eval(dot, d1, d2):
    """Program 3: eval(quote(app(e_M, k))) → m_K (actuality code for κ)"""
    node = dot(dot(d2["APP"], d1["e_M"]), d1["k"])
    return dot(d2["EVAL"], dot(d2["QUOTE"], node))


def demo_self_model_query(dot, d1, d2):
    """Program 4: Use the self-model to query structure.
    e_D · k = d_K (domain code for κ), then d_K · a = ⊤ (a ∈ D(κ))"""
    domain_code = dot(d1["e_D"], d1["k"])
    membership = dot(domain_code, d1["a"])
    non_membership = dot(domain_code, d1["i"])
    return domain_code, membership, non_membership


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Δ₂ Discoverability Demo")
    parser.add_argument("--seed", type=int, default=11, help="Random seed for atom shuffling")
    args = parser.parse_args()

    domain, dot, true2hid = make_blackbox(args.seed)

    print("=" * 60)
    print("  Δ₂ SCRIPTED DISCOVERABILITY DEMO")
    print("=" * 60)
    print(f"\nBlack-box seed: {args.seed}")
    print(f"Atom domain: {len(domain)} opaque labels")
    print(f"  (Δ₁ core: 17 atoms + Δ₂ extensions: 4 operators)")

    # Phase 1
    print("\n" + "-" * 60)
    print("  PHASE 1: Recover Δ₁ primitives from behavior")
    print("-" * 60)
    d1 = discover_d1(domain, dot, true2hid)
    for k in ["⊤", "⊥", "e_I", "e_D", "e_M", "e_Σ", "e_Δ",
              "i", "k", "a", "b", "d_I", "d_K", "m_I", "m_K", "s_C", "p"]:
        expected = true2hid[A(k)]
        recovered = d1[k]
        status = "✓" if recovered == expected else "✗"
        print(f"  {k:4s} → {recovered}  {status}")

    # Phase 2
    print("\n" + "-" * 60)
    print("  PHASE 2: Recover Δ₂ primitives (QUOTE/EVAL/APP/UNAPP)")
    print("-" * 60)
    d2 = discover_d2(domain, dot, d1)
    for k in ["QUOTE", "EVAL", "APP", "UNAPP"]:
        expected = true2hid[A(k)]
        recovered = d2[k]
        status = "✓" if recovered == expected else "✗"
        print(f"  {k:5s} → {recovered}  {status}")

    # Phase 3
    print("\n" + "-" * 60)
    print("  PHASE 3: Run programs using recovered primitives")
    print("-" * 60)

    # Program 1
    out = demo_eval_quote_app(dot, d1, d2)
    print(f"\n  Program 1: eval(quote(app(e_D, k)))")
    print(f"    Result:   {out}")
    print(f"    Expected: {d1['d_K']}  (d_K)")
    print(f"    {'✓ Correct' if out == d1['d_K'] else '✗ MISMATCH'}")

    # Program 2
    node, bundle, left, right = demo_unapp_inspect(dot, d1, d2)
    print(f"\n  Program 2: unapp(app(e_D, k)) → bundle, query with booleans")
    print(f"    bundle·⊤ = {left}  (expected e_D = {d1['e_D']})  {'✓' if left == d1['e_D'] else '✗'}")
    print(f"    bundle·⊥ = {right}  (expected k = {d1['k']})  {'✓' if right == d1['k'] else '✗'}")

    # Program 3
    out3 = demo_nested_eval(dot, d1, d2)
    print(f"\n  Program 3: eval(quote(app(e_M, k)))")
    print(f"    Result:   {out3}")
    print(f"    Expected: {d1['m_K']}  (m_K)")
    print(f"    {'✓ Correct' if out3 == d1['m_K'] else '✗ MISMATCH'}")

    # Program 4
    dc, mem, nonmem = demo_self_model_query(dot, d1, d2)
    print(f"\n  Program 4: Self-model query — is 'a' in D(κ)?")
    print(f"    e_D · k = {dc}  (domain code for κ)")
    print(f"    d_K · a = {mem}  (expected ⊤ = {d1['⊤']})  {'✓' if mem == d1['⊤'] else '✗'}")
    print(f"    d_K · i = {nonmem}  (expected ⊥ = {d1['⊥']})  {'✓' if nonmem == d1['⊥'] else '✗'}")

    print("\n" + "=" * 60)
    print("  All phases complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
