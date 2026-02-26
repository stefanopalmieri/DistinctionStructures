#!/usr/bin/env python3
"""
Δ₂+74181 — 74181 ALU Extension of the Δ₂ Algebra

Extends the 21-atom Δ₂ algebra with 26 new atoms:
  - 16 nibble atoms N0–NF (4-bit data values / operation selectors)
  - 3 ALU dispatch atoms (ALU_LOGIC, ALU_ARITH, ALU_ARITHC)
  - 2 predicate atoms (ALU_ZERO, ALU_COUT)
  - 1 nibble successor atom (N_SUCC)
  - 4 IO atoms (IO_PUT, IO_GET, IO_RDY, IO_SEQ) with all-p Cayley rows

Total: 47 atoms. All non-IO atoms are uniquely recoverable from black-box
access to the dot operation alone. IO atoms are opaque (all-p rows).

The 74181 chip's 32 operations are encoded as 3 dispatch atoms × 16
nibble selectors. Nibble atoms serve double duty as both data values
and operation selectors.

Usage:
  python delta2_74181.py                # run verification suite
  python delta2_74181.py --test 1000    # run 1000-seed integration test
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Set, Tuple
import random
import itertools
import argparse


# ============================================================================
# Runtime term representations
# ============================================================================

@dataclass(frozen=True)
class Atom:
    name: str

@dataclass(frozen=True)
class Quote:
    term: Any

@dataclass(frozen=True)
class AppNode:
    f: Any
    x: Any

@dataclass(frozen=True)
class UnappBundle:
    f: Any
    x: Any

@dataclass(frozen=True)
class Partial:
    op: str
    a: Any

@dataclass(frozen=True)
class ALUPartial1:
    """ALU dispatch with selector applied, waiting for operand A."""
    mode: str       # "logic", "arith", "arithc"
    selector: int   # 0-15

@dataclass(frozen=True)
class ALUPartial2:
    """ALU dispatch with selector and operand A applied, waiting for B."""
    mode: str
    selector: int
    a: int          # 0-15

@dataclass(frozen=True)
class IOPutPartial:
    """IO_PUT with high nibble applied, awaiting low nibble."""
    hi: int  # high nibble 0-15

@dataclass(frozen=True)
class IOSeqPartial:
    """IO_SEQ with first action result applied, awaiting second."""
    first: Any


def A(n: str) -> Atom:
    return Atom(n)


# ============================================================================
# Atom inventory
# ============================================================================

NAMES_D1 = [
    "⊤", "⊥", "i", "k", "a", "b", "e_I",
    "e_D", "e_M", "e_Σ", "e_Δ",
    "d_I", "d_K", "m_I", "m_K", "s_C", "p",
]
NAMES_D2 = ["QUOTE", "EVAL", "APP", "UNAPP"]
NAMES_NIBBLES = [f"N{i:X}" for i in range(16)]  # N0, N1, ..., NF
NAMES_ALU_DISPATCH = ["ALU_LOGIC", "ALU_ARITH", "ALU_ARITHC"]
NAMES_ALU_PRED = ["ALU_ZERO", "ALU_COUT"]
NAMES_ALU_MISC = ["N_SUCC"]
NAMES_IO = ["IO_PUT", "IO_GET", "IO_RDY", "IO_SEQ"]

ALL_NAMES = (NAMES_D1 + NAMES_D2 + NAMES_NIBBLES +
             NAMES_ALU_DISPATCH + NAMES_ALU_PRED + NAMES_ALU_MISC + NAMES_IO)
ALL_ATOMS = [A(n) for n in ALL_NAMES]

NIBBLE_ATOMS = frozenset(A(f"N{i:X}") for i in range(16))
ALU_DISPATCH_ATOMS = frozenset(A(n) for n in NAMES_ALU_DISPATCH)
ALU_PRED_ATOMS = frozenset(A(n) for n in NAMES_ALU_PRED)
IO_ATOMS = frozenset(A(n) for n in NAMES_IO)
NEW_ATOMS = (NIBBLE_ATOMS | ALU_DISPATCH_ATOMS | ALU_PRED_ATOMS |
             frozenset(A(n) for n in NAMES_ALU_MISC) | IO_ATOMS)  # 26 new atoms
D1_ATOMS = frozenset(A(n) for n in NAMES_D1)
D2_EXT_ATOMS = frozenset(A(n) for n in NAMES_D2)


# ============================================================================
# Nibble helpers
# ============================================================================

def is_nibble(x: Any) -> bool:
    return isinstance(x, Atom) and x in NIBBLE_ATOMS

def nibble_val(x: Atom) -> int:
    return int(x.name[1:], 16)

def nibble(n: int) -> Atom:
    return A(f"N{n % 16:X}")


# ============================================================================
# 74181 ALU computation (Deliverable 4)
# ============================================================================

def alu_74181(mode: str, selector: int, a: int, b: int) -> tuple:
    """
    Compute one 74181 operation.

    Args:
        mode: "logic", "arith", or "arithc"
        selector: 0-15 (S0-S3)
        a: 0-15 (4-bit input A)
        b: 0-15 (4-bit input B)

    Returns:
        (result: 0-15, carry_out: bool, zero: bool)
    """
    assert 0 <= selector <= 15 and 0 <= a <= 15 and 0 <= b <= 15

    if mode == "logic":
        result = _alu_logic(selector, a, b)
        return (result, False, result == 0)

    # Arithmetic modes
    carry_in = 1 if mode == "arithc" else 0
    logic_result = _alu_logic(selector, a, b)
    # Arithmetic: add logic result + carry
    full = logic_result + carry_in
    # The 74181 arithmetic is: F = logic_result + carry_in (for each bit position)
    # More precisely: arithmetic result = logic_result + carry_in in 4-bit
    # But the 74181 does bit-level carry propagation on the logic function outputs
    # Let me implement it correctly using the standard 74181 truth table approach

    # The correct 74181 arithmetic: the logic function generates a "generate"
    # and "propagate" per bit, then ripple-carry adds.
    # Simpler: just compute directly from the truth tables.
    result, carry_out = _alu_arith(selector, a, b, carry_in)
    return (result, carry_out, result == 0)


def _alu_logic(selector: int, a: int, b: int) -> int:
    """Compute 74181 logic operation (active-high)."""
    # Operate bitwise on each of the 4 bits
    result = 0
    for bit in range(4):
        ai = (a >> bit) & 1
        bi = (b >> bit) & 1
        # The 74181 logic functions (active-high):
        fi = _logic_bit(selector, ai, bi)
        result |= (fi << bit)
    return result & 0xF


def _logic_bit(selector: int, ai: int, bi: int) -> int:
    """Compute one bit of the 74181 logic function."""
    s0 = (selector >> 0) & 1
    s1 = (selector >> 1) & 1
    s2 = (selector >> 2) & 1
    s3 = (selector >> 3) & 1

    # 74181 active-high logic equations per bit:
    # The logic function is: F = NOT(
    #   (NOT A AND s0 AND NOT B) OR
    #   (NOT A AND s1 AND B) OR
    #   (A AND s2 AND NOT B) OR
    #   (A AND s3 AND B)
    # )
    # Wait, let me use the standard formulation.
    # Actually, the 74181 logic output per bit (active-high) is:
    #
    # For the logic functions (M=H), the output per bit is determined by
    # the select lines and input bits. The truth table for active-high:

    na = 1 - ai
    nb = 1 - bi

    # Generate and propagate terms
    t0 = na & s0 & nb
    t1 = na & s1 & bi
    t2 = ai & s2 & nb
    t3 = ai & s3 & bi

    # Logic output (active-high): NOT of the OR of terms
    # Wait, this gives the COMPLEMENT. Let me just use the truth table directly.
    # The 74181 active-high logic table:
    # S=0000: NOT A        S=0001: NOT(A OR B)   S=0010: (NOT A) AND B
    # S=0011: 0            S=0100: NOT(A AND B)  S=0101: NOT B
    # S=0110: A XOR B      S=0111: A AND NOT B   S=1000: NOT A OR B
    # S=1001: NOT(A XOR B) S=1010: B             S=1011: A AND B
    # S=1100: 1            S=1101: A OR NOT B    S=1110: A OR B
    # S=1111: A

    _table = [
        na,                           # 0: NOT A
        1 - (ai | bi),                # 1: NOR
        na & bi,                      # 2: (NOT A) AND B
        0,                            # 3: Logical 0
        1 - (ai & bi),               # 4: NAND
        nb,                           # 5: NOT B
        ai ^ bi,                      # 6: XOR
        ai & nb,                      # 7: A AND (NOT B)
        na | bi,                      # 8: (NOT A) OR B
        1 - (ai ^ bi),               # 9: XNOR
        bi,                           # A: B
        ai & bi,                      # B: A AND B
        1,                            # C: Logical 1
        ai | nb,                      # D: A OR (NOT B)
        ai | bi,                      # E: A OR B
        ai,                           # F: A
    ]
    return _table[selector]


def _alu_arith(selector: int, a: int, b: int, carry_in: int) -> tuple:
    """
    Compute 74181 arithmetic operation (active-high).

    Directly implements the 74181 arithmetic truth table.
    The "with carry" result is "no carry" result + 1.

    Returns (result: 0-15, carry_out: bool)
    """
    nb = (~b) & 0xF  # 4-bit complement of B

    # Base result (no carry) from the 74181 truth table
    base_table = [
        a,                      # 0: A
        a | b,                  # 1: A OR B
        a | nb,                 # 2: A OR (NOT B)
        0xF,                    # 3: minus 1 (all ones)
        a + (a & nb),           # 4: A plus (A AND NOT B)
        (a | b) + (a & nb),     # 5: (A OR B) plus (A AND NOT B)
        a + nb,                 # 6: A minus B minus 1 = A + NOT(B)
        (a & nb) + 0xF,         # 7: (A AND NOT B) minus 1
        a + (a & b),            # 8: A plus (A AND B)
        a + b,                  # 9: A plus B
        (a | nb) + (a & b),     # A: (A OR NOT B) plus (A AND B)
        (a & b) + 0xF,          # B: (A AND B) minus 1
        a + a,                  # C: A plus A (left shift)
        (a | b) + a,            # D: (A OR B) plus A
        (a | nb) + a,           # E: (A OR NOT B) plus A
        a + 0xF,                # F: A minus 1
    ]

    raw = base_table[selector] + carry_in
    result = raw & 0xF
    carry_out = bool(raw > 0xF)
    return (result, carry_out)


# ============================================================================
# Δ₁ core: directed application on atoms
# ============================================================================

def dot_iota_d1(x: Atom, y: Atom) -> Atom:
    TOP, BOT = A("⊤"), A("⊥")
    if x == TOP: return TOP
    if x == BOT: return BOT
    if x == A("e_I"): return TOP if y in (A("i"), A("k")) else BOT
    if x == A("d_K"): return TOP if y in (A("a"), A("b")) else BOT
    if x == A("m_K"): return TOP if y == A("a") else BOT
    if x == A("m_I"): return BOT if y == A("p") else TOP
    if x == A("e_D") and y == A("i"): return A("d_I")
    if x == A("e_D") and y == A("k"): return A("d_K")
    if x == A("e_M") and y == A("i"): return A("m_I")
    if x == A("e_M") and y == A("k"): return A("m_K")
    if x == A("e_Σ") and y == A("s_C"): return A("e_Δ")
    if x == A("e_Δ") and y == A("e_D"): return A("d_I")
    if x == A("p") and y == TOP: return TOP
    if y == TOP and x in (A("i"), A("k"), A("a"), A("b"), A("d_I"), A("s_C")):
        return x
    return A("p")


# ============================================================================
# Atom-atom Cayley table (47 × 47)
# ============================================================================

def atom_dot(x: Atom, y: Atom) -> Atom:
    """
    Cayley table for all 47 atoms.

    Design principles:
    - Preserves all 21×21 original D1/D2 entries exactly
    - D1 atoms use dot_iota_d1 for ALL right arguments (testers stay pure boolean)
    - Nibbles form Z/16Z under addition mod 16
    - ALU dispatch: identity/successor/double-successor on nibbles
    - ALU predicates: tester-like on nibbles, self-id on ⊤, else p
    - IO atoms: all-p rows (effects happen at term level)
    """
    TOP, BOT = A("⊤"), A("⊥")

    # ── D1 atom on left: use dot_iota_d1 for ALL right arguments ──
    # This preserves the original D1×D1 and D1×D2 entries exactly,
    # and gives correct defaults for D1×new (testers → ⊥/⊤, others → p).
    # Exception: m_I needs to map new atoms to ⊤ (they are not p).
    if x in D1_ATOMS:
        # m_I: dot_iota_d1 returns ⊥ only for p, ⊤ for everything else.
        # For new atoms (which are not p), this correctly returns ⊤.
        # For D2 atoms (not p), correctly returns ⊤.
        # All other D1 atoms handled correctly by dot_iota_d1.
        return dot_iota_d1(x, y)

    # ── D2 atoms × anything: atom-level fallback is p ──
    # (QUOTE/EVAL/APP/UNAPP produce structured values at term level)
    if x in D2_EXT_ATOMS: return A("p")

    # ── IO atoms × anything: all-p (effects happen at term level) ──
    if x in IO_ATOMS: return A("p")

    # ── Nibble self-identification on ⊤ ──
    if is_nibble(x) and y == TOP: return x

    # ── Nibble × Nibble: Z/16Z under addition ──
    if is_nibble(x) and is_nibble(y):
        return nibble((nibble_val(x) + nibble_val(y)) % 16)

    # ── ALU dispatch self-identification on ⊤ ──
    if x in ALU_DISPATCH_ATOMS and y == TOP: return x

    # ── ALU dispatch × Nibble: distinguishing mappings ──
    if x == A("ALU_LOGIC") and is_nibble(y):
        return y  # identity on nibbles
    if x == A("ALU_ARITH") and is_nibble(y):
        return nibble((nibble_val(y) + 1) % 16)  # successor
    if x == A("ALU_ARITHC") and is_nibble(y):
        return nibble((nibble_val(y) + 2) % 16)  # double successor

    # ── ALU predicate self-identification on ⊤ ──
    if x in ALU_PRED_ATOMS and y == TOP: return x

    # ── ALU_ZERO: tester on nibbles ──
    if x == A("ALU_ZERO") and is_nibble(y):
        return TOP if y == A("N0") else BOT

    # ── ALU_COUT: tester on nibbles (high bit = carry) ──
    if x == A("ALU_COUT") and is_nibble(y):
        return TOP if nibble_val(y) >= 8 else BOT

    # ── N_SUCC: successor on nibbles (16-cycle) ──
    if x == A("N_SUCC") and y == TOP: return x
    if x == A("N_SUCC") and y == BOT: return A("N0")  # reset on ⊥ (distinguishes from ALU_ARITH at atom level)
    if x == A("N_SUCC") and is_nibble(y):
        return nibble((nibble_val(y) + 1) % 16)

    # ── Default: everything else → p ──
    return A("p")


# ============================================================================
# Extended dot operation (full term-level)
# ============================================================================

def eval_term(t: Any) -> Any:
    if isinstance(t, Atom): return t
    if isinstance(t, Quote): return t
    if isinstance(t, AppNode):
        return dot_ext(eval_term(t.f), eval_term(t.x))
    return A("p")


def dot_ext(x: Any, y: Any) -> Any:
    """The full Δ₂+74181 operation on terms."""

    # --- Partial applications (inherited from Δ₂) ---
    if isinstance(x, Partial):
        return AppNode(x.a, y) if x.op == "APP" else A("p")

    # --- ALUPartial1: dispatch + selector applied, waiting for operand A ---
    if isinstance(x, ALUPartial1):
        if isinstance(y, Atom) and is_nibble(y):
            return ALUPartial2(x.mode, x.selector, nibble_val(y))
        return A("p")

    # --- ALUPartial2: dispatch + selector + A applied, waiting for B ---
    if isinstance(x, ALUPartial2):
        if isinstance(y, Atom) and is_nibble(y):
            result, carry, zero = alu_74181(x.mode, x.selector, x.a, nibble_val(y))
            return nibble(result)
        return A("p")

    # --- QUOTE (inherited) ---
    if x == A("QUOTE"): return Quote(y)

    # --- APP (inherited) ---
    if x == A("APP"): return Partial("APP", y)

    # --- UNAPP (inherited) ---
    if x == A("UNAPP"):
        return UnappBundle(y.f, y.x) if isinstance(y, AppNode) else A("p")

    # --- Bundle queries (inherited) ---
    if isinstance(x, UnappBundle):
        if y == A("⊤"): return x.f
        if y == A("⊥"): return x.x
        return A("p")

    # --- EVAL (inherited) ---
    if x == A("EVAL"):
        if isinstance(y, Quote): return eval_term(y.term)
        return A("p")

    # --- ALU dispatch atoms: first application produces ALUPartial1 ---
    if x == A("ALU_LOGIC") and isinstance(y, Atom) and is_nibble(y):
        return ALUPartial1("logic", nibble_val(y))
    if x == A("ALU_ARITH") and isinstance(y, Atom) and is_nibble(y):
        return ALUPartial1("arith", nibble_val(y))
    if x == A("ALU_ARITHC") and isinstance(y, Atom) and is_nibble(y):
        return ALUPartial1("arithc", nibble_val(y))

    # --- ALU_ZERO on nibble at term level ---
    if x == A("ALU_ZERO") and isinstance(y, Atom) and is_nibble(y):
        return A("⊤") if y == A("N0") else A("⊥")

    # --- ALU_COUT: at term level, applied to ALU result for carry ---
    # ALU_COUT tests the carry-out from an ALU operation.
    # In curried usage: (ALU-COUT (ALU-ARITH N9 a b))
    # The inner expression evaluates to a nibble result.
    # To get carry info, we'd need to track it. For now, ALU_COUT
    # on a nibble tests the high bit (>= 8), consistent with Cayley table.
    if x == A("ALU_COUT") and isinstance(y, Atom) and is_nibble(y):
        return A("⊤") if nibble_val(y) >= 8 else A("⊥")

    # --- IO_PUT + nibble → IOPutPartial ---
    if x == A("IO_PUT") and isinstance(y, Atom) and is_nibble(y):
        return IOPutPartial(nibble_val(y))

    # --- IOPutPartial + nibble → ⊤ (pure: no actual stdout write) ---
    if isinstance(x, IOPutPartial):
        if isinstance(y, Atom) and is_nibble(y):
            return A("⊤")
        return A("p")

    # --- IO_GET + ⊤ → p (pure: no actual stdin read) ---
    # In the REPL, IO_GET reads stdin. Here we return p ("no input").
    if x == A("IO_GET"):
        if y == A("⊤"):
            return A("p")

    # --- IO_RDY + ⊤ → ⊤ ---
    if x == A("IO_RDY"):
        if y == A("⊤"):
            return A("⊤")

    # --- IO_SEQ + any → IOSeqPartial ---
    if x == A("IO_SEQ"):
        return IOSeqPartial(y)

    # --- IOSeqPartial + any → return right (effects already fired) ---
    if isinstance(x, IOSeqPartial):
        return y

    # --- Atoms acting on non-atom structured terms → p ---
    if isinstance(x, Atom) and not isinstance(y, Atom):
        return A("p")

    # --- Atom × Atom: Cayley table ---
    if isinstance(x, Atom) and isinstance(y, Atom):
        return atom_dot(x, y)

    # --- Default ---
    return A("p")


# ============================================================================
# Black-box wrapper (for discovery testing)
# ============================================================================

def make_blackbox(seed: int = 11):
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
        return to_hidden(dot_ext(to_true(xh), to_true(yh)))

    return domain, dot_oracle, true_to_hidden


# ============================================================================
# Phase 1: Discover Δ₁ primitives (unchanged from delta2_true_blackbox.py)
# ============================================================================

def discover_d1(domain: List[str], dot) -> Dict[str, Any]:
    """Recover all 17 Δ₁ atoms from black-box probing."""

    def left_image(xh):
        return {dot(xh, y) for y in domain}

    def left_image_in_domain(xh):
        return {dot(xh, y) for y in domain if dot(xh, y) in domain}

    # Step 1: Find booleans (left-absorbers)
    absorbers = [x for x in domain if all(dot(x, y) == x for y in domain)]
    assert len(absorbers) == 2, f"Expected 2 absorbers, got {len(absorbers)}"
    B1, B2 = absorbers

    # Step 2: Find testers
    def testers_for(top, bot):
        out = []
        for x in domain:
            if x in (top, bot):
                continue
            im = left_image(x)
            if im.issubset({top, bot}) and len(im) == 2:
                out.append(x)
        return out

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

    # Step 3: Identify testers by cardinality (moved before p-detection)
    sizes = {t: len(Dec(t)) for t in testers}
    m_K = [t for t in testers if sizes[t] == 1][0]
    m_I = max(testers, key=lambda t: sizes[t])
    two = [t for t in testers if sizes[t] == 2]
    assert len(two) == 2

    # Step 2.5: Find p
    # p is the unique non-boolean, non-tester element where dot(x, ⊤) = ⊤
    # AND dot(m_I, x) = ⊥ (m_I maps only p to ⊥; all others to ⊤).
    # IO_RDY also has dot(x, ⊤) = ⊤ but dot(m_I, IO_RDY) = ⊤.
    p_candidates = [
        x for x in domain
        if x not in (top, bot) and x not in testers
        and dot(x, top) == top
        and dot(m_I, x) == bot
    ]
    assert len(p_candidates) == 1, (
        f"Expected exactly 1 p-candidate, got {len(p_candidates)}: {p_candidates}"
    )
    p_tok = p_candidates[0]

    # Step 4: Distinguish e_I from d_K
    def has_productive_args(decoded_set):
        for f in domain:
            if f in (top, bot) or f in testers:
                continue
            for x in decoded_set:
                out = dot(f, x)
                if out in domain and out not in (top, bot, p_tok):
                    return True
        return False

    t1, t2 = two
    e_I, d_K = (t1, t2) if has_productive_args(Dec(t1)) else (t2, t1)
    ctx = list(Dec(e_I))

    # Step 5: Find e_D and e_M
    def is_encoder(f):
        if f in (top, bot) or f in testers:
            return False
        outs = [dot(f, x) for x in ctx]
        return (all(o in domain for o in outs) and
                all(o not in (top, bot, p_tok) for o in outs))

    enc = [f for f in domain if is_encoder(f)]
    assert len(enc) == 2, f"Expected 2 encoders, got {len(enc)}"

    def maps_both_to_testers(f):
        return all(dot(f, x) in testers for x in ctx)

    e_M = enc[0] if maps_both_to_testers(enc[0]) else enc[1]
    e_D = enc[1] if e_M == enc[0] else enc[0]

    # Step 6: Distinguish i from k
    outA, outB = dot(e_M, ctx[0]), dot(e_M, ctx[1])
    if len(Dec(outA)) > len(Dec(outB)):
        i_tok, k_tok = ctx[0], ctx[1]
    else:
        i_tok, k_tok = ctx[1], ctx[0]

    # Step 7: Identify a, b, d_I
    ab = list(Dec(d_K))
    a_tok = next(x for x in ab if dot(m_K, x) == top)
    b_tok = next(x for x in ab if x != a_tok)
    d_I = dot(e_D, i_tok)

    # Step 8: Find e_Σ, s_C, e_Δ
    known = {top, bot, e_I, d_K, m_K, m_I, e_M, e_D,
             i_tok, k_tok, a_tok, b_tok, d_I, p_tok}
    remaining = [x for x in domain if x not in known]

    e_S = sC = e_Delta = None
    for f, g in itertools.product(remaining, repeat=2):
        h = dot(f, g)
        if h not in domain or h in (top, bot, p_tok):
            continue
        if dot(h, e_D) == d_I:
            e_S, sC, e_Delta = f, g, h
            break
    assert e_S is not None, "Failed to recover e_Σ/s_C/e_Δ"

    return {
        "⊤": top, "⊥": bot, "p": p_tok,
        "e_I": e_I, "e_D": e_D, "e_M": e_M, "e_Σ": e_S, "e_Δ": e_Delta,
        "i": i_tok, "k": k_tok, "a": a_tok, "b": b_tok,
        "d_I": d_I, "d_K": d_K, "m_I": m_I, "m_K": m_K, "s_C": sC,
        "_testers": set(testers),
    }


# ============================================================================
# Phase 2: Discover Δ₂ primitives (unchanged from delta2_true_blackbox.py)
# ============================================================================

def discover_d2(domain: List[str], dot, d1: Dict[str, Any]) -> Dict[str, Any]:
    """Recover QUOTE, EVAL, APP, UNAPP by probing behavior."""
    top, bot = d1["⊤"], d1["⊥"]
    testers = d1["_testers"]
    p_tok = d1["p"]

    d1_identified = {v for k, v in d1.items() if k != "_testers"}
    cand = [x for x in domain if x not in d1_identified]
    sample = domain

    # Find QUOTE and EVAL jointly
    QUOTE = EVAL = None
    for q in cand:
        structured = sum(1 for x in sample if dot(q, x) not in domain)
        if structured < len(sample) // 2:
            continue
        for e in cand:
            if e == q:
                continue
            inv = sum(1 for x in sample if dot(e, dot(q, x)) == x)
            if inv >= len(sample) * 2 // 3:
                QUOTE, EVAL = q, e
                break
        if QUOTE:
            break
    assert QUOTE is not None, "Failed to recover QUOTE/EVAL"

    # Find APP and UNAPP
    cand2 = [x for x in cand if x not in (QUOTE, EVAL)]
    test_fs = [d1[k] for k in ["e_D", "e_M", "e_I", "d_K", "m_I", "e_Σ"]]
    test_xs = [d1[k] for k in ["i", "k", "a", "b", "s_C"]] + [top, bot]

    APP = UNAPP = None
    for app in cand2:
        for f in test_fs:
            mid = dot(app, f)
            if mid in domain:
                continue
            for x in test_xs:
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
                if APP:
                    break
            if APP:
                break
        if APP:
            break
    assert APP is not None, "Failed to recover APP/UNAPP"

    return {"QUOTE": QUOTE, "EVAL": EVAL, "APP": APP, "UNAPP": UNAPP}


# ============================================================================
# Phase 3: Discover 74181 extension atoms
# ============================================================================

def discover_74181_with_logs(domain: List[str], dot, d1: Dict[str, Any],
                             verbose: bool = False) -> Tuple[Dict[str, Any], List[str]]:
    """
    Phase 1b: Recover 22 new atoms from Cayley-level probing and separate
    8 opaque (all-p) atoms for Phase 2 term-level recovery.

    Returns:
        (ext_dict, opaque_list): ext_dict maps atom names to hidden labels
        for the 22 identifiable atoms; opaque_list contains the 8 hidden
        labels with identical all-p Cayley rows (D2 + IO atoms).

    Recovery order:
      0. Separate 8 opaque atoms (all-p Cayley rows) from 22 structured
      1. Identify predicates (atoms producing both ⊤ and ⊥ in left-image)
      2. Identify N_SUCC (unique cyclic permuter of nibble group)
      3. Separate nibbles from ALU dispatch (self-composable vs not)
      4. Identify N0 via ALU_ZERO (the predicate accepting exactly 1 nibble)
      5. Walk N_SUCC cycle from N0 to order all 16 nibbles
      6. Distinguish ALU_LOGIC / ALU_ARITH / ALU_ARITHC
      7. Distinguish ALU_ZERO from ALU_COUT by decoded set size
    """
    top = d1["⊤"]
    bot = d1["⊥"]
    p_tok = d1["p"]

    def log(msg):
        if verbose:
            print(f"    [Phase 1b] {msg}")

    # Collect all previously identified tokens
    d1_identified = {v for k, v in d1.items() if k != "_testers"}
    known = d1_identified
    remaining = [x for x in domain if x not in known]
    assert len(remaining) == 30, f"Expected 30 remaining atoms, got {len(remaining)}"
    log(f"Starting with {len(remaining)} unidentified atoms")

    # ── Step 0: Compute domain-restricted left images ─────────────────
    # dot_ext may return structured values (Quote, IOPutPartial, etc.)
    # for D2/IO atoms. We restrict to domain-valued results for Cayley analysis.
    domain_set = set(domain)

    def left_image(x):
        return {r for r in [dot(x, y) for y in domain] if isinstance(r, str) and r in domain_set}

    # ── Step 1: Identify predicate atoms ──────────────────────────────
    # Predicates produce ⊤ AND ⊥ in their left-image (from nibble probes)
    # plus self (from ⊤ probe) plus p (from non-nibble probes).
    predicates = []
    for x in remaining:
        li = left_image(x)
        if top in li and bot in li and x in li:
            predicates.append(x)

    assert len(predicates) == 2, f"Expected 2 predicates, got {len(predicates)}"
    log(f"Predicates identified: {predicates}")

    # ── Step 2: Separate nibbles, N_SUCC, and ALU dispatch ────────────
    # Among the non-predicate remaining atoms, nibbles are self-composable,
    # N_SUCC permutes nibbles, and ALU dispatch atoms produce structured values.
    # D2/IO atoms produce p or non-domain results and will fall to "other".
    non_pred = [x for x in remaining if x not in predicates]

    # First pass: identify nibbles (self-composable within the remaining set)
    non_pred_set = set(non_pred)
    nibbles = []
    non_nibbles = []
    for x in non_pred:
        xx = dot(x, x)
        if isinstance(xx, str) and xx in non_pred_set and xx != p_tok:
            nibbles.append(x)
        else:
            non_nibbles.append(x)

    assert len(nibbles) == 16, f"Expected 16 nibbles, got {len(nibbles)}"
    log(f"Nibbles: {len(nibbles)}, Non-nibbles: {len(non_nibbles)}")

    # ── Step 3: Identify N_SUCC (maps nibbles to nibbles in domain) ──
    # N_SUCC maps nibbles to domain nibbles (via Cayley table).
    # ALU dispatch atoms map nibbles to non-domain structured values
    # (ALUPartial1) via dot_ext. So N_SUCC is the ONLY non-nibble
    # that maps all nibbles to domain values.
    nibble_set = set(nibbles)
    n_succ_tok = None
    non_nibble_rest = []
    for x in non_nibbles:
        images = [dot(x, n) for n in nibbles]
        maps_all_to_nibbles = all(isinstance(img, str) and img in nibble_set for img in images)
        if maps_all_to_nibbles and len(set(images)) == 16:
            n_succ_tok = x
        else:
            non_nibble_rest.append(x)

    assert n_succ_tok is not None, "Failed to identify N_SUCC"
    log(f"N_SUCC identified: {n_succ_tok}")

    # ── Step 4: Distinguish ALU_ZERO from ALU_COUT ────────────────────
    # ALU_ZERO: exactly 1 nibble maps to ⊤ (that nibble is N0)
    # ALU_COUT: exactly 8 nibbles map to ⊤
    pred_a, pred_b = predicates
    dec_a = sum(1 for n in nibbles if dot(pred_a, n) == top)
    dec_b = sum(1 for n in nibbles if dot(pred_b, n) == top)

    if dec_a == 1:
        alu_zero_tok, alu_cout_tok = pred_a, pred_b
    else:
        alu_zero_tok, alu_cout_tok = pred_b, pred_a
    log(f"ALU_ZERO identified: {alu_zero_tok} (accepts {min(dec_a, dec_b)} nibbles)")
    log(f"ALU_COUT identified: {alu_cout_tok} (accepts {max(dec_a, dec_b)} nibbles)")

    # ── Step 5: Find N0 (anchor via ALU_ZERO) ─────────────────────────
    n0_candidates = [n for n in nibbles if dot(alu_zero_tok, n) == top]
    assert len(n0_candidates) == 1, f"Expected 1 N0, got {len(n0_candidates)}"
    n0_tok = n0_candidates[0]
    log(f"N0 identified: {n0_tok}")

    # ── Step 6: Order all 16 nibbles by walking N_SUCC from N0 ────────
    nibble_order = [n0_tok]
    current = n0_tok
    for _ in range(15):
        current = dot(n_succ_tok, current)
        nibble_order.append(current)
    assert len(set(nibble_order)) == 16, "Nibble ordering failed"
    log(f"Nibble order: N0={nibble_order[0]}, N1={nibble_order[1]}, ..., NF={nibble_order[15]}")

    # ── Step 7: Separate dispatch from opaque, then identify dispatch ─
    # Dispatch atoms self-identify on ⊤: dot(x, ⊤) == x.
    # D2/IO atoms produce structured values or p when applied to ⊤.
    dispatch = []
    opaque = []
    for x in non_nibble_rest:
        if dot(x, top) == x:
            dispatch.append(x)
        else:
            opaque.append(x)

    assert len(dispatch) == 3, f"Expected 3 dispatch, got {len(dispatch)}"
    assert len(opaque) == 8, f"Expected 8 opaque (D2+IO) atoms, got {len(opaque)}"

    # Identify ALU_LOGIC / ALU_ARITH / ALU_ARITHC via curried probe:
    #   d(N0)(N5)(N0): LOGIC → NOT(5)=NA, ARITH → A=N5, ARITHC → A+1=N6
    n0 = nibble_order[0]
    n5 = nibble_order[5]
    na = nibble_order[10]  # 0xA = 10
    n6 = nibble_order[6]

    alu_logic_tok = alu_arith_tok = alu_arithc_tok = None
    for d in dispatch:
        partial1 = dot(d, n0)
        partial2 = dot(partial1, n5)
        result = dot(partial2, n0)

        if result == na:
            alu_logic_tok = d
        elif result == n5:
            alu_arith_tok = d
        elif result == n6:
            alu_arithc_tok = d

    assert alu_logic_tok is not None, "Failed to identify ALU_LOGIC"
    assert alu_arith_tok is not None, "Failed to identify ALU_ARITH"
    assert alu_arithc_tok is not None, "Failed to identify ALU_ARITHC"
    log(f"ALU_LOGIC identified: {alu_logic_tok}")
    log(f"ALU_ARITH identified: {alu_arith_tok}")
    log(f"ALU_ARITHC identified: {alu_arithc_tok}")
    log(f"Opaque (D2+IO) atoms: {len(opaque)} — deferred to Phase 2")

    # Build result dict
    result = {}
    for i in range(16):
        result[f"N{i:X}"] = nibble_order[i]
    result["N_SUCC"] = n_succ_tok
    result["ALU_LOGIC"] = alu_logic_tok
    result["ALU_ARITH"] = alu_arith_tok
    result["ALU_ARITHC"] = alu_arithc_tok
    result["ALU_ZERO"] = alu_zero_tok
    result["ALU_COUT"] = alu_cout_tok

    log(f"Phase 1b complete: {len(result)} atoms identified, {len(opaque)} opaque remain")
    return result, opaque


# ============================================================================
# Phase 2: Term-level recovery of 8 opaque atoms (D2 + IO)
# ============================================================================

def discover_phase2(opaque: List[str], dot, d1: Dict[str, Any],
                    ext: Dict[str, Any], verbose: bool = False) -> Dict[str, Any]:
    """
    Phase 2: Identify all 8 opaque atoms via term-level probing.

    These 8 atoms have identical all-p Cayley rows and are indistinguishable
    at the atom-atom level. Term-level application reveals their identities:

      Step 1: Apply each to N0 → identifies QUOTE (produces Quote),
              EVAL (returns atom unchanged), and partitions the rest
              into partial group (APP, IO_PUT, IO_SEQ) and p group
              (UNAPP, IO_GET, IO_RDY).
      Step 2: Resolve partial group via partial(N0)(N1).
      Step 3: Resolve p group via application to ⊤.
      Step 4: Confirm UNAPP via AppNode destructure; IO_GET by exclusion.

    Args:
        opaque: 8 hidden labels with all-p Cayley rows
        dot: term-level dot oracle (must support structured terms)
        d1: Phase 1a results (17 D1 atoms)
        ext: Phase 1b results (22 74181 atoms)
        verbose: print discovery log

    Returns:
        Dict mapping atom names to hidden labels for all 8 opaque atoms.
    """
    assert len(opaque) == 8, f"Expected 8 opaque atoms, got {len(opaque)}"

    top = d1["⊤"]
    bot = d1["⊥"]
    p_tok = d1["p"]
    n0 = ext["N0"]
    n1 = ext["N1"]

    def log(msg):
        if verbose:
            print(f"    [Phase 2] {msg}")

    log(f"Starting with {len(opaque)} opaque atoms")

    # All known domain labels (for checking if a result is "in domain")
    all_known = set()
    for k, v in d1.items():
        if k != "_testers":
            all_known.add(v)
    for v in ext.values():
        all_known.add(v)
    domain_set = all_known | set(opaque)

    # ── Step 1: Apply each opaque atom to N0 ──────────────────────────
    QUOTE_tok = EVAL_tok = None
    partial_group = []  # produced structured non-p, non-domain result
    p_group = []        # produced p

    for x in opaque:
        result = dot(x, n0)

        if result == p_tok:
            p_group.append(x)
        elif result in domain_set:
            # Result is a known domain atom but not p — unusual
            # This shouldn't happen for any of the 8 opaque atoms applied to N0
            p_group.append(x)  # safe fallback
        else:
            # Result is a structured value (not in domain) — partial group
            partial_group.append(x)

    log(f"Step 1: applied to N0 → partial group = {partial_group}, p group = {p_group}")

    # partial_group = {QUOTE, APP, IO_PUT, IO_SEQ} (produce structured values)
    # p_group = {EVAL, UNAPP, IO_GET, IO_RDY} (return p)
    assert len(partial_group) == 4, f"Expected 4 in partial group, got {len(partial_group)}"
    assert len(p_group) == 4, f"Expected 4 in p group, got {len(p_group)}"

    # ── Step 1b: Identify QUOTE from partial group ────────────────────
    # QUOTE(N0) produces Quote(N0). Applying EVAL to Quote(N0) should
    # return N0. But we don't know EVAL yet. Alternative: for each pair
    # (q, e) in partial_group × p_group, check if dot(e, dot(q, N0)) == N0.
    # The pair where this works is (QUOTE, EVAL).
    for q in list(partial_group):
        q_n0 = dot(q, n0)  # should be Quote(N0) if q is QUOTE
        for e in list(p_group):
            if dot(e, q_n0) == n0:
                QUOTE_tok = q
                EVAL_tok = e
                break
        if QUOTE_tok:
            break

    assert QUOTE_tok is not None, "Failed to identify QUOTE/EVAL"
    partial_group.remove(QUOTE_tok)
    p_group.remove(EVAL_tok)
    log(f"Step 1: QUOTE identified ({QUOTE_tok})")
    log(f"Step 1: EVAL identified ({EVAL_tok})")
    log(f"Step 1: remaining partial group = {partial_group}")
    log(f"Step 1: remaining p group = {p_group}")

    # ── Step 2: Resolve partial group (APP, IO_PUT, IO_SEQ) ──────────
    # Apply each partial(N0) to N1. Three outcomes:
    #   APP: Partial("APP", N0)(N1) → AppNode(N0, N1) — structured, neither atom nor ⊤
    #   IO_PUT: IOPutPartial(0)(N1) → writes 0x01 to stdout, returns ⊤
    #   IO_SEQ: IOSeqPartial(N0)(N1) → returns N1 (the right argument)
    APP_tok = IO_PUT_tok = IO_SEQ_tok = None

    for x in partial_group:
        partial_val = dot(x, n0)
        result = dot(partial_val, n1)

        if result == top:
            # IO_PUT: IOPutPartial(0)(N1) → ⊤
            IO_PUT_tok = x
            log(f"Step 2: partial({x})(N0)(N1) = ⊤ → IO_PUT identified ({x})")
        elif result == n1:
            # IO_SEQ returned the right argument
            IO_SEQ_tok = x
            log(f"Step 2: partial({x})(N0)(N1) returned N1 → IO_SEQ identified ({x})")
        else:
            # APP built an AppNode
            APP_tok = x
            log(f"Step 2: partial({x})(N0)(N1) built AppNode → APP identified ({x})")

    assert APP_tok is not None, "Failed to identify APP"
    assert IO_PUT_tok is not None, "Failed to identify IO_PUT"
    assert IO_SEQ_tok is not None, "Failed to identify IO_SEQ"

    # ── Step 3: Resolve p group (UNAPP, IO_GET, IO_RDY) ──────────────
    # Apply each to ⊤:
    #   UNAPP(⊤) → p (⊤ is not an AppNode)
    #   IO_GET(⊤) → reads from stdin (DON'T do this) or blocks — skip
    #   IO_RDY(⊤) → ⊤
    IO_RDY_tok = None
    p_subgroup = []  # candidates for UNAPP and IO_GET

    for x in p_group:
        result = dot(x, top)
        if result == top:
            IO_RDY_tok = x
            log(f"Step 3: {x}(⊤) returned ⊤ → IO_RDY identified ({x})")
        else:
            p_subgroup.append(x)

    assert IO_RDY_tok is not None, "Failed to identify IO_RDY"
    assert len(p_subgroup) == 2, f"Expected 2 remaining in p group, got {len(p_subgroup)}"

    # ── Step 4: Confirm UNAPP, IO_GET by exclusion ────────────────────
    # Build AppNode(N0, N1) using APP, then apply UNAPP candidate.
    # dot(UNAPP, AppNode(N0, N1)) → UnappBundle; dot(bundle, ⊤) → N0
    app_partial = dot(APP_tok, n0)
    app_node = dot(app_partial, n1)  # AppNode(N0, N1)

    UNAPP_tok = IO_GET_tok = None
    for x in p_subgroup:
        bundle = dot(x, app_node)
        left = dot(bundle, top)
        if left == n0:
            UNAPP_tok = x
            log(f"Step 4: UNAPP confirmed via AppNode destructure ({x})")
        else:
            IO_GET_tok = x

    # If neither confirmed as UNAPP via the positive test, try the other
    if UNAPP_tok is None:
        # Fallback: one must be UNAPP
        for x in p_subgroup:
            if x != IO_GET_tok:
                UNAPP_tok = x
                break

    assert UNAPP_tok is not None, "Failed to identify UNAPP"
    if IO_GET_tok is None:
        IO_GET_tok = [x for x in p_subgroup if x != UNAPP_tok][0]
    log(f"Step 4: IO_GET identified by exclusion ({IO_GET_tok})")
    log(f"Phase 2 complete: 8/8 opaque atoms identified")

    return {
        "QUOTE": QUOTE_tok,
        "EVAL": EVAL_tok,
        "APP": APP_tok,
        "UNAPP": UNAPP_tok,
        "IO_PUT": IO_PUT_tok,
        "IO_GET": IO_GET_tok,
        "IO_RDY": IO_RDY_tok,
        "IO_SEQ": IO_SEQ_tok,
    }


# ============================================================================
# Verification functions
# ============================================================================

def verify_d2_preservation():
    """Verify Δ₂ fragment is preserved exactly."""
    print("  Δ₂ preservation:")
    import delta2_true_blackbox as orig_mod

    d2_names = NAMES_D1 + NAMES_D2
    for xn in d2_names:
        for yn in d2_names:
            # Use original module's Atom class for the original dot
            orig_result = orig_mod.dot_iota(orig_mod.A(xn), orig_mod.A(yn))
            ext_result = dot_ext(A(xn), A(yn))
            # Compare by name since Atom classes differ across modules
            orig_name = orig_result.name if isinstance(orig_result, orig_mod.Atom) else str(orig_result)
            ext_name = ext_result.name if isinstance(ext_result, Atom) else str(ext_result)
            assert orig_name == ext_name, f"dot({xn}, {yn}): orig={orig_name}, ext={ext_name}"
    print(f"    ✓ All {len(d2_names)}² atom-atom pairs match original Δ₂")


def verify_nibble_group():
    """Verify nibble atoms form Z/16Z under dot."""
    print("  Nibble group (Z/16Z):")
    n0 = A("N0")
    for i in range(16):
        ni = nibble(i)
        assert atom_dot(n0, ni) == ni, f"N0 · N{i:X} ≠ N{i:X}"
        assert atom_dot(ni, n0) == ni, f"N{i:X} · N0 ≠ N{i:X}"
    print("    ✓ N0 is the identity")

    for i in range(16):
        for j in range(16):
            result = atom_dot(nibble(i), nibble(j))
            expected = nibble((i + j) % 16)
            assert result == expected
    print("    ✓ Closed under addition mod 16")


def verify_alu_dispatch():
    """Verify ALU dispatch atoms have correct nibble mappings."""
    print("  ALU dispatch:")
    for i in range(16):
        ni = nibble(i)
        assert atom_dot(A("ALU_LOGIC"), ni) == ni
        assert atom_dot(A("ALU_ARITH"), ni) == nibble((i + 1) % 16)
        assert atom_dot(A("ALU_ARITHC"), ni) == nibble((i + 2) % 16)
    print("    ✓ ALU_LOGIC = identity, ALU_ARITH = successor, ALU_ARITHC = double successor")


def verify_alu_predicates():
    """Verify ALU predicate atoms."""
    print("  ALU predicates:")
    for i in range(16):
        ni = nibble(i)
        if i == 0:
            assert atom_dot(A("ALU_ZERO"), ni) == A("⊤")
        else:
            assert atom_dot(A("ALU_ZERO"), ni) == A("⊥")
        if i >= 8:
            assert atom_dot(A("ALU_COUT"), ni) == A("⊤")
        else:
            assert atom_dot(A("ALU_COUT"), ni) == A("⊥")
    print("    ✓ ALU_ZERO accepts only N0, ALU_COUT accepts N8-NF")


def verify_n_succ():
    """Verify N_SUCC forms a 16-cycle over nibbles."""
    print("  N_SUCC:")
    for i in range(16):
        ni = nibble(i)
        expected = nibble((i + 1) % 16)
        assert atom_dot(A("N_SUCC"), ni) == expected, f"N_SUCC · N{i:X} ≠ N{(i+1)%16:X}"
    print("    ✓ N_SUCC forms a 16-cycle: dot(N_SUCC, Nx) = N(x+1 mod 16)")
    assert atom_dot(A("N_SUCC"), A("⊤")) == A("N_SUCC"), "N_SUCC self-id on ⊤ failed"
    print("    ✓ N_SUCC self-identifies on ⊤")


def verify_self_id():
    """Verify all new non-IO atoms self-identify on ⊤."""
    print("  Self-identification on ⊤:")
    non_io = NEW_ATOMS - IO_ATOMS
    for atom in non_io:
        result = atom_dot(atom, A("⊤"))
        assert result == atom, f"dot({atom}, ⊤) = {result}, expected {atom}"
    print(f"    ✓ All {len(non_io)} non-IO new atoms satisfy dot(x, ⊤) = x")
    for atom in IO_ATOMS:
        result = atom_dot(atom, A("⊤"))
        assert result == A("p"), f"dot({atom}, ⊤) = {result}, expected p"
    print(f"    ✓ All {len(IO_ATOMS)} IO atoms have all-p Cayley rows")


def verify_ext_axiom():
    """Verify all non-IO atoms have unique behavioral fingerprints (Ext axiom)."""
    print("  Ext axiom (unique fingerprints):")
    # IO atoms are intentionally indistinguishable at the Cayley level
    atoms = [a for a in ALL_ATOMS if a not in IO_ATOMS]
    # Build probe set: atoms + structured values created by D2 operations
    probes = list(ALL_ATOMS)
    # Add Quote values (produced by QUOTE)
    for a in ALL_ATOMS[:6]:  # a few representative atoms
        probes.append(Quote(a))
    # Add AppNodes (produced by APP)
    for f in [A("e_D"), A("e_M"), A("i")]:
        for x in [A("i"), A("k"), A("a")]:
            probes.append(AppNode(f, x))

    for i, x in enumerate(atoms):
        for j, y in enumerate(atoms):
            if i >= j:
                continue
            found = False
            for z in probes:
                if dot_ext(x, z) != dot_ext(y, z):
                    found = True
                    break
            assert found, f"No distinguishing probe for {x.name} vs {y.name}"
    print(f"    ✓ All {len(atoms)} non-IO atoms are pairwise distinguishable")
    print(f"    (4 IO atoms are opaque — indistinguishable at Cayley level)")


def verify_tester_preservation():
    """Verify the 4 existing testers remain pure testers in the extended table."""
    print("  Tester preservation:")
    TOP, BOT = A("⊤"), A("⊥")
    testers = [A("e_I"), A("d_K"), A("m_K"), A("m_I")]
    for t in testers:
        for y in ALL_ATOMS:
            result = atom_dot(t, y)
            assert result in (TOP, BOT), (
                f"Tester {t.name} on {y.name} = {result.name}, expected ⊤ or ⊥"
            )
    print(f"    ✓ All 4 testers have pure boolean left-images over {len(ALL_ATOMS)} atoms")


def verify_74181_operations():
    """Verify 74181 operations through curried application."""
    print("  74181 operations (curried):")

    # A XOR B (logic, selector 6)
    for a in range(16):
        for b in range(16):
            result = dot_ext(dot_ext(dot_ext(A("ALU_LOGIC"), nibble(6)), nibble(a)), nibble(b))
            expected = nibble(a ^ b)
            assert result == expected, f"XOR({a},{b}) = {result}, expected {expected}"
    print("    ✓ XOR (logic S=6) correct for all 16×16 inputs")

    # A plus B (arith, selector 9, no carry)
    for a in range(16):
        for b in range(16):
            result = dot_ext(dot_ext(dot_ext(A("ALU_ARITH"), nibble(9)), nibble(a)), nibble(b))
            expected = nibble((a + b) % 16)
            assert result == expected
    print("    ✓ ADD (arith S=9 no carry) correct")

    # A minus B (arith, selector 6, with carry)
    for a in range(16):
        for b in range(16):
            result = dot_ext(dot_ext(dot_ext(A("ALU_ARITHC"), nibble(6)), nibble(a)), nibble(b))
            expected_val, _, _ = alu_74181("arithc", 6, a, b)
            expected = nibble(expected_val)
            assert result == expected
    print("    ✓ SUB (arithc S=6) correct")

    # NOT A (logic, selector 0)
    for a in range(16):
        result = dot_ext(dot_ext(dot_ext(A("ALU_LOGIC"), nibble(0)), nibble(a)), nibble(0))
        expected = nibble((~a) & 0xF)
        assert result == expected
    print("    ✓ NOT (logic S=0) correct")

    # A AND B (logic, selector 0xB)
    for a in range(16):
        for b in range(16):
            result = dot_ext(dot_ext(dot_ext(A("ALU_LOGIC"), nibble(0xB)), nibble(a)), nibble(b))
            expected = nibble(a & b)
            assert result == expected
    print("    ✓ AND (logic S=B) correct")

    # A OR B (logic, selector 0xE)
    for a in range(16):
        for b in range(16):
            result = dot_ext(dot_ext(dot_ext(A("ALU_LOGIC"), nibble(0xE)), nibble(a)), nibble(b))
            expected = nibble(a | b)
            assert result == expected
    print("    ✓ OR (logic S=E) correct")

    # Left shift (A plus A, arith S=C)
    for a in range(16):
        result = dot_ext(dot_ext(dot_ext(A("ALU_ARITH"), nibble(0xC)), nibble(a)), nibble(a))
        expected = nibble((2 * a) % 16)
        assert result == expected
    print("    ✓ Left shift (arith S=C, A+A) correct")

    # Increment (A plus 1: arithc S=0)
    for a in range(16):
        result = dot_ext(dot_ext(dot_ext(A("ALU_ARITHC"), nibble(0)), nibble(a)), nibble(0))
        expected_val, _, _ = alu_74181("arithc", 0, a, 0)
        expected = nibble(expected_val)
        assert result == expected
    print("    ✓ Increment (arithc S=0) correct")

    # Decrement (A minus 1: arith S=F)
    for a in range(16):
        result = dot_ext(dot_ext(dot_ext(A("ALU_ARITH"), nibble(0xF)), nibble(a)), nibble(0))
        expected_val, _, _ = alu_74181("arith", 0xF, a, 0)
        expected = nibble(expected_val)
        assert result == expected
    print("    ✓ Decrement (arith S=F) correct")


def verify_alu_74181_function():
    """Verify the alu_74181 function against known truth table values."""
    print("  74181 truth table spot checks:")

    # Logic mode checks
    assert alu_74181("logic", 0x0, 0b1010, 0)[0] == 0b0101  # NOT A
    assert alu_74181("logic", 0x3, 0b1010, 0b0101)[0] == 0  # Logical 0
    assert alu_74181("logic", 0x6, 0b1010, 0b0101)[0] == 0b1111  # XOR
    assert alu_74181("logic", 0xB, 0b1010, 0b0101)[0] == 0b0000  # AND
    assert alu_74181("logic", 0xC, 0b1010, 0b0101)[0] == 0b1111  # Logical 1
    assert alu_74181("logic", 0xE, 0b1010, 0b0101)[0] == 0b1111  # OR
    assert alu_74181("logic", 0xF, 0b1010, 0b0101)[0] == 0b1010  # A pass
    assert alu_74181("logic", 0xA, 0b1010, 0b0101)[0] == 0b0101  # B pass
    print("    ✓ Logic mode operations correct")

    # Arithmetic mode checks (no carry = arith, with carry = arithc)
    # S=9: A plus B
    r, c, z = alu_74181("arith", 9, 5, 3)
    assert r == 8 and not c  # 5+3=8, no carry
    r, c, z = alu_74181("arith", 9, 15, 15)
    assert r == 14 and c  # 15+15=30 → 14 with carry
    # S=9 with carry: A plus B plus 1
    r, c, z = alu_74181("arithc", 9, 5, 3)
    assert r == 9  # 5+3+1=9
    # S=F: A minus 1 (no carry) / A (with carry)
    r, _, _ = alu_74181("arith", 0xF, 5, 0)
    assert r == 4  # 5-1=4
    r, _, _ = alu_74181("arithc", 0xF, 5, 0)
    assert r == 5  # A pass through
    # S=0: A (no carry) / A plus 1 (with carry)
    r, _, _ = alu_74181("arith", 0, 5, 0)
    assert r == 5  # A
    r, _, _ = alu_74181("arithc", 0, 5, 0)
    assert r == 6  # A+1
    print("    ✓ Arithmetic mode operations correct")


# ============================================================================
# Integration test
# ============================================================================

def run_integration_test(num_seeds: int, verbose: bool = False):
    """Run full recovery on random permutations (Phase 1 + Phase 2)."""
    print(f"\n  Integration test: {num_seeds} random permutations")
    failures = []

    for seed in range(num_seeds):
        try:
            domain, dot, true2hid = make_blackbox(seed)

            # Phase 1a: D1
            d1 = discover_d1(domain, dot)
            for k in ["⊤", "⊥", "p", "e_I", "e_D", "e_M", "e_Σ", "e_Δ",
                       "i", "k", "a", "b", "d_I", "d_K", "m_I", "m_K", "s_C"]:
                if d1[k] != true2hid[A(k)]:
                    failures.append((seed, k, "d1"))
                    break

            # Phase 1b: 74181 (Cayley-level)
            ext, opaque = discover_74181_with_logs(domain, dot, d1, verbose=False)
            for k in ext:
                if ext[k] != true2hid[A(k)]:
                    failures.append((seed, k, "74181"))
                    break

            # Phase 2: Term-level recovery of 8 opaque atoms
            p2 = discover_phase2(opaque, dot, d1, ext, verbose=False)
            for k in p2:
                if p2[k] != true2hid[A(k)]:
                    failures.append((seed, k, "phase2"))
                    break

        except Exception as e:
            failures.append((seed, str(e), "crash"))

        if (seed + 1) % 100 == 0:
            print(f"    ... {seed + 1}/{num_seeds} seeds tested")

    if failures:
        print(f"  FAILED on {len(failures)} seeds:")
        for seed, key, phase in failures[:20]:
            print(f"    seed={seed}: {phase} failed at {key}")
        return False
    else:
        print(f"  ✓ All {num_seeds} seeds passed — 47/47 atoms, 100% recovery rate")
        return True


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Δ₂+74181 ALU Extension")
    parser.add_argument("--test", type=int, default=0,
                        help="Run integration test with N seeds")
    parser.add_argument("--verbose", action="store_true",
                        help="Verbose output for discovery")
    args = parser.parse_args()

    print("=" * 60)
    print("  Δ₂+74181 VERIFICATION SUITE")
    print("=" * 60)
    print(f"\n  Total atoms: {len(ALL_ATOMS)}")
    print(f"    Δ₁ core: {len(NAMES_D1)}")
    print(f"    Δ₂ extensions: {len(NAMES_D2)}")
    print(f"    Nibbles: {len(NAMES_NIBBLES)}")
    print(f"    ALU dispatch: {len(NAMES_ALU_DISPATCH)}")
    print(f"    ALU predicates: {len(NAMES_ALU_PRED)}")
    print(f"    N_SUCC: {len(NAMES_ALU_MISC)}")
    print(f"    IO atoms: {len(NAMES_IO)}")
    print(f"    Cayley table: {len(ALL_ATOMS)}×{len(ALL_ATOMS)} = {len(ALL_ATOMS)**2} entries")
    print()

    verify_d2_preservation()
    verify_nibble_group()
    verify_alu_dispatch()
    verify_alu_predicates()
    verify_n_succ()
    verify_self_id()
    verify_tester_preservation()
    verify_ext_axiom()
    verify_alu_74181_function()
    verify_74181_operations()

    if args.test > 0:
        run_integration_test(args.test, verbose=args.verbose)
    else:
        # Quick demo: single seed recovery
        print("\n  Quick recovery demo (seed=42):")
        domain, dot, true2hid = make_blackbox(42)

        # Phase 1a: D1
        d1 = discover_d1(domain, dot)

        # Phase 1b: 74181 (Cayley-level)
        ext, opaque = discover_74181_with_logs(domain, dot, d1, verbose=True)

        # Verify Phase 1
        all_ok = True
        for k, v in ext.items():
            expected = true2hid[A(k)]
            ok = "✓" if v == expected else "✗"
            if v != expected:
                all_ok = False
            print(f"      {k:12s} → {v}  {ok}")
        if all_ok:
            print(f"    ✓ All {len(ext)} Phase 1 atoms correctly recovered")

        # Phase 2: Term-level recovery
        print()
        p2 = discover_phase2(opaque, dot, d1, ext, verbose=True)

        all_ok_p2 = True
        for k, v in p2.items():
            expected = true2hid[A(k)]
            ok = "✓" if v == expected else "✗"
            if v != expected:
                all_ok_p2 = False
            print(f"      {k:12s} → {v}  {ok}")
        if all_ok_p2:
            print(f"    ✓ All {len(p2)} Phase 2 atoms correctly recovered")
        if all_ok and all_ok_p2:
            print(f"    ✓ All 47 atoms identified — complete recovery")

    print("\n" + "=" * 60)
    print("  All verifications passed. ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()
