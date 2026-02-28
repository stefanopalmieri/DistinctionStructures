"""
Fingerprint-Addressed Cayley ROM Demo

The Cayley ROM is addressed and valued entirely in fingerprint space.
No scanner, no cache, no physical indices. The ROM is a fixed canonical
constant — the same bytes forever.

Usage:
    uv run python -m examples.fingerprint_demo
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from emulator.host import EmulatorHost
from emulator.fingerprint import NUM_FP, FP_TOP, FP_BOT, FP_TO_NAME
from emulator import cayley


def main():
    print("=" * 60)
    print("  Fingerprint-Addressed Cayley ROM")
    print("=" * 60)
    print()

    # Show ROM properties
    rom = cayley.build_fingerprint_rom()
    print(f"  ROM size: {len(rom)} bytes ({NUM_FP}x{NUM_FP})")
    print(f"  Absorber check: rom[TOP*66+BOT] = {FP_TO_NAME[rom[FP_TOP * NUM_FP + FP_BOT]]}")
    print()

    # Run programs
    host = EmulatorHost()

    # Absorber
    r = host.eval(("⊤", "⊥"))
    print(f"  (⊤ ⊥) = {r['result']}  [{r['stats']['cycles']} cycles]")

    # ALU
    r = host.eval(((("ALU_ARITH", "N9"), "N3"), "N5"))
    print(f"  (((ALU_ARITH N9) N3) N5) = {r['result']}  [{r['stats']['cycles']} cycles]")

    # IO_PUT
    r = host.eval((("IO_PUT", "N4"), "N8"))
    tx = host.uart_recv()
    print(f"  ((IO_PUT N4) N8) → TX: {tx!r}  [{r['stats']['cycles']} cycles]")

    # IO_SEQ: "Hi"
    r = host.eval(
        (("IO_SEQ", (("IO_PUT", "N4"), "N8")),
         (("IO_PUT", "N6"), "N9"))
    )
    tx = host.uart_recv()
    print(f"  IO_SEQ → TX: {tx!r}  [{r['stats']['cycles']} cycles]")

    # QUOTE/EVAL roundtrip
    r = host.eval(("EVAL", ("QUOTED", "N7")))
    print(f"  (EVAL (QUOTE N7)) = {r['result']}  [{r['stats']['cycles']} cycles]")

    print()
    print(f"  ROM reads: {host.machine.rom_reads}")
    print()
    print("  One ROM. Fingerprint-addressed. No scanner. No cache.")
    print()
    print("=" * 60)


if __name__ == "__main__":
    main()
