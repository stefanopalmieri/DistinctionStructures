"""
Coordinate-Free Hello World

Demonstrates coordinate-free programs on the fingerprint-addressed Cayley ROM.
Programs use canonical atom names — no indices, no scanner, no cache.

Usage:
    uv run python -m examples.coordinate_free_demo
"""

import sys
import os

# Allow running from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from emulator.coordinate_free import CoordinateFreeProgram, InvariantLoader, run_coordinate_free


# The program: print "Hi" via IO_PUT + IO_SEQ
# Uses ONLY canonical atom names — no indices.
PROGRAM = CoordinateFreeProgram(
    term=(("IO_SEQ", (("IO_PUT", "N4"), "N8")),     # H = 0x48
          (("IO_PUT", "N6"), "N9")),                  # i = 0x69
    name="hello",
)


def main():
    print("=" * 60)
    print("  Coordinate-Free Program Demo")
    print("=" * 60)
    print()
    print(f"  Program: {PROGRAM.name!r}")
    print(f"  Atoms used: {sorted(PROGRAM.atom_names())}")
    print()

    result, host = run_coordinate_free(PROGRAM)
    output = host.uart_recv()
    ok = "OK" if result["ok"] else "FAIL"
    print(f"  UART: {output!r}  status: {ok}")
    print(f"  ROM reads: {host.machine.rom_reads}")

    print()
    if output == b"Hi":
        print("  Coordinate-free programs work on the fingerprint-addressed ROM.")
    else:
        print("  ERROR: unexpected output!")
        sys.exit(1)
    print()
    print("=" * 60)


if __name__ == "__main__":
    main()
