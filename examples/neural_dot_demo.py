"""
Neural Cayley Table Demo

Trains an MLP to memorize the Kamea's 66x66 Cayley table, then runs
programs on the neural machine. Sweeps hidden dimensions to find the
minimum network size for perfect accuracy.

Usage:
    uv run python -m examples.neural_dot_demo
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from emulator.neural_dot import NeuralCayleyTable
from emulator.neural_machine import NeuralKameaMachine
from emulator.host import EmulatorHost
from emulator.fingerprint import NUM_FP


def run_hello_world_neural(table: NeuralCayleyTable) -> tuple[bytes, dict]:
    """Run 'Hi' program on the neural machine."""
    machine = NeuralKameaMachine(table)
    host = EmulatorHost.__new__(EmulatorHost)
    host.machine = machine

    r = host.eval(
        (("IO_SEQ", (("IO_PUT", "N4"), "N8")),
         (("IO_PUT", "N6"), "N9"))
    )
    output = host.uart_recv()
    return output, r


def run_hello_world_rom() -> bytes:
    """Run 'Hi' program on the ROM-based machine."""
    host = EmulatorHost()
    r = host.eval(
        (("IO_SEQ", (("IO_PUT", "N4"), "N8")),
         (("IO_PUT", "N6"), "N9"))
    )
    return host.uart_recv()


def train_and_report(hidden_dim: int, max_epochs: int = 10000,
                     lr: float = 1e-3, retries: int = 1) -> dict:
    """Train at a given hidden dim, best of retries attempts."""
    best = None
    for _ in range(retries):
        table = NeuralCayleyTable(hidden_dim=hidden_dim, device="cpu")
        stats = table.train(epochs=max_epochs, lr=lr, target_accuracy=1.0)
        acc, correct, total = table.accuracy()
        result = {
            "hidden_dim": hidden_dim,
            "parameters": stats["parameters"],
            "epochs": stats["epochs"],
            "accuracy": acc,
            "correct": correct,
            "total": total,
            "time": stats["training_time"],
            "table": table,
        }
        if best is None or correct > best["correct"]:
            best = result
        if correct == total:
            break
    return best


def main():
    print("=" * 70)
    print("  Neural Cayley Table — MLP Implementation of dot")
    print("=" * 70)
    print()
    print(f"  Table size: {NUM_FP}x{NUM_FP} = {NUM_FP * NUM_FP} entries")
    print()

    # ── Phase 1: Train at default size and verify ──────────────────────

    print("  Phase 1: Training (hidden_dim=128)")
    print("  " + "-" * 50)

    table_128 = NeuralCayleyTable(hidden_dim=128, device="cpu")
    stats = table_128.train(epochs=5000, lr=1e-3, target_accuracy=1.0,
                            print_every=500)
    acc, correct, total = table_128.accuracy()

    print(f"\n  Results:")
    print(f"    Parameters:  {stats['parameters']:,}")
    print(f"    Epochs:      {stats['epochs']}")
    print(f"    Accuracy:    {acc:.4f} ({correct}/{total})")
    print(f"    Time:        {stats['training_time']:.2f}s")
    print(f"    Compression: {table_128.compression_ratio:.3f} "
          f"(table/params)")
    print()

    # ── Phase 2: Run hello world on neural machine ─────────────────────

    print("  Phase 2: Neural Machine — Hello World")
    print("  " + "-" * 50)

    rom_output = run_hello_world_rom()
    neural_output, neural_result = run_hello_world_neural(table_128)

    print(f"    ROM machine:    {rom_output!r}")
    print(f"    Neural machine: {neural_output!r}")
    print(f"    Match: {'YES' if rom_output == neural_output else 'NO'}")
    if neural_result["ok"]:
        print(f"    Neural dot calls: "
              f"{neural_result['stats'].get('neural_dot_calls', '?')}")

    if rom_output != neural_output:
        print("  ERROR: Neural machine output differs from ROM!")
        sys.exit(1)

    # Test with programs that hit the Cayley table
    machine = NeuralKameaMachine(table_128)
    host = EmulatorHost.__new__(EmulatorHost)
    host.machine = machine

    # Absorber: (top bot) = top — hits Cayley dispatch
    r = host.eval(("⊤", "⊥"))
    assert r["ok"] and r["result"] == "⊤", f"Absorber failed: {r}"
    print(f"    (⊤ ⊥) = {r['result']}  "
          f"[{machine.neural_dot_calls} neural dot calls]")

    # ALU: 3+5=8
    machine.reset_counters()
    r = host.eval(((("ALU_ARITH", "N9"), "N3"), "N5"))
    assert r["ok"] and r["result"] == "N8", f"ALU failed: {r}"
    print(f"    3+5 = {r['result']}  "
          f"[{machine.neural_dot_calls} neural dot calls]")
    print()

    # ── Phase 3: Dimension sweep ───────────────────────────────────────

    print("  Phase 3: Hidden Dimension Sweep")
    print("  " + "-" * 50)
    print()
    print(f"  {'Dim':>6s}  {'Params':>8s}  {'Epochs':>7s}  "
          f"{'Accuracy':>10s}  {'Ratio':>8s}  {'Time':>6s}")
    print(f"  {'---':>6s}  {'---':>8s}  {'---':>7s}  "
          f"{'---':>10s}  {'---':>8s}  {'---':>6s}")

    sweep_results = []
    for dim in [128, 64, 32, 16, 8, 7, 6, 4, 2]:
        epochs = 20000 if dim <= 8 else 10000
        tries = 3 if dim <= 8 else 1
        r = train_and_report(dim, max_epochs=epochs, lr=1e-3, retries=tries)
        ratio = (NUM_FP * NUM_FP) / r["parameters"]
        status = "PERFECT" if r["correct"] == r["total"] else f"{r['correct']}/{r['total']}"
        print(f"  {dim:>6d}  {r['parameters']:>8,}  {r['epochs']:>7d}  "
              f"{status:>10s}  {ratio:>8.3f}  {r['time']:>5.1f}s")
        sweep_results.append(r)

        # Run hello world if perfect
        if r["correct"] == r["total"]:
            out, _ = run_hello_world_neural(r["table"])
            if out != b"Hi":
                print(f"    WARNING: perfect accuracy but hello world "
                      f"failed: {out!r}")

    print()

    # ── Summary ────────────────────────────────────────────────────────

    min_perfect = None
    for r in sweep_results:
        if r["correct"] == r["total"]:
            min_perfect = r

    if min_perfect:
        print(f"  Minimum dimension for 100%: {min_perfect['hidden_dim']}")
        print(f"  Parameters at minimum:      {min_perfect['parameters']:,}")
        ratio = (NUM_FP * NUM_FP) / min_perfect["parameters"]
        if ratio > 1:
            print(f"  Compression ratio:          {ratio:.2f}x "
                  f"(table compresses!)")
        else:
            print(f"  Compression ratio:          {ratio:.3f} "
                  f"(more params than table entries)")
    else:
        print("  No dimension achieved 100% accuracy")

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
