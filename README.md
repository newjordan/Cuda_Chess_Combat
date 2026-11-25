# Cuda Chess Combat

A CUDA chess engine and referee, plus Rust and Python bindings. The rules
layer (move generation, make/unmake, check, and Zobrist hashing) also
compiles with a host C++ compiler, so the test suite runs on a machine
with no GPU and no CUDA toolkit.

Search, evaluation, and the UCI engine binary stay on the CUDA build.
The CPU foreign-function interface returns no search move.

This workspace was tested without `nvcc`. Nothing below claims a GPU
run. Throughput and parity figures in [COMBAT.md](COMBAT.md) and
`combat_shipping/results/` were not remeasured.

## Layout

```
cuda/engine/          engine sources (CUDA, and the same rules files on the host)
cuda/engine/cpu/      host C ABI used by the Rust `cpu` feature
combat_shipping/      live and replay arbiter harness
rust/dojo-ref/        Rust binding
rust/dojo-ref-py/     PyO3 module `dojo_ref`
tests/                Python tests for host bookkeeping
```

The engine mailbox is `a8 = 0` … `h1 = 63`. Piece codes are `0` empty,
`1–6` white pawn through king, `7–12` black. A move packs from (bits
0–5), to (6–11), promotion piece type (12–15), and flags (16–19).
`sizeof(Position)` measured here is 74 bytes.

`combat_shipping/live_arbiter/gpu_state.py` is separate host
bookkeeping (clocks, castling rights, repetition, insufficient
material). Its board index is `a1 = 0`. It talks to the engine in FEN
and UCI, not raw mailbox indices.

## CPU build and tests

Needs `g++` (C++17), `make`, Python 3 with development headers, and
Rust/`cargo`. The run recorded below used g++ 13.3.0, rustc 1.83.0,
and Python 3.12.3. The Rust edition is 2021.

```bash
make test
```

That target:

1. Builds and runs `cuda/engine/build/cpu/cpu_rules_test` (`make -C cuda/engine cpu-test`).
2. Runs `cargo test -p dojo-ref --no-default-features --features cpu`.
3. Type-checks the Python extension against the same CPU backend.
4. Runs `python3 -m unittest discover -s tests -v`.

`make -C cuda/engine cpu-asan` rebuilds that C++ suite with
`-fsanitize=address,undefined`.

### What the C++ suite checks

Perft counts are the published figures from
[chessprogramming.org/Perft_Results](https://www.chessprogramming.org/Perft_Results):

| position | depths |
|---|---|
| startpos | 1–5 |
| Kiwipete | 1–4 |
| Position 3 | 1–5 |
| Position 4 (promotions) | 1–4 |
| Position 5 | 1–4 |
| Position 6 | 1–4 |

On this machine, one `-O2` run reported
`perft_cases=26 perft_passed=26 perft_nodes=16564718 elapsed_ms=685 failures=0`.
The clock starts at the beginning of the process and also covers the
FEN, castling, en-passant, mate, stalemate, unmake, and Zobrist checks
that follow the perft loop. The same binary reported
`unmake_checks=11679 unmake_mismatches=0` and
`zobrist_checks=11679 zobrist_mismatches=0`.

One `-O1` AddressSanitizer and UndefinedBehaviorSanitizer run of the
same program reported `elapsed_ms=2864` and `failures=0`, with the same
26 passing cases and the same node, unmake, and Zobrist counts.

The Rust CPU tests (6, debug profile) finished in 0.01 s as reported
by `cargo test`. The Python bookkeeping tests (8) finished in 0.000 s
as reported by `unittest`.

### Board printer

`make -C cuda/engine cpu-test` also builds `cuda/engine/build/cpu/cpu_inspect`.
Uppercase pieces are white. `a8` is the top-left square.

```bash
cuda/engine/build/cpu/cpu_inspect
cuda/engine/build/cpu/cpu_inspect --play e2e4 e7e5 g1f3
cuda/engine/build/cpu/cpu_inspect "FEN" --play e5d6
```

## Rust

The `dojo-ref` crate defaults to the `gpu` feature, which links
`librefcuda.so` and the CUDA runtime. The `cpu` feature compiles the
rules sources with `g++` and does not need a device. The two features
cannot be enabled together.

```bash
cargo test --manifest-path rust/Cargo.toml -p dojo-ref --no-default-features --features cpu
cargo run  --manifest-path rust/Cargo.toml --no-default-features --features cpu --example smoke
```

`play_game` and `match` call search. On the CPU feature, search returns
no move and those examples stop. They need the default GPU build and
`librefcuda.so` on the linker path:

```bash
cargo run --manifest-path rust/Cargo.toml --release --example play_game
cargo run --manifest-path rust/Cargo.toml --release --example match
```

## Python module

`dojo_ref` is locked to PyO3 0.28.3. That crate's build script accepts
CPython through 3.14, which is the interpreter version that rejected
PyO3 0.24 on the first CI run. It still builds with Rust 1.83.
`maturin` `>=1.7,<2` and Python `>=3.11` are declared in
`rust/dojo-ref-py/pyproject.toml`. Install headers (`python3-dev`)
before building. The local `cargo check` used Python 3.12.3.

```bash
# CPU rules, no CUDA
cd rust/dojo-ref-py
maturin develop --no-default-features --features cpu

# GPU (default). Build librefcuda.so first; see below.
maturin develop --release
```

```python
import dojo_ref
pos = dojo_ref.Position.startpos()
print(pos.legal_moves()[:4])
```

`Engine.search_depth` returns `None` when the module was built with
the CPU feature.

## GPU build

Not run in this workspace (`nvcc` was not installed).

`make` inside `cuda/engine` builds the engine when `nvcc` is on `PATH`,
and builds the CPU tests otherwise. `NVCC` overrides the compiler.
`CUDA_ARCH` overrides the architecture. With `CUDA_ARCH` unset, the
default `CUDA_SINGLE_GPU_TARGET=dgx-spark` selects `sm_121`.
`CUDA_SINGLE_GPU_TARGET=h200` selects `sm_90a`, and `native` passes
`-arch=native`.

```bash
make -C cuda/engine engine librefcuda
make -C cuda/engine engine_test_perft
```

`librefcuda.so` is what the default Rust and Python builds link.

## Live arbiter

The comparison harness is under `combat_shipping/`. A live game needs
the GPU module, Docker, Node.js, fighter programs, and the unmodified
AgentChess arbiter sources. Point `ARBITER_SRC` at that checkout's
`src/` directory.

```bash
export ARBITER_SRC=/path/to/chess-agents-arbiter/src

python3 combat_shipping/live_arbiter/live_match.py \
    --white /path/to/white.js \
    --black /path/to/black.js \
    --max-plies 500 \
    --move-timeout-ms 5500
```

Head-to-head and PGN replay entry points:

```bash
bash combat_shipping/live_compare/run.sh --n 8 --max-plies 120
bash combat_shipping/harness/run_comparison.sh --n 1000
```

`make test` does not run these. They are not part of the CPU path.

## Continuous integration

`.github/workflows/ci.yml` checks out the repo on `ubuntu-latest`,
installs `g++`, `make`, `python3`, and `python3-dev`, installs stable
Rust, and runs `make test`. There is no GPU job.

## License

Apache-2.0. See [LICENSE](LICENSE).

The CPU referee imported by the comparison bridges
(`chess-engine.js`) is the AgentChess arbiter by
[@jaymaart](https://github.com/jaymaart),
[chess-agents-arbiter](https://github.com/jaymaart/chess-agents-arbiter),
used unmodified.
