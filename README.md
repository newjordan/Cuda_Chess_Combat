# Cuda_Chess_Combat

A lightweight, GPU-resident chess arbiter for [chessagents.ai](https://chessagents.ai) tournaments. Drop-in replacement for the rule-enforcement layer of the production AgentChess `match-processor`, with all chess compute (legality, terminal detection, make/unmake) executing on NVIDIA hardware via `librefcuda.so`.

## Status

- **Accuracy:** 1.0000 referee parity vs. the production AgentChess CPU arbiter on a 333-game random sample (σ = 0.0000), and 7/7 on a targeted re-replay of every game where the prior implementation diverged. See [`combat_shipping/results/ACCURACY_REPORT_2026-04-29.md`](combat_shipping/results/ACCURACY_REPORT_2026-04-29.md) for the full 37,236-game shadow run, root-cause analysis of the original divergences, and the fix.
- **Live wall-clock parity** with the CPU arbiter at 1.004 × ratio under matched fighter resources (4 / 4 games match move-for-move).
- **Per-call referee throughput:** ~57 ms / game replay (vs. ~2 ms / game CPU). The gap is FFI-per-call overhead; the batched ABI in [`COMBAT.md`](COMBAT.md) is the documented path to ~50 × recovery. End-to-end games-per-hour is dominated by the per-move fighter budget (5500 ms cap) and unaffected by referee choice.

## Repository layout

```
Cuda_Chess_Combat/
├── combat_shipping/         the arbiter + validation harness
│   ├── live_arbiter/        live game serving (GPU referee + docker fighters)
│   ├── bridges/             CPU and GPU referee adapters used by the harness
│   ├── live_compare/        head-to-head N-pair runner (CPU vs GPU)
│   ├── harness/             back-catalog replay parity scanner
│   └── results/             accuracy report + run output dir
├── cuda/engine/             CUDA chess engine + librefcuda.so source
└── rust/
    ├── dojo-ref/            Rust FFI binding to librefcuda.so
    └── dojo-ref-py/         PyO3 Python bindings (the `dojo_ref` module)
```

## Prerequisites

- NVIDIA GPU with CUDA toolkit (tested on H100; nvcc 12.x).
- Rust toolchain (1.75+) with `cargo` and [`maturin`](https://www.maturin.rs/) (`pip install maturin`).
- Node.js 18+.
- Docker (for the live arbiter's fighter sandbox).
- The production AgentChess arbiter source — clone [`jaymaart/chess-agents-arbiter`](https://github.com/jaymaart/chess-agents-arbiter) (or your fork) somewhere and point `ARBITER_SRC` at its `src/` directory. The bridges import `chess-engine.js` from there as the rule-set source of truth; nothing in this repo modifies it.

## Build

```bash
# CUDA engine + librefcuda.so
cd cuda/engine && make engine librefcuda

# Python bindings
cd ../../rust/dojo-ref-py && maturin develop --release

# Verify
python3 -c "import dojo_ref; print(dojo_ref.Position.startpos().legal_moves()[:3])"
# -> ['a2a3', 'a2a4', 'b2b3']
```

## Run

### Live game (single match, GPU referee)

```bash
export ARBITER_SRC=/path/to/chess-agents-arbiter/src

python3 combat_shipping/live_arbiter/live_match.py \
    --white  /path/to/white.js \
    --black  /path/to/black.js \
    --max-plies 500 \
    --move-timeout-ms 5500
```

### Live head-to-head (this arbiter vs. CPU arbiter, N pairs)

```bash
export ARBITER_SRC=/path/to/chess-agents-arbiter/src
export FIGHTERS_DIR=/path/to/match-processor/data    # directory of fighter .js / .py blobs
# Optional: lift docker caps symmetrically on both sides.
export AGENT_CPUS=4
export AGENT_MEMORY=1g

bash combat_shipping/live_compare/run.sh --n 8 --max-plies 120
```

### Replay parity (back-catalog scan against a corpus of PGNs)

```bash
export ARBITER_SRC=/path/to/chess-agents-arbiter/src
export PGN_CORPUS=/path/to/games.pgn

bash combat_shipping/harness/run_comparison.sh --n 1000
```

## Architecture in one paragraph

`dojo_ref.Position` is the GPU referee — every legal-move generation, every make/unmake, every check / mate / stalemate verdict is a `librefcuda.so` call. Host-side Python (`combat_shipping/live_arbiter/gpu_state.py:HostState`) maintains the small ancillary state the GPU ABI does not expose (castling rights, en-passant target, halfmove clock, threefold repetition history, insufficient-material predicate) using bookkeeping logic that mirrors `chess-engine.js` byte-for-byte. The replay bridge (`combat_shipping/bridges/cuda_referee.py`) reuses that same `HostState` so the threefold / 50-move / insufficient-material checks are exact-equivalent to the production arbiter.

## License

Apache-2.0. See [LICENSE](LICENSE).

## Acknowledgements

The CPU referee (`chess-engine.js`) imported by the comparison bridges is the production AgentChess arbiter source by [@jaymaart](https://github.com/jaymaart) at [`chess-agents-arbiter`](https://github.com/jaymaart/chess-agents-arbiter), used unmodified.
