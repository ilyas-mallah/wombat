# Wombat

Wombat is a research chess engine built to test one question. Does the same trained network play better when it is
allowed more sequential computation per position? The network is a small transformer whose weights are reused at
every step, so "thinking longer" means running the same core more times on a persistent hidden state. It is written
in PyTorch.

This repository is the research log of a project that is paused. The last experiment ran on 2026-09-27. The central
idea is not proven. Extra steps help up to the number the network was trained for and not beyond, and the best
checkpoint trails DeepMind's published ChessBench 9M model by a wide margin. The failed experiments are recorded in
`docs/plan.md` as plainly as the ones that worked.

## The question

Standard engines buy strength with search. Stockfish runs alpha-beta search over millions of nodes with a small neural
evaluator at the leaves. AlphaZero and Leela run Monte Carlo tree search guided by a policy and value network.
DeepMind's searchless ChessBench 9M transformer has no tree, but it spends one forward pass per legal move.

Wombat has no tree and no per-move passes. It encodes the position once, applies the same core T times to a hidden
state, and reads the move from the final state. Test-time compute is the iteration count T, not the number of nodes.
The question, from the top of `docs/plan.md`: can a fixed-size recurrent network learn an internal chess-solving
process such that giving the same trained weights more sequential computation reliably raises its playing strength?

## How it works

The recipe of the long runs (called v27a, used by L2 and L3):

- **Input.** The board is turned to the side to move and becomes 64 square tokens plus one state token. Each square
  carries hand-computed features: attacker counts, cheapest attacker, pins, check, sole defenders, x-rays, checking
  squares and material. The network therefore does not learn chess from raw pieces alone.
- **Recurrent core.** Two transformer blocks (RMSNorm, multi-head attention, SwiGLU, residual) with the same weights
  at every step. There is no step embedding and no built-in maximum depth. The state is RMS-normalized after each step.
  Attention is local, so each square sees its king-step neighbors. It uses learned direction-aware relative attention,
  a learned 64 by 64 square bias and a bias generated from the state at every step.
- **Readout.** Per-square move logits plus a query-key score between a move's origin and destination tokens. For every
  legal move, a few static facts are computed on the GPU and added with weights the model produces per move: the
  exchange value of the moved piece, the opponent's best capture, mate in one and check. Position and per-move values
  are 64-bin distributions.
- **Size.** Width 384, 8 heads, MLP width 1,536, 5,596,077 parameters.
- **Training.** Each batch runs a random prefix of 0 to 8 steps without gradient, then 8 steps with gradient. A quarter
  of the batches are Lichess puzzle positions, run for 4 steps per remaining solver move. The policy target is a soft
  distribution over every legal move, built from Stockfish win chances. The optimizer is AdamW with EMA weights and a
  final cooldown that scales the learning rate by one minus the square root of the cooldown progress.
- **Data.** Positions from the Lichess evaluation database with every legal move scored by Stockfish 19, plus a
  13.7 million position Lichess puzzle corpus. The first game corpus holds 10,092,005 labelled positions (4.0 million
  at 50,000 nodes and 6.1 million at 20,000 nodes). Splits are made by FEN hash, and the frozen evaluation sets are
  excluded.
- **Hardware.** A home RTX 3070 Ti with 8 GB of memory ran the pilots. A rented RTX 5090 with 32 GB ran the long runs L2
  and L3 and the Stockfish labelling.

Every experiment has its gate in `docs/plan.md`, written before the run. The repository has a fresh history, so the
dates in the plan are the record of that order. The commit hashes quoted in it belong to the original history, which is
not published.

## Results

Numbers come from `docs/plan.md` (section named in brackets). The raw run folders are not part of this repository.

### Games against Stockfish

Stockfish 19 limited to 50,000 nodes per move and to a UCI_Elo level, 100 paired games per level from the same
openings, Wombat at T = 8 steps. Scores are wins plus half the draws, as a fraction of 100 games. ("Games ladder
result".)

| Stockfish level | L2 cooled (5.6M parameters) | ChessBench 9M |
| --- | --- | --- |
| 1320 | 0.890 (88 wins, 2 draws, 10 losses) | 0.995 |
| 1500 | 0.645 | 0.995 |
| 1700 | 0.365 | 0.960 |
| 2000 | 0.250 | 0.865 |
| 2300 | 0.180 | 0.705 |
| 2600 | 0.030 | 0.515 |
| full strength | 0.000 | 0.010 |

The 1320 result has a 95% interval of 0.834 to 0.946 over 50 game pairs (`match.json` of that run). Both models lose to
full-strength Stockfish. The UCI_Elo anchors are not consistent under a node limit, so I compare per-level scores and
do not quote the implied ratings as Elo. At level 1700, 400 paired games gave L2 cooled 0.420 and L3 cooled 0.485 on
the first 200 openings. Over 800 games each on 400 openings, L3 was better by 0.029 per game with a 95% interval of
-0.019 to +0.078, so the two are level in play. ("Scaling check", "Overnight at home".)

### Static yardsticks

On a fixed 10,000 puzzle benchmark taken from Wombat's own test split and on 2,000 judged positions, L2 cooled solves
80.8% of whole puzzles at T = 32 and reaches 88.8 Lichess accuracy at T = 12. The ChessBench 9M scores 87.4% and 93.3
on the same data. On 42,242 live positions from ChessBench's held-out games, L2 cooled loses 20 or more win-chance
points on 2.14% of moves at T = 8. The ChessBench 9M does so on 0.38%. ("ChessBench 9M on our held-out data",
"Error-tail screen: built and calibrated".)

### Iteration inside the trained horizon

- Pilot v7 at step 53,000: accuracy 73.1 at T = 1 and 81.6 at T = 8, a gain of 8.5 points (interval 7.2 to 9.8).
- Pilot v13 on 4-move puzzles: T = 16 beats T = 8 by 4.4 points (interval 3.3 to 5.5). It was the first model whose
  answers improved measurably with more steps.
- L2's gain from T = 1 to T = 8 on validation grows with training, from +7.2 points at step 5,000 to +16.1 at step
  60,000.

### Where it fails

- **No gain beyond the trained depth.** Pilots v10, v11, v12, v13 (gate F2) and v14 (gate F2') all failed the test for
  gains past their trained depth. Every Wombat model peaks at or near its trained depth. L2's game strength peaks at
  T = 12 and falls by T = 32.
- **The core does not calculate.** `docs/thinking-steps-analysis-l2.md` follows the hidden state of L2 on 1,300 hard
  puzzles. The change per step halves every step or two, identically on easy and hard puzzles, and the solver's second
  move never becomes decodable from the state. The loop settles to a fixed point. It does not search.
- **The recurrent design did not beat a feedforward one.** In the only comparison with equal parameters (pilot v7,
  gate G5), the feedforward model won by 3.3 accuracy points.
- **The success gate was never evaluated.** The registered gate asked that T = 64 beat T = 1 by 100 Elo over 1,000
  paired games on two seeds. That match was not run.
- **Averages hid the problem.** The ChessBench 9M scores 0.105 to 0.615 more than L2 per ladder level, although their
  Lichess accuracy differs by 4.5 points. All 43,082 ladder positions were judged by Stockfish at 250,000 nodes. L2 makes a
  20+ point error on 5.0% of live moves against 1.4% for the 9M, at the same rate in games as on static positions.
  Re-labelling 3,000 positions and a fit check ruled out the training data and labels as the cause. Evaluation moved to
  an error-tail screen calibrated against games. ("Why L2 loses games: move-by-move analysis", "Item 24".)
- **Width 512 diverged (pilot v31a).** The largest query-key logit reached 162 (EMA weights) and 241 (raw weights)
  against 14 to 21 in healthy width-384 models. Normalizing queries and keys per head fixed the divergence, but the
  stable width-512 model scored 70.5% on puzzles against 71.3% for width 384, so the recipe stayed at 384.
- **An explicit imagined-line workspace added nothing (pilot v34).** It was flat across step budgets and stopped at
  step 4,000.
- **The paired T-budget match and three pilots have no results.** Items 25 to 27 (attack-pair attention bias,
  phase-specific output layers, king-relative square encoding) are implemented and tested. Their only pilot (v36a) was
  stopped at step 2,000.

### Reproducing a published baseline

DeepMind's released 9M model was ported to PyTorch and checked against their JAX code. It chose the same move on 408
positions, with log probabilities within 8.7e-5. It reproduced 86.2% on ChessBench's own puzzles, not the paper's 88.9%.
Of ChessBench's 10,000 test puzzles, 9,707 sit in Wombat's training split, so a fixed sample of Wombat's own test split
became the benchmark. An exact-position audit found 191 of those 10,000 puzzles with a position in training. ("ChessBench
9M on our held-out data", "Evaluation overlap with training".)

## What is not shown

- Recurrence was not shown to make Wombat stronger with more compute. The gains stop at the trained depth.
- It was not shown to beat a feedforward network of the same size.
- It is not close to the ChessBench 9M. It trails on every measure, by 0.105 to 0.615 in score per ladder level from
  1320 to 2600.
- The cause of the gap to the 9M is not established. The live 20+ error rate falls by about x0.85 per doubling of
  training samples, in a straight line in log-log with no flattening across the L1, L2 and L3 snapshots. Extrapolating
  to the 9M's 0.38% needs about ten more doublings, around 15 billion samples. This is a fit and not a measurement
  at that scale.
- The network does not learn chess without priors. The board features and the per-move static facts are hand-computed.
- L2 cooled saw about 14.7 million training samples drawn from 10.1 million labelled positions. The 9M learned from
  about 530 million positions ("Backlog item 19").

## Checkpoints

Two checkpoints of 5,596,077 parameters each are prepared for release outside this repository. Both are the cooled runs
at step 115,000. L2 is the one behind the table above. L3 adds 14.8 million game positions and is level with L2 in play.
Load one with `wombat.eval.puzzle_corpus_eval.load_checkpoint` or serve it with `wombat uci --checkpoint <file>`.

## Install and use

Python 3.12 or later. CPU is enough to run the tests and to play a checkpoint.

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements-cu130.txt
wombat doctor
python -m pytest
```

`requirements-cu130.txt` pins PyTorch 2.14.0 for CUDA 13.0. On other setups install a matching `torch` first and then
`pip install -e .[dev]`. GPU tests run with `WOMBAT_GPU_TESTS=1`.

The training, labelling and evaluation commands are listed by `wombat --help`. They need data that this repository does
not contain. `configs/downloads.yaml` records the Lichess sources. Stockfish 19, Cute Chess CLI and (optionally) Ordo are
found through `configs/local.yaml` (copy `configs/local.example.yaml`), environment variables or the `PATH`. Play a
checkpoint over UCI with:

```
wombat uci --checkpoint wombat-l2-cooled.pt
```

The `ReasoningSteps` UCI option sets T. The default is 8.

## Layout

- `src/wombat/model` holds the recurrent network, board features and move consequences.
- `src/wombat/chess` holds the canonical board encoding and the 4,672-action move codec.
- `src/wombat/data` and `src/wombat/eval` hold corpus building, Stockfish labelling, and the evaluations and matches.
- `src/wombat/interpretability` holds probes and hidden-state interventions. `src/wombat/visualizer` is a browser tool.
- `src/wombat/external` holds the PyTorch port of the ChessBench 9M model.
- `tests/` holds the test suite. `docs/` holds the plan and the mechanism study.

## License

Apache License 2.0 (`LICENSE`, `NOTICE`). Wombat imports python-chess, which is GPL-3.0-or-later, and runs Stockfish,
Cute Chess CLI and Ordo as separate programs. `THIRD_PARTY.md` lists every license and what it means for anyone who
ships Wombat together with them.
