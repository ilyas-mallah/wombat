# Wombat

Wombat is a research chess engine built to test one question. Does the same trained network play better when it runs
more steps of computation per position? The network is a small transformer whose weights are reused at every step.
"Thinking longer" means running the same core more times on a hidden state that persists between steps. It is written
in PyTorch.

The project is paused. The last experiment ran on 2026-09-27. The central idea is not proven, and the test it was built
around was never run. That test was registered before any training. It asked that T = 64 steps beat T = 1 by 100 Elo
over 1,000 paired games of one checkpoint, on two seeds. Extra steps help up to the number
the network was trained for and not beyond. The best checkpoint trails DeepMind's ChessBench 9M model by a wide margin.
`docs/plan.md` records the failed experiments as plainly as the ones that worked.

## The question

Standard engines buy strength with search. Stockfish runs alpha-beta search over millions of nodes with a small neural
evaluator at the leaves. AlphaZero and Leela run Monte Carlo tree search guided by a policy and value network.
DeepMind's ChessBench 9M transformer has no tree, but it spends one forward pass per legal move.

Wombat has no tree and no per-move passes. It encodes the position once and applies the same core T times to a hidden
state. Then it reads the move from the final state. Test-time compute is the step count T, not a number of nodes. The
plan opens with this question.

> Can a fixed-size recurrent neural network learn an internal chess-solving process that plays reliably stronger when
> the same trained model gets more sequential computation?

## How it works

This is the recipe of the long runs L2 and L3 (called v27a in the plan).

The board is turned to the side to move and becomes 64 square tokens plus one state token. Each square carries
hand-computed features. They are attacker counts, the cheapest attacker, pins, check, sole defenders, x-rays, checking
squares and material. So the network does not learn chess from raw pieces alone.

The recurrent core is two transformer blocks (RMSNorm, multi-head attention, SwiGLU, residual) with the same weights at
every step. There is no step embedding and no built-in maximum depth. The state is RMS-normalized after each step.
Attention is local, so each square sees the squares one king step away. It uses learned direction-aware relative
attention, a learned 64 by 64 square bias, and a bias generated from the state at every step.

The readout gives per-square move logits plus a query-key score between a move's origin and destination tokens. For
every legal move the GPU also computes four static facts on the position after it. They are the exchange value of the
moved piece, the opponent's best exchange gain, mate in one, and check. The model weighs these per move. Position and
per-move values are 64-bin distributions. The network has width 384, 8 heads, MLP width 1,536 and 5,596,077 parameters.

Each training batch first runs 0 to 8 steps without gradient, then 8 steps with gradient. A quarter of the batches are
Lichess puzzle positions, run for 4 steps per remaining solver move. The policy target is a soft distribution over
every legal move, built from Stockfish win chances. The optimizer is AdamW with EMA weights. Training ends with a
cooldown that scales the learning rate by one minus the square root of the cooldown progress.

The game positions come from the Lichess evaluation database. Stockfish 19 scored every legal move in them. The first
game corpus holds 10,092,005 labelled positions, 4.0 million at 50,000 nodes and 6.1 million at 20,000 nodes. Training
also draws on a Lichess puzzle corpus. Splits are made by a hash of the position, and the frozen evaluation sets are
excluded.

A home RTX 3070 Ti with 8 GB of memory ran most pilots. A rented RTX 5090 with 32 GB ran the long runs L2 and L3 and
part of the Stockfish labelling.

Every experiment's gate is written in `docs/plan.md` before its run. This repository has a fresh history, so the dates
in the plan are the record of that order. The commit hashes quoted there belong to the original history, which is not
published.

## Results

Every number below comes from `docs/plan.md`, with the section named in brackets. The raw run folders are not part of
this repository.

### Games against Stockfish

Stockfish 19 played at 50,000 nodes per move and at a set UCI_Elo level. Each level had 100 paired games from the same
openings, with Wombat at T = 8. A score counts wins plus half the draws, as a fraction of 100 games ("Games ladder
result").

| Stockfish level | L2 cooled (5.6M parameters) | ChessBench 9M |
| --- | --- | --- |
| 1320 | 0.890 (88 wins, 2 draws, 10 losses) | 0.995 |
| 1500 | 0.645 | 0.995 |
| 1700 | 0.365 | 0.960 |
| 2000 | 0.250 | 0.865 |
| 2300 | 0.180 | 0.705 |
| 2600 | 0.030 | 0.515 |
| full strength | 0.000 | 0.010 |

Both models lose to full-strength Stockfish. The UCI_Elo anchors are not consistent under a node limit. I compare
per-level scores and do not quote the implied ratings as Elo.

At level 1700, L2 cooled scored 0.420 and L3 cooled 0.485 over 400 games each on 200 openings ("State at wrap-up").
Doubling to 800 games each on 400 openings, L3 was ahead by 0.029 per game. The 95% interval was -0.019 to +0.078, so
the two are level in play ("Scaling check").

### Static yardsticks

The puzzle benchmark is 10,000 puzzles from Wombat's own test split. A puzzle counts as solved only when every solver
move is right. L2 cooled solves 80.8% at T = 32, and the ChessBench 9M solves 87.4%. A second set holds 2,000 game
positions judged by Stockfish. There L2 cooled reaches 88.8 Lichess accuracy at T = 12, and the 9M 93.3 ("ChessBench
9M on our held-out data", "Backlog item 20").

The error-tail screen uses 42,242 live positions from ChessBench's held-out games. L2 cooled loses 20 or more win-chance
points on 2.14% of its moves at T = 8. The 9M does so on 0.38% ("Error-tail screen: built and calibrated").

### More steps inside the trained depth

Pilot v7 at step 53,000 reached accuracy 73.1 at T = 1 and 81.6 at T = 8. That gain of 8.5 points had a 95% interval of
7.2 to 9.8 ("v7 outcome").

Pilot v13 improved by 4.4 points from T = 8 to T = 16 on 4-move puzzles (interval 3.3 to 5.5). It was the first model
whose answers improved measurably with more steps ("v13 outcome").

L2's validation gain from T = 1 to T = 8 grew with training. It was +7.2 points at step 5,000 and +16.1 at step 60,000
("Backlog item 20").

### What did not work

The registered success test was never run. It asked that T = 64 beat T = 1 by 100 Elo over 1,000 paired games on two
training seeds ("Success gate"). L2 and L3 were each trained on one seed, seed 0 in their configs.

No model gained beyond its trained depth. Pilots v10 to v12 failed gate E1, v13 failed F2 and v14 failed F2'. Each gate
asked for gains past the trained depth (sections "v10 outcome" to "v14 outcome"). L2's Lichess accuracy peaks at 88.8
at T = 12 and falls to 86.9 at T = 32 ("Backlog item 20"). On the error-tail screen every Wombat model does best at
T = 8 and slightly worse at T = 16 ("Error-tail screen: built and calibrated").

The core does not calculate. `docs/thinking-steps-analysis-l2.md` follows L2's hidden state on 1,300 hard puzzles. The
change per step halves every step or two, the same on easy and hard puzzles. The solver's second move never becomes
decodable from the state. The loop settles to a fixed point instead of searching ("Mechanism of the thinking steps").

The recurrent design did not beat a feedforward one. The only comparison at equal parameters was pilot v7 against
four untied blocks read out once. The feedforward model won by 3.3 accuracy points (gate G5, "v7 outcome").

Averages hid the problem. The 9M scores 0.105 to 0.615 more than L2 per ladder level, while their Lichess accuracy
differs by 4.5 points. Stockfish at 250,000 nodes judged all 43,082 ladder positions. L2 lost 20 or more points on
5.0% of live moves and the 9M on 1.4%. L2's rate was the same in games as on static positions ("Why L2 loses games").
Relabelling 3,000 positions with one search per move ruled out label quality as the cause (item 23, same section). From
then on, experiments were judged on an error-tail screen calibrated against games ("Measuring playing strength").

Width 512 diverged in pilot v31a. Its largest query-key logit reached 162 with EMA weights and 241 with raw weights.
Healthy width-384 models stayed at 14 to 21. Normalizing queries and keys per head made width 512 stable. It then
solved 70.5% of puzzles against 71.3% for width 384, so the recipe stayed at 384 ("Why v31a diverged", "v33 and v34").

A workspace that plays out imagined lines added nothing in pilot v34. Validation was flat across step budgets, and the
pilot was stopped at step 4,000 ("v33 and v34").

Items 25 to 27 (attack-pair attention bias, phase-specific output layers, king-relative square encoding) are coded and
tested in this repository. They have no results ("Items 25 to 27"). A first pilot of item 25 started on 2026-09-27 and
was stopped at step 2,000 before any evaluation. Its log is not in this repository.

### Reproducing a published baseline

I ported DeepMind's released 9M model to PyTorch and checked it against their JAX code. It chose the same move on 408
positions, with log probabilities within 8.7e-5. It reproduced 86.2% on ChessBench's own puzzles, not the paper's
88.9%. Of ChessBench's 10,000 test puzzles, 9,707 sit in Wombat's training split. So a fixed sample of Wombat's own
test split became the benchmark. 191 of those 10,000 puzzles share an exact position
with training ("ChessBench 9M on our held-out data", "Goal and backlog", "Evaluation overlap with training").

## What is not shown

Recurrence was not shown to make Wombat stronger with more compute. The gains stop at the trained depth, and the
registered T = 64 against T = 1 match was never played.

Wombat was not shown to beat a feedforward network of the same size.

Wombat is not close to the ChessBench 9M. It trails on every measure, by 0.105 to 0.615 in score per ladder level from
1320 to 2600.

The cause of the gap to the 9M is not established. The live 20+ error rate falls by about x0.85 per doubling of
training samples. Across the L1, L2 and L3 snapshots it is a straight line in log-log with no flattening. Reaching the
9M's 0.38% on that line takes about ten more doublings, around 15 billion samples. That is an extrapolation of a fit,
not a measurement at that scale ("Scaling check").

The network does not learn chess without priors. The board features and the per-move static facts are hand-computed.

L2 cooled saw about 14.7 million training samples drawn from 10.1 million labelled positions ("Scaling check"). The 9M
learned from about 530 million positions ("Backlog item 19").

## Checkpoints

Two checkpoints are published as separate model files, `wombat-l2-cooled` and `wombat-l3-cooled` (TODO: add the
Hugging Face URL once they are uploaded). Each has 5,596,077 parameters. Both are cooled runs at step 115,000. L2 is
the one behind the results above. L3 adds 14.8 million game positions and is level with L2 in play. Load one with
`wombat.eval.puzzle_corpus_eval.load_checkpoint`, or serve it with `wombat uci --checkpoint <file>`.

## Install and use

Wombat needs Python 3.12 or later. A CPU is enough to run the tests and to play a checkpoint.

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements-cu130.txt
wombat doctor
python -m pytest
```

On Linux, activate with `source .venv/bin/activate`. `requirements-cu130.txt` pins PyTorch 2.14.0 for CUDA 13.0. On
other setups, install a matching `torch` first and then run `pip install -e .[dev]`. GPU tests run with
`WOMBAT_GPU_TESTS=1`.

`wombat --help` lists the training, labelling and evaluation commands. They need data that this repository does not
contain. `configs/downloads.yaml` records the Lichess sources. Stockfish 19, Cute Chess CLI and Ordo (optional) are
found through `configs/local.yaml` (copy `configs/local.example.yaml`), environment variables or the `PATH`. Play a
checkpoint over UCI with:

```
wombat uci --checkpoint wombat-l2-cooled.pt
```

The UCI option `ReasoningSteps` sets T. The default is 8.

## Layout

- `src/wombat/model` holds the recurrent network, the board features and the move consequences.
- `src/wombat/chess` holds the canonical board encoding and the 4,672-action move codec.
- `src/wombat/data` and `src/wombat/eval` hold corpus building, Stockfish labelling, and the evaluations and matches.
- `src/wombat/interpretability` holds probes and hidden-state interventions.
- `src/wombat/visualizer` is a local web page for playing against and watching a checkpoint on CPU.
- `src/wombat/external` holds the PyTorch port of the ChessBench 9M model.
- `tests/` holds the test suite, and `docs/` the plan and the mechanism study.

## License

Wombat is licensed under the Apache License 2.0 (`LICENSE`, `NOTICE`). It imports python-chess, which is
GPL-3.0-or-later. It runs Stockfish, Cute Chess CLI and Ordo as separate programs. `THIRD_PARTY.md` lists every license
and what it means for anyone who ships Wombat together with them.
