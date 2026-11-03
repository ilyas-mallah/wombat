# Wombat v1 execution plan

This is the working plan and log of the project, kept in the order it was written between 2026-09-14 and 2026-09-27.
Each experiment's gate is written before its run, and the outcome is appended afterwards. Paths such as `runs/...`
and `data/...` point to folders that are not part of this repository.

## Objective

Answer one question with a controlled experiment:

> Can a fixed-size recurrent neural network learn an internal chess-solving process that plays reliably stronger when the same trained model gets more sequential computation?

The primary result is a compute-scaling curve from one recurrent checkpoint evaluated at `T = 1, 2, 4, 8, 16, 32, 64, 128`, extending farther when practical. Model size, weights, positions, openings, opponents, and evaluation budgets remain fixed. These are measurement points, not architectural stages or semantic roles.

## Success gate

Wombat v1 is successful only if all of the following reproduce across two training seeds:

- `T=64` beats `T=1` by at least 100 Elo in at least 1,000 paired games.
- At least two intermediate increases among `T=1,4,16,64` are independently positive, rather than all improvement appearing at one privileged endpoint.
- The compute-scaling slope is positive for deep-Stockfish regret and puzzle accuracy on frozen test sets.
- No sustained degradation begins before `T=64`. Isolated regressions must remain inside predeclared confidence intervals.
- `T=128`, which lies beyond the initial training curriculum, is always reported as an extrapolation test even if it plateaus or degrades.
- The comparison uses exactly the same checkpoint weights.

Before the gate is met, do not add self-play, learned halting, distributed training, C++, TensorRT, or models above roughly 5 million parameters.

## Local constraints

Observed on 2026-09-14:

- Windows, Python 3.12.6, Git available.
- NVIDIA RTX 3070 Ti with 8 GB VRAM, 32 GB system RAM.
- Approximately 35 GB free on `D:`.
- CUDA-enabled PyTorch, python-chess, zstandard, tqdm, pytest, Stockfish 19, and Cute Chess CLI 1.5.1 are installed. Ordo remains optional and is not installed.
- The project directory is empty and is not yet a Git repository.

Consequences:

- Stream and shard datasets. Never unpack the full Lichess corpus.
- Start with 10,000 positions, then 100,000, and cap the first local corpus near 1 million after measuring its actual size.
- Keep raw downloads outside Git and enforce a disk-space guard.
- Use mixed precision and gradient accumulation rather than assuming a large physical batch fits in VRAM.

## Repository target

```text
wombat/
  pyproject.toml
  README.md
  configs/
    model_small.yaml
    train_smoke.yaml
    eval_v1.yaml
  src/wombat/
    chess/
      encode.py
      moves.py
    data/
      schema.py
      splits.py
      lichess.py
      shards.py
    model/
      encoder.py
      recurrent.py
      heads.py
      network.py
    eval/
      suite.py
      agreement.py
      regret.py
      puzzles.py
      matches.py
      ratings.py
    engine/
      uci.py
    train.py
    cli.py
  tests/
    fixtures/
  data/                 # ignored; manifests tracked
  checkpoints/          # ignored
  runs/                 # ignored; small summaries may be promoted
  matches/              # ignored; selected PGNs/summaries may be promoted
```

## Locked v1 design decisions

### Position encoding

- Canonicalize every position to the side-to-move perspective: rotate black-to-move positions by 180 degrees and swap piece colors.
- Use 65 tokens: 64 square tokens plus one state token.
- A square token is the sum of a 13-way piece embedding and a learned square embedding.
- The state token encodes side to move before canonicalization, castling rights, en-passant file or none, and halfmove clock bucket.
- Ignore repetition history in v1 because it cannot be recovered reliably from isolated FEN records.

### Move encoding

- Use `64 x 73 = 4,672` logits: 56 queen-like rays, 8 knight jumps, and 9 underpromotion planes.
- Queen promotions use the corresponding ray plane. Rook, bishop, and knight underpromotions use dedicated planes.
- Generate a legal mask with python-chess, map legal UCI moves into the canonical coordinates, and invert the transform when returning an engine move.
- Illegal logits are set to negative infinity before selection or softmax.

### Labels and splits

- Store FEN, best move, centipawns or mate, source depth, source nodes, and optional alternative PV scores.
- Normalize value as `tanh(cp / 600)` and map mate to `+1` or `-1` from the canonical side-to-move perspective.
- Verify source score perspective with hand-checked positions before ingesting real data.
- Split by a stable hash of normalized FEN: 98% train, 1% validation, 1% test.
- Freeze evaluation manifests by content hash. Training must reject any FEN present in a frozen evaluation manifest.

### Model

- `d_model=192`, 6 attention heads, MLP width 768.
- The current reasoning-v2 experiment uses no non-recurrent prelude. `H0` is a
  position-aware but non-aggregated baseline. The first application of the
  shared core performs the first board-wide computation. The older three-block
  prelude remains a compatibility option, not the active design.
- Initialize `H0` as 65 board/state tokens plus 32 learned latent scratchpad tokens.
- One recurrent transformer block implements `F(B, H)` in reasoning-v2 and
  reuses the same weights at every iteration. Immutable `B` remains available
  as external conditioning, while the entire 97-token `H`, including its mutable
  board-aligned workspace, is rewritten and persisted.
- Each recurrent block uses RMSNorm, multi-head attention over stable board memory plus the current hidden state, SwiGLU, and residual updates.
- Enforce `shape(F(B, H)) == shape(H)` as a tested invariant.
- Do not provide a step-number embedding or impose a learned maximum step index. The recurrence is architecturally valid for any positive iteration count.
- Produce `64 x 73` policy logits and scalar value only from the current latent workspace.
- Emit heads after each requested supervised depth.
- Record exact parameter count. The v1 target is 2-5 million, and the estimate is not treated as guaranteed.
- Do not hand-encode attacks, threats, candidate moves, branches, principal variations, or semantic roles for individual latent tokens or iterations.

### Training

- First baseline: prelude plus heads, without the recurrent loop.
- Supervise a soft distribution over every available MultiPV candidate rather than a one-hot best move. Convert score differences into temperature-controlled probabilities. Train a separate legal-move value head on each evaluated candidate so near-equal moves are not labelled as equally wrong.
- Root MultiPV scores are only the first stage. A corpus intended to teach implicit search needs several candidates for nearly every position. It also needs short continuation lines and opponent replies for those candidates. Report candidate-count coverage before training. Do not call a mostly single-PV corpus sufficient.
- Decode continuation moves from the recurrent state plus the preceding candidate-move prefix. Tie action embeddings and the decoder classifier to the normal policy head. This auxiliary objective then teaches counterfactual line information without imposing a literal tree or named latent slots on Wombat.
- Use paired teacher depths on the same position. Train `H1` to expose the
  cheap-search proposal distribution and its replies, while randomly sampled
  later states target stronger MultiPV candidates, candidate values, and
  continuation lines. This creates a candidate-specific rollout-and-backup
  learning signal without giving the core a step number or assigning semantic
  roles to scratchpad slots.
- Recurrent training samples every integer length from a curriculum that expands from `1..8` through `1..16`, `1..32`, and `1..64`. An approximately log-uniform mixture keeps short and long trajectories both represented.
- Sample one final compute budget per example. Put most target loss on that final state and only a weak auxiliary loss on an earlier prefix. No particular iteration receives a unique semantic target.
- Add an explicit per-position continued-improvement hinge on teacher quality. The implemented form is `relu(Q_later - stopgrad(Q_earlier) + min(margin, fraction * Q_earlier))`, masked for already-solved positions. Detaching the comparator prevents the model from satisfying the hinge by making the earlier state worse.
- Randomize both the final depth and the earlier comparison prefix, with H0 as a possible prefix. No midpoint or other iteration receives a privileged identity.
- Initialize recurrent residual output projections to zero when starting from an H0 checkpoint, making the initial recurrence an exact identity map. This avoids treating random recurrent damage as useful compute.
- Keep each attention and MLP residual branch under a smooth, tested per-token RMS bound. Sweep this bound as a hyperparameter rather than assuming it is universally optimal.
- On untouched validation positions, record the complete per-position quality curve and the adjacent-depth improvement and regression fractions. Also record oscillation categories and model-relative difficulty quartiles. Treat training-set improvement as a mechanics check, not evidence that compute generalizes.
- Start with `policy_loss + 0.25 * value_loss`.
- AdamW, initial learning rate `3e-4`, cosine decay, gradient clipping, CUDA automatic mixed precision, effective batch 128-256.
- Every run records config, seed, package versions, Git commit, data manifest hash, metrics JSONL, and checkpoints.

## Ordered implementation milestones

Current status (2026-09-15): M0-M3 are complete, including the frozen 50k puzzle suite and leak-resistant 1,000,000-position corpus. Implemented so far are identity initialization, randomized prefix comparisons, a detached per-position adaptive improvement loss and RMS-bounded updates. Content-dependent update gates, soft MultiPV policy, per-candidate values, continuation decoding and held-out trajectory reporting are implemented too. The 100k corpus has 590,819 shallow candidate lines and 2,150,991 PV plies. The reasoning-v2 shallow-to-deep curriculum is the first material held-out compute result. On the untouched 943-position test split, combined teacher quality improves from 3.35879 at H1 to 3.30621 at H2. 60.1% of positions improve, and the gain remains positive through H8 (3.29963). Top-1 rises from 23.0% at H1 to 24.1% at H8 and 24.6% at H16. Frozen-puzzle exact-move accuracy still falls with recurrence. This is useful teacher-distribution refinement, not yet proof of stronger chess. A follow-up weighted the shallow/deep teacher disagreements. It improved soft loss on those hard positions but not the overall test results, so weighting alone is insufficient.

Research update (2026-09-15): the closest demonstrated system is DiffuSearch
(ICLR 2025). It jointly denoises alternating future action and resulting-state
tokens with bidirectional attention. Its ablations show that future moves alone
are insufficient: valid action-conditioned future states are the critical
ingredient. Deep Thinking with Recall and Huginn also use a training mechanism
we had not faithfully reproduced. A random no-gradient recurrent prefix is
detached and followed by a random gradient-carrying tail, with no iteration
index given to the model. That exact 50/50 progressive objective is now
implemented. A clean 10k paired-depth run (`reasoning-v3-progressive-recall-10k`)
passed the training mechanics check but failed held-out scaling. Test quality
was 4.4242, 4.4307, 4.4683, 4.4936, 4.5215, and 4.5542 at H1/H2/H4/H8/H16/H32.
Top-1 fell from 19.6% at H1 to 16.8% at H32. This rules out progressive training
alone on the current representation. The next model must make several predicted
action/state futures first-class recurrent workspace tokens. Their contents feed
the root decision, instead of continuation prediction being an auxiliary
readout. Use permutation-invariant candidate-line assignment, exact chess
transitions as supervision, cross-line attention, and distributional values.

A paired-depth pilot now contains 10,000/50,000-node MultiPV supervision for 10,000 train, 957 validation, and 943 test positions (46% shallow/deep best-move disagreement). It also makes each root candidate retrieve an unordered scratchpad context, so continuation gradients reach latent tokens. This did not generalize useful recurrence beyond H1/H2. A dynamics audit found the unconstrained update RMS increased from 0.153 at H1 to 0.203 at H16. A detached contraction loss was added and reduced update RMS monotonically from 0.0976 at H1 to 0.0140 at H16. It improved H32 validation top-1 from 15.2% to 17.1%, but deep quality still remained worse than H1. A first learned-backup head was also tested. Every interchangeable latent token proposed moves with the tied action head and carried a learned scalar value. Each token contributed a differentiable log-sum-exp backup to policy and per-move values. It preserved a tiny H1-to-H2 gain at epoch 4 (4.3849 to 4.3824) but degraded thereafter (4.4224/H4, 4.4726/H8). This simple one-level aggregation is rejected too. Do not scale either variant. A 200-position frozen-puzzle Stockfish-regret check at 20,000 nodes also rejects the current best checkpoint as a strength result. Median regret is 1,324 cp at H1 and 1,337 cp at H8/H16, while exact reference-move selection falls from 12.5% to 11.5%/11.0%. The next architecture needs recurrently updated candidate identities, opponent-conditioned child states and an alternating side-to-move/minimax backup. An auxiliary decoder or pooled scalar slots are not enough. It must pass the 10k paired-depth validation and external-regret gates before any 100k/1M annotation or training. The shallow-to-deep epoch-5 checkpoint remains the current best teacher-loss checkpoint, not a successful chess engine.

### M0: Bootstrap and reproducibility

Deliverables:

- Initialize Git and add Python/cache/data/checkpoint/run ignores.
- Create `.venv`, packaging metadata, pinned direct dependencies, and CLI entry point.
- Install a CUDA-enabled PyTorch build and verify a tensor operation on the 3070 Ti.
- Install or configure Stockfish. Record its executable path and reported version.
- Add deterministic seed handling, structured run directories, config snapshots, and environment capture.
- Add CI-friendly CPU smoke tests even though training uses CUDA.

Acceptance:

- `python -m pytest` passes from a clean environment.
- `python -m wombat.cli doctor` reports Python, packages, CUDA, GPU, disk space, and external engine availability.
- A one-batch CUDA smoke operation succeeds without modifying tracked files.

### M1: Chess representation and invariants

Deliverables:

- Position canonicalization and 65-token encoder.
- Bidirectional 4,672-action move codec.
- Legal-action mask.
- Hand-authored FEN fixtures covering ordinary moves, every castling case, en passant, check/evasion, checkmate, stalemate, queen promotion, all underpromotions, and black-to-move transforms.

Acceptance:

- Every legal move round-trips `Move -> action -> Move` for at least 10,000 deterministic random legal positions.
- Exactly the python-chess legal move set is represented by the mask for those positions.
- No legal moves are produced for checkmate or stalemate.
- Canonicalized black and equivalent white-perspective fixtures encode consistently.

### M2: Evaluation infrastructure before training

Deliverables:

- A model-adapter protocol so a random model, heuristic fixture model, baseline, and recurrent checkpoint use the same evaluator.
- Frozen position/puzzle/opening manifests with hashes and explicit train exclusion.
- Move agreement, top-k agreement, policy loss, value MAE, and calibration metrics.
- Stockfish-regret evaluator with a fixed node budget, caching keyed by position/move/engine/version/budget, and side-to-move-correct score handling.
- Puzzle first-critical-move evaluator with rating buckets.
- Minimal UCI engine supporting `uci`, `isready`, `ucinewgame`, `position startpos|fen ... moves ...`, `go` with a fixed `ReasoningSteps` option, `stop`, and `quit`.
- Match runner using paired openings and PGN plus machine-readable result output.
- Elo estimate and confidence interval. Preserve pentanomial pair outcomes for later analysis.
- Latency, VRAM, throughput, parameter-count, and model-size measurements.

Acceptance:

- The complete evaluator runs against an untrained checkpoint without crashing.
- A fixture policy that always chooses a known move receives the expected exact score.
- Regret is zero for a fixture where the model selects Stockfish's cached best move.
- A two-game color-swapped UCI smoke match completes and produces valid PGN.
- Re-running the same manifest/config produces the same sampled positions and cache keys.

### M3: Data pipeline

Deliverables:

- Streaming parser for compressed Lichess evaluation records.
- Validated typed record schema and rejection log.
- Deterministic split assignment and evaluation-FEN exclusion.
- Compact sharded format with manifest, checksums, counts, schema version, source metadata, and disk-space guard.
- A 10,000-position debug corpus, followed by a 100,000-position pilot.
- Separate frozen puzzle evaluation slice. Begin smaller than 50,000 if download/storage pressure warrants it.

Acceptance:

- Parsing a fixed input twice produces identical shard hashes and split counts.
- No FEN hash overlaps among train, validation, test, and puzzle evaluation manifests.
- Every stored best move is legal in its stored FEN.
- Manually checked score-sign fixtures pass before bulk ingestion.

### M4: Non-recurrent baseline

Deliverables:

- Encoder, transformer prelude, policy head, value head, training loop, checkpoint/resume, and metrics logging.
- Parameter-count and peak-memory report.
- A deliberate 10,000-example overfit run.
- A 100,000-position pilot only after overfit succeeds.

Acceptance:

- The model overfits the 10,000-example corpus to a predeclared high move-agreement threshold (initially 95% training top-1). Otherwise it produces a diagnosis of why the threshold is inappropriate.
- Resume reproduces the uninterrupted run within documented numerical tolerance.
- Evaluation suite emits a versioned report for the checkpoint.

### M5: Open-ended recurrent experiment

Deliverables:

- Immutable board memory, persistent latent workspace, weight-tied recurrent core, broad variable-length training, and intermediate supervision.
- Tests proving parameter count is independent of `T` and recurrent block parameters are shared.
- Tests proving there is no step embedding or architecture-level maximum recurrence length.
- Evaluation of the same checkpoint at `T=1,2,4,8,16,32,64,128`, extending farther when latency permits.
- Per-position trajectories showing whether policy/value predictions converge, oscillate, or degrade with more steps.
- Per-position quality sequences `Q1..Qn`, classified as immediate convergence, steady improvement, late breakthrough, oscillation, or degradation. Automatically retain late-breakthrough and degradation cases for tracing.
- Hidden-state interventions: reset at a midpoint, replace with an earlier state, add controlled corruption, and swap states between unrelated positions.
- Opt-in traces for `H0..Hn` containing the full hidden workspace, legal-move probabilities, value estimates, per-head attention maps, and scratchpad activations. Trace hooks must exist before training but remain disabled on the performance-critical path.
- Probe datasets aligned to each recurrent state for attacked squares, legal moves, king safety, teacher evaluation/best move, and future PV moves. Probes are diagnostic and never feed handcrafted features back into the base model.
- Maintain a separate interpretability package with versioned, operational definitions for hypothesized concepts. Examples are hanging pieces, forks, pins, overloaded defenders, king danger, candidate moves, and likely replies. The base network never receives these labels.

Acceptance:

- One checkpoint hash is recorded across every depth comparison.
- Logits and gradients are finite through the current training curriculum. Inference is finite at least one doubling beyond it.
- The evaluator produces depth curves for agreement, regret, puzzles, latency, and memory.
- Continuing `H32 -> H64` must be compared with resetting to `H0` and running 32 more steps. Accumulated-compute gains should disappear or shrink under reset, corruption, earlier-state replacement, and cross-position state swaps.
- Compare `T_train,max` with strictly larger test depths. A model that only improves at trained anchor depths does not pass the open-ended-compute claim.
- Probe accuracy is charted against recurrence depth and correlated with external chess-quality gains. Any feature intervention suggested by a probe must be tested causally before it is treated as explanatory.
- Every concept result compares a probe with shuffled-label and capacity-matched controls. A concept counts as behaviorally relevant only when suppressing or modifying its direction has a predictable policy/value effect. That effect must exceed unrelated-direction controls.
- Human-readable statements such as “the knight becomes decodable at step 14” must state the probe, label definition, uncertainty, and causal-test status. Decodability alone is never described as the model using that concept.

### M6: Auxiliary search-depth ladder

Deliverables:

- Deterministically sample 20,000 positions from the training-eligible pool.
- Analyze each with the configured Stockfish build at fixed budgets of 256, 1,024, 4,096, 16,384, and 65,536 nodes.
- Store engine/version/options/PV/score/nodes and resumable progress.
- Optionally train intermediate depths against corresponding node-budget targets as a separate, clearly labeled ablation. This is never the fundamental objective and must not make a chosen recurrent step architecturally special.

Acceptance:

- Interrupted generation resumes without duplicate or mismatched records.
- A sample re-analysis matches best moves/scores within declared engine determinism constraints.
- Ladder training is compared against the ordinary recurrent objective without changing the frozen evaluator.

### M7: Credible engine-strength test

Deliverables:

- Install/configure Cute Chess CLI and optionally Ordo.
- Build a fixed opening suite and Stockfish node-budget opponent ladder.
- Run 200 paired games per comparison for iteration, then 1,000 paired games for the success gate.
- Report W/D/L, pentanomial pair counts, Elo estimate, confidence interval, crashes, illegal moves, and time losses.

Acceptance:

- The tournament is resumable and reproducible from its manifest.
- There are zero illegal moves and zero unexplained engine crashes.
- The same recurrent checkpoint is used for all `T` settings.
- The success gate is evaluated without moving its thresholds after results are known.

## First executable slice

The first slice implements only M0-M2 scaffolding and stops before real data download or training:

1. Bootstrap the package, tests, configs, ignores, and `doctor` command.
2. Implement and exhaustively test board canonicalization, move encoding, decoding, and legal masks.
3. Define the frozen-manifest and evaluation-report schemas.
4. Implement a random/fixture model adapter, agreement evaluator, minimal UCI loop, and paired-match command.
5. Install/configure Stockfish and make cached fixed-node regret work on a tiny hand-authored suite.
6. Run all tests and produce the first evaluation report from an untrained fixture model.

This slice is complete when evaluation works end-to-end with synthetic fixtures. It deliberately does not download Lichess data or start training.

## Required report shape

Every evaluated checkpoint produces one directory containing:

```text
report.json
positions.jsonl
puzzles.jsonl
matches.pgn
match_summary.json
environment.json
config.yaml
```

`report.json` includes checkpoint SHA-256, Git commit, dataset manifest hashes, seed, reasoning steps, parameter count, latency percentiles and peak VRAM. It also holds policy/value metrics, regret distribution, puzzle buckets, match results, Elo estimate, confidence interval, and hidden-state intervention results.

## Stop conditions

Pause and diagnose instead of scaling when any of these occurs:

- A legal-move, score-perspective, split-leakage, or reproducibility invariant fails.
- The 10,000-example baseline cannot overfit.
- Additional recurrent steps improve training loss but not any frozen external metric.
- The strength/quality curve is flat or degrades across increasing compute after the planned evaluation budget.
- Dataset growth would leave less than 10 GB free on `D:`.

Only after the success gate should the project consider larger models/data, learned halting, optimized inference, or online play. Once supervised Wombat is competent enough to produce meaningful games, add self-play gradually while retaining teacher data. Then increasingly optimize actual game outcomes rather than permanent teacher imitation.

## Latest recurrent-compute result (2026-09-17)

`reasoning-v5-truncated-backed-100k` is the first checkpoint to pass all three
small-scale compute-use gates. It uses eight recurrent candidate workspaces,
explicit future move/resulting-board prediction, candidate-specific MultiPV
value supervision, and random detached prefixes followed by at most four
gradient-carrying steps. Neither the prefix length nor requested inference
depth is exposed to the model.

- Untouched test quality improves at every doubling from H1 through H8:
  `2.79338 -> 2.77610 -> 2.76538 -> 2.76115`. It begins regressing at H16.
- Frozen 50,000-puzzle accuracy improves from `12.758%` at H1 to `13.056%`
  at H2 and remains about `13.04%` through H8, before declining.
- On the fixed 200-puzzle, 20,000-node Stockfish check, median regret improves
  from `1,348 cp` at H1 to `1,255 cp` at H8 and `1,250 cp` at H16. Clipped
  mean regret improves from `4,306.5 cp` to `4,098.9 cp`, while exact
  reference-move selection rises from `7.0%` to `10.0%`.
- Constant-memory prefix/tail training reduced the depth-32 epoch from a VRAM
  paging stall at roughly 7.9 GB to 139 seconds at roughly 4.6 GB.

This is evidence of useful implicit computation, not yet a strong chess
engine. The next experiment should stabilize the post-H8 dynamics and then
run paired games at H1/H8/H16. Do not increase the corpus or model size until
that comparison confirms a playing-strength gain.

## v6 puzzle-compute experiment (registered 2026-09-22, before any result)

v5's H1 is a single transformer layer, so its H1 to H8 gain cannot be
separated from ordinary network depth. Its base policy (24% top-1, 13%
puzzles) is also too weak for extra compute to refine. v6 tests the compute
question on forcing Lichess puzzle lines. There lookahead demonstrably matters
and labels are clean. It follows the Deep Thinking recipe that has reported
test-time iteration gains on chess puzzles, and the findings on
stability and objectives of `reasoner`, an earlier project that is not part of
this repository.

Design:

- Corpus: every solver-to-move position of every Lichess puzzle line, tagged
  with rating and solver moves remaining. Splits are by puzzle id (98/1/1).
  Frozen 50k puzzle ids and FEN hashes are excluded from every split.
- Readout: each square token emits the 73 move logits for its own from-square.
  This replaces the single global token that had to produce all 4,672 logits.
- Stability: RMSNorm on H after each step with no residual cap. A short pilot
  compares it against the v5 cap and gates, chosen on validation only.
- Model: one prelude block, a two-block shared recurrent core, 16 scratch
  tokens, trained from scratch. No checkpoint lineage.
- Objective: cross-entropy on the puzzle move from the final state only. A
  random no-gradient prefix of `n` steps is followed by a gradient tail of `k`
  steps (`n` uniform in `0..15`, `k` uniform in `1..min(8, 16 - n)`). The
  deepest trained state is H16. No improvement hinge, early auxiliary loss,
  future slots, or continuation decoder.
- Control: the same architecture, data, and step budget trained only at T=1.

Gates, on the held-out puzzle test split, with paired 95% confidence intervals:

- G1: accuracy at T=16 exceeds T=1 by at least 5 percentage points.
- G2: accuracy at T=32 and T=64 is no more than 1 point below T=16.
- G3: the recurrent checkpoint at its best T beats the T=1 control.

Also reported without a gate: gains by rating bucket and by solver moves
remaining, and the full-line solve rate. First-move accuracy on the frozen 50k
suite is reported too (v5: 13.06% at best depth). Any move that delivers checkmate counts as
correct, matching Lichess.

### v6 outcome (2026-09-23): stopped at step 8,000 of 60,000, gates not evaluated

- Stability pilot (3,000 steps, 8,192 validation positions): normalized state
  reached 33.9% at T1 and 41.1% at T4 and T8 (39.5% at T32). The v5-style cap
  reached 34.2% at T1 and 38.9% at best. It ran at 941 against 1,533
  positions/s and needed about 4.3 against 3.3 GiB. Normalized was chosen.
- Strength probes on CPU used 300 whole validation puzzles and 150 ordinary
  positions from the main-1m test split, judged by Stockfish at 20,000 nodes.
  At step 5,000 the puzzle rating was about 960 at T1 and 967 at T8 (step
  4,000: 851 and 980). On ordinary positions 47% of moves were good, 42% were
  blunders, and Lichess accuracy was 58% at T1.
- Reason for stopping: puzzle-only training produces a tactics specialist that
  blunders in ordinary positions. Every training position contains a winning
  tactic, and one-hot labels never teach that several moves can be fine. The
  registered gates would have measured recurrence on tactics, not whether
  Wombat becomes a better player.
- Engineering results that carry forward: a 13.7M-position puzzle corpus for
  evaluation and CUDA-graph training (1,393 against 1,333 positions/s eager).
  A 3.5 GiB VRAM ceiling keeps the desktop responsive at full speed.

## v7 graded game-position experiment (registered 2026-09-23, before any result)

Goal: a stronger general player, and a test of whether extra recurrent steps
improve move quality in ordinary positions, judged the way Lichess judges moves.

Data:

- Lichess evaluation database. Per position, use the evaluation with the most
  principal variations among those at depth 20 or more (ties go to the deeper
  one). Keep positions with at least 3 scored candidate moves. In a
  200,000-record sample, 68% of positions have 3 or more candidates in some
  evaluation.
- Split by normalized-FEN hash (98/1/1). Exclude frozen 50k puzzle FENs.
  About 10M training positions. Puzzles are not used for training.

Targets, all from the side to move:

- Each candidate's win chance W on Lichess's curve,
  `W = 50 + 50 * (2 / (1 + exp(-0.00368208 * cp)) - 1)`, mates at the limits.
- Policy target proportional to `exp(-(W_best - W_i) / 4)` over the scored
  candidates. Unscored legal moves get zero mass.
- The per-move value head regresses each scored candidate's `2W/100 - 1`. The
  position value head regresses the best candidate's.
- Loss: policy cross-entropy + 0.5 * candidate value MSE + 0.25 * position value MSE.

Model and training: the v6 normalized design, trained from scratch (one
prelude block, two-block shared core, d 256, 16 scratch tokens, per-square
readout). It uses the v6 progressive objective (`n` in `0..15`, `k` in
`1..min(8, 16 - n)`).
60,000 steps at batch 128. Model snapshots every 5,000 steps. Control: the same
run trained only at T=1. Each run is launched as its own job.

Evaluation, frozen before training:

- 2,000 held-out game positions, each move judged by Stockfish at 20,000 nodes
  as win chance lost against Stockfish's best move. Good is below 5 points,
  mistake 10 to 15, blunder 15 or more. Per-move Lichess accuracy is
  `103.1668 * exp(-0.04354 * lost) - 3.1669`. Reported at T = 0, 1, 2, 4, 8,
  16, 32, 64.
- Puzzle rating from 1,000 whole held-out puzzles (Elo fit), same depths.
- Games against Stockfish with `UCI_LimitStrength` at several `UCI_Elo`
  levels, 100 paired games per level at T1 and the best T. Reported, not gated.

Gates, paired 95% intervals on the 2,000 game positions:

- G1: Lichess accuracy at T=16 exceeds T=1 by at least 2 points, and the
  good-move rate is higher.
- G2: accuracy at T=32 and T=64 is no more than 1 point below T=16.
- G3: the recurrent model at its best T beats the T=1 control on accuracy.
- G4: the blunder rate at the best T is below the v6 step-8,000 checkpoint's
  on the same positions, measured before v7 training starts.

Amendment (2026-09-23, before any v7 training or result):

- Every legal move is scored for 4,000,000 training positions drawn at random
  (seed 0). One Stockfish search per position covers all legal moves at 50,000
  nodes (about 68 ms). Deep Lichess scores are kept for the candidates. Quick
  scores fill the other legal moves, capped at the worst deep candidate. The
  policy target, per-move values and their mask then cover every legal move.
  The position value stays the best deep candidate.
- The main run and the T=1 control both train only on the labelled
  positions, for 120,000 steps at batch 128 (about 3.8 passes).
- Reason: with 3 to 5 scored candidates, unscored moves got zero target and
  no value error. The model never learned how bad the remaining moves are,
  which is exactly what blunder avoidance needs.
- Values are reported in pawns through the inverse of Lichess's curve,
  `pawns = ln(W / (1 - W)) / 0.368208`.
- Evaluation and gates are unchanged.

Second amendment (2026-09-23, before any v7 training or result). A literature
survey found no case where plain policy and value supervision made extra
recurrent depth pay off. Depth paid off when targets required predicting the
future (DiffuSearch, KataGo's opponent-reply target, lookahead on future-move
squares in Leela). It also paid off when every iteration was supervised (HRM
and TRM deep supervision, Deep Thinking). v7 therefore adds:

- Lookahead targets on the square tokens. For labelled positions the labeller
  stores Stockfish's line for the deep-best move (up to 4 plies). It keeps the
  reply's from and to squares, and the piece on every square after ply 2 and
  ply 4.
  Heads on the square tokens predict the reply's from and to squares (two
  64-way softmaxes). They also predict the 13-way piece class of each square
  after plies 2 and 4. These terms weigh 0.25 in total. Positions labelled before this amendment have
  no line and are masked out of these terms.
- Grounded imagination. The ply-2 board prediction uses the board
  encoder's own piece embeddings as the classifier. An imagined square
  then lives in the same space as a real one.
- Every step of the gradient tail is supervised, with weights rising linearly
  so the last step counts most. The no-gradient prefix is unchanged.
- An exponential moving average of the weights (decay 0.999) is kept and used
  for checkpoints and evaluation.
- A learned 64 x 64 attention bias per head between square tokens.
- Gate G5: at its best T, v7 beats an equal-parameter feedforward model on
  Lichess accuracy, paired 95% interval. That model has four untied
  transformer blocks, the same heads, targets, data and step budget, and is
  read out once. Strength against
  iterations, including stopping once the policy stops changing, is reported.

### v7 outcome (2026-09-23): stopped at step 53,000 of 120,000

- Data: the first 2,499,551 labelled positions (500 of 800 label chunks, frozen
  as `train-labels-2500k`). 82% carry reply and ply-2 targets, 68% ply-4. The
  run needed `--max-vram-gb 4.5` (3.1 GiB eager peak against 2.7 for v6) and
  trained at 1,450 to 1,560 positions/s.
- Validation top-1 at step 53,000: T0 24.3%, T1 49.3%, T8 62.1%, T16 62.3%,
  T32 61.6%. The T8 minus T1 gap grew from 5 points at step 10,000 to 13.
- Strength at the step-10,000 snapshot (registered 2,000 positions, 20,000
  nodes): Lichess accuracy 70.1 at T1 and 73.8 at T8. Good moves 60.7% and
  blunders 23.1% at T8, against 48.9, 32.4% and 53.5% for v6. T16 minus T1 is
  2.9 accuracy points (95% interval 1.8 to 4.1). T32 is 0.7 and T64 is 1.4
  points below T16. Puzzle rating fell from about 1,040 to about 690.
- Strength at step 53,000 (same report): accuracy 73.1 at T1 and 81.6 at T8.
  T8 minus T1 is 8.5 points (95% interval 7.2 to 9.8). Blunders 23.8% at T1
  and 14.1% at T8, puzzle rating 745 at T1 and 1,076 at T8. T32 is 0.4 and
  T64 2.0 points below T16. The gain from iterating grew with training.
- 100 paired games against Stockfish 19 at `UCI_Elo` 1320, T8, 0.1 s per move.
  Step 10,000 scored +1 =11 -88 (about 860 Elo, 725 to 935). Step 45,000
  scored +5 =31 -64 (about 1,085 Elo, 1,014 to 1,141).
- Diagnosis of the remaining blunders (step-10,000 games judged at 100,000
  nodes): 7.8% of moves lose 15 or more points. 97 of 100 games contain
  one, half by ply 8. The model leaves a piece en prise in 15.6% of its moves
  against 4.5% of target moves. Linear probes on the square tokens at step
  45,000 read attacked squares, hanging and pinned pieces at AUROC 0.97 to
  0.99 by T4. The board facts are present. The lookahead heads did not
  learn to imagine. On held-out positions they get 11% of the squares that
  change within two plies right, and no whole board. That is close to copying
  the current board. Extra steps refine one picture of the present position.
  Nothing in the state holds a position after a move.
- Reason for stopping: these results answer what v7 was run to learn. The T=1
  control was not run, so G3 is not evaluated.
- G5 fails. The feedforward baseline (four untied blocks, 3,364,258
  parameters, read out once) runs on the matched schedule. At step 53,000 it
  scores 84.9 accuracy, 75.7% good moves, 10.5% blunders and a puzzle rating
  of about 1,406. It beats v7 at T8 by 3.3 accuracy points (95% interval 2.3
  to 4.3) and by 3.6 points of blunder rate. It uses about an eighth of the
  compute per move. v7's recurrence reaches less than the same parameters stacked once.

## v8 board-shaped scratch space (registered 2026-09-23, before any v8 code or result)

Question: given free working space shaped like chess boards, does the
recurrent core discover its own way to calculate? The space is shaped, not
directed: no rules, no candidate selection, no targets on its contents.

Design: v7 at step 53,000 plus four scratch boards. Everything else is the
same (data `train-labels-2500k`, targets, losses, every-step supervision, EMA,
square bias, 16 scratch tokens, batch 128, progressive objective).

- Each scratch board is 64 tokens. A token's input is the shared square
  embedding of its square plus a learned embedding of its board. A learned
  content vector shared by all squares is added. Nothing depends on the
  position, so every board starts the same.
- The recurrent core attends over the real board, the hidden squares, the
  scratch tokens and all scratch boards. The learned 64 x 64 square bias
  applies between any two square tokens, real or scratch. A square's
  geometry is then the same on every board.
- Readouts, heads and targets read only the hidden square and state tokens,
  exactly as in v7. The per-square move readout is left unchanged so that the
  scratch boards are the only difference.
- Batch stays 128. If a run does not fit under 4.5 GiB, activations are
  recomputed per recurrent step in the backward pass, which leaves the
  numbers unchanged.

Runs, each its own job, 53,000 steps with snapshots every 5,000:

- v8: as above.
- v8-flat control: the same token count (256 extra tokens) with no square
  embedding and no square bias on them. It separates board shape from extra
  space and compute.

Evaluation, frozen now: the v7 strength report (2,000 positions, 20,000
nodes, T 0 to 64) and 100 paired games against Stockfish 1320 at T8. Both
compare against v7's step-53,000 checkpoint. Linear probes on the scratch
boards are fitted after training on held-out positions. They test whether the
boards encode the current board, the board after the model's own move, or the
board after Stockfish's reply, and how that changes with T.

Amendment (2026-09-23, before any v8 training): v7 stopped at step 53,000 on
a 120,000-step learning-rate schedule. Every run compared with it uses
`--steps 120000 --stop-after 53000`. The feedforward baseline (G5) was started
on a 53,000-step schedule, stopped at step 13,000 and resumed on the matched
schedule. Its learning rate over the first 13,000 steps was up to 7% lower.
At step 10,000 it already scored 75.8 accuracy and 20.5% blunders against
v7's 73.8 and 23.1% at T8.

Second amendment (2026-09-23, before any v8 training). Two findings change
the design. First, the feedforward baseline matched v7's step-53,000 strength
by its own step 25,000 (82.0 against 81.6 accuracy, paired difference 0.5,
interval -0.4 to 1.4). Beating v7 is no longer enough. Second, v7 ignores
its 16 scratch tokens. Resetting them after every step changes the T8
good-move rate from 62.7% to 62.5%. Its memory between steps lives entirely
in the 64 square tokens, so added scratch space has no reason to be used.

- v8 resets the square and state tokens to the board encoding before every
  step (`--refresh-board`). Only the 16 latent tokens and the scratch board
  persist, so any gain from more steps has to pass through the scratch space.
  What goes there is still left to the model.
- One scratch board instead of four, which costs about 1.8 times v7 per step.
- Pilot first: 10,000 steps with `--steps 120000 --stop-after 10000`,
  compared at step 10,000 with the feedforward baseline (75.8 accuracy, 20.5%
  blunders) and v7 (73.8 and 23.1% at T8). Continue to 53,000 steps and run a
  flat control only if the pilot's best T beats the feedforward baseline on
  Lichess accuracy. The pilot's T8 minus T1 is reported as the measure of how
  much it uses the scratch space.

Gates, paired 95% intervals on the 2,000 positions:

- H0: v8 at its best T beats the feedforward baseline at the same step on
  Lichess accuracy. This is now the main gate.
- H1: v8 at its best T beats v7 at step 53,000 on Lichess accuracy, and its
  blunder rate is lower.
- H2: v8 beats v8-flat on Lichess accuracy at their best T.
- Reported: T16 minus T1 for all three models and the probe results.

### v8 pilot outcome (2026-09-23): stopped at step 10,000, H0 fails

- One refreshed-board scratch board trained at about 380 to 480 positions/s
  with `--recompute-steps` under 4.5 GiB.
- Strength at step 10,000: accuracy 74.7 at T1 and 75.0 at T8. Good moves
  61.0% and 61.2%, blunders 21.9% and 21.3%, puzzle rating about 717 and 739.
  T8 minus T1 is 0.3 accuracy points (95% interval -0.2 to 0.8).
- The feedforward baseline at the same step leads by 0.8 accuracy points
  (interval -0.3 to 1.8) and 2.3 points of good-move rate (0.7 to 4.0).
- Scratch tokens were the only memory between steps, and still the model did
  not learn to carry anything useful through them. It became a slightly
  weaker single-pass network. By the registered rule, v8 does not continue.
- What the scratch space held (step 10,000, 4,096 validation positions):
  wiping the latent tokens and the scratch board after every step lowers the
  good-move rate at T2 from 52.0% to 50.9%. It changes nothing at T1. After
  step 1 a linear probe reads the opponent's attacked squares from the scratch
  board at AUROC 0.90, fading to 0.64 by T8. By T8 the state has stopped
  moving. The scratch board changes by 0.5% per step, the squares by 0%.

## v9 fixed-horizon pilots (registered 2026-09-23, before any v9 training)

Reason: the `reasoner` project met the same symptoms (a carried state that
resets without loss, accuracy flat after T1). There every example drew a
random horizon and every step was graded against the final answer. That
objective rewards answering at step 1 and holding. The recorded rule from that
project is that no signal may reward early answer correctness. v7 and v8 used
exactly that objective.

Two pilots, identical to v7 except for the objective and, in B, one scratch
board. Every example runs exactly 8 recurrent steps with gradients through
all of them, and only the final state is graded (`--fixed-depth 8
--no-supervise-every-step`). The board is encoded once and not refreshed.
Both use `--steps 120000 --stop-after 10000` and are evaluated with the
registered strength report at step 10,000.

- A: v7 with the fixed horizon.
- B: A plus one scratch board (`--scratch-boards 1`), with activations
  recomputed if needed to fit under 4.5 GiB.
- C: A plus Deep Thinking's progressive loss, registered before C runs. Half
  the batches instead run a random no-gradient prefix of 0 to 8 steps and a
  tail of 4 to 8 steps graded only at its end. They never run beyond 12 steps
  in total (`--progressive-share 0.5 --min-gradient-steps 4 --max-depth 12`).
  The same step must then improve whatever state it is given. Question for C:
  does accuracy keep rising past the depths it trained on? C passes that test
  if its T16 beats its T8 on Lichess accuracy with a paired 95% interval above
  zero. T16 is never trained.

Gate: a pilot continues to 53,000 steps only if its best T beats the
feedforward baseline at step 10,000 (75.8 accuracy) on Lichess accuracy. The
paired 95% interval must be above zero. Reported for each pilot: T8 minus T1,
the cost of wiping the latent tokens (and in B the scratch board) after every
step, and whether the state still changes at T8.

Pilot A outcome (2026-09-23): stopped by decision at step 7,000, before its
strength report. On the validation good-move rate it led the feedforward
baseline at every step from 2,000 (40.5% at T8 against 39.3%) to 7,000 (52.1%
against 51.0%). Its answer builds up over the steps (T1 34.9%, T4 49.6%, T8
52.1% at step 7,000). It degrades past the trained depth (T16 45.6%, T32
32.0%), which is the gap pilot C targets. Its strength report at step 7,000:
accuracy 63.1 at T1, 74.2 at T8, 71.5 at T16 and 62.8 at T32, blunders 22.8%
at T8. The feedforward baseline's step-10,000 report is 1.6 points higher
(interval 0.5 to 2.6), a comparison 3,000 steps apart.

Pilot C outcome (2026-09-23): stopped by decision at step 4,000. The
progressive half made long runs stable (T32 42.7% against A's 29.9% at step
4,000). It never made T16 better than T8 (45.4% against 46.2%). It cost T8
1.9 points against A and 1.6 against the feedforward baseline. It learned
to hold its answer, not to keep improving it. With a fixed target per
position, most of the reachable accuracy is available within 8 steps, so
further steps have nothing left to find.

Pilot B was not run. A scratch board only matters once extra steps pay off,
which v10 addresses first.

## v10 calculation where it is needed (registered 2026-09-23, before any v10 code or result)

Question: when positions that need more calculation get more steps, does the
recurrent core learn a way of calculating? Such a way would keep improving its
answer on harder positions than it trained on, by thinking longer.

Design: pilot A's model and objective (every batch runs its full horizon with
gradients and is graded only at its end), trained on a mix:

- Three batches in four are game positions from `train-labels-2500k`,
  graded as in v7 at 8 steps.
- One batch in four is puzzle positions from `puzzles-v1/train` with 1 or 2
  solver moves remaining (drawn with equal chance per batch). They are graded
  on the solver's move at 4 steps per remaining move: 4 steps for 1, 8 for 2.
  Puzzle batches carry no value, move-value or lookahead targets. Those
  terms are masked, not set to zero targets.
- `--steps 120000 --stop-after 10000`, snapshots every 5,000.

Evaluation, frozen now:

- Held-out puzzles from `puzzles-v1/test` grouped by solver moves remaining:
  1 and 2 (trained), 3, 4, and 5 or more (never trained). First-move accuracy
  at T 1, 2, 4, 8, 12, 16, 24 and 32, each T paired with T8.
- The game-position strength report at T8, as a guardrail on general play.
- The same puzzle breakdown for pilot A's step-7,000 checkpoint and the
  feedforward baseline's step-10,000 snapshot, for reference.

Gates:

- E1 (the question): on puzzles with 3 or more moves remaining, the best T
  above 8 beats T8 on first-move accuracy with a paired 95% interval above
  zero.
- E2 (guardrail): game-position Lichess accuracy at T8 is no more than 2
  points below the feedforward baseline at step 10,000 (75.8).

References on the same test puzzles (first-move accuracy, 3+ group): pilot A
at step 7,000 peaks at T8 with 31.9% (T16 28.6%, T32 19.9%). The feedforward
baseline at step 10,000 scores 34.1%.

### v10 outcome (2026-09-23): stopped at step 7,000

On validation puzzles (512 per group) the model tied its thinking time to
difficulty. 1-move puzzles peak at T4 and 2-move puzzles at T8, their trained
horizons. Every longer group also peaked at T8 at every row from step 4,000
to 7,000. At 7,000: 3 moves 38.3% at T8 against 34.2% at T16, 4 moves 39.5%
against 32.0%, 5+ moves 43.6% against 35.0%. It learned to spend its longest
trained horizon on hard puzzles, not to keep improving past it. That points
to counting steps. Game validation at step 7,000 was 50.2% at T8 against
pilot A's 52.1%. Stopped once the pattern was stable.
The registered puzzle-depth report at step 7,000 confirms it. On all 27,830
test puzzles with 3 or more moves, accuracy is 39.4% at T4, 40.3% at T8,
37.3% at T12 and 31.8% at T16. T12 minus T8 is -3.0 points (interval -3.4 to
-2.6), so E1 fails. The puzzle mix still helps on those untrained lengths:
40.3% at T8 against 34.1% for the feedforward baseline and 31.9% for pilot A.

## v11 no step counting (registered 2026-09-23, before any v11 training)

v10 plus `--random-prefix 4`. Every batch, game or puzzle, first runs 0 to 4
extra no-gradient steps. Then it runs its usual tail (4 steps for 1-move
puzzles, 8 for 2-move puzzles and games), graded at its end. The graded step
then varies by up to 4 within each difficulty. The answer cannot be tied to a
step number and has to be reached and held from the position itself. Validation
adds T12 and T24. `--steps 120000 --stop-after 10000`.

Gates: E1 and E2 as in v10. Early stop: if at step 5,000 every validation
puzzle group of 3 or more moves still peaks at T8 or earlier, stop and
record.

### v11 outcome (2026-09-23): stopped at step 5,000, E1 fails

Validation groups of 512 puzzles suggested later peaks for harder puzzles at
step 2,000. The full puzzle-depth report at step 5,000 does not support
that. On all 27,830 test puzzles with 3 or more moves, accuracy is 35.1%
at T8, 34.3% at T12 (difference -0.8, interval -1.1 to -0.5), 33.6% at T16
and 32.0% at T24. The random prefix did remove most of v10's decline past
the trained horizon (v10: -3.0 at T12 and -8.5 at T16). The 5+ group is
flat through T24. The model now holds its answer but still does not
improve it with more steps. Game validation at step 5,000 was 45.0% at T8
against v10's 47.9%.

## v12 local recurrent steps (registered 2026-09-23, before any v12 training)

Reason: in every design so far a single recurrent step can attend across the
whole board. Whatever the model can compute fits in one or two steps, and
further steps only hold the answer. The recurrent models that did improve
with more steps limited what one step can do. Deep Thinking's chess-puzzle
networks used 3 x 3 convolutions. `reasoner` allowed one memory read per
step, so larger problems needed more steps.

v11's objective and data with a local core. The objective is fixed depth 8
graded at the end, a random prefix of 0 to 4, and a quarter puzzle batches with
1 or 2 moves remaining at 4 steps per move.

- `--local-attention`: each recurrent square token attends only to squares
  within one king step, on the board memory and in H, plus the board's state
  token. It cannot read H's state token, which still reads the whole board
  for the value head.
- `--prelude-blocks 0` and `--latent-tokens 0`, so no global computation
  precedes or bypasses the local steps. This leaves 2,368,422 parameters.
- `--steps 120000 --stop-after 10000`, validation at T 0 to 32 with T12 and
  T24.

Gates: E1 and E2 as in v10. Early stop: the registered puzzle-depth report
is run on the step-5,000 snapshot. If on the 3+ group no T above 8 beats T8
with a paired interval above zero, stop and record.

### v12 outcome (2026-09-23): stopped at step 5,000, E1 fails

A first launch crashed during CUDA-graph capture (the king-neighbourhood mask
was a CPU tensor). It now is a device buffer and a GPU test covers local
attention. On all 27,830 test puzzles with 3 or more moves the step-5,000
snapshot scores 36.0% at T8, 35.0% at T12 (difference -0.9, interval -1.2 to
-0.6), 34.3% at T16 and 33.6% at T24. The local core is still competitive.
At step 5,000 it beats v11 on those puzzles (36.0% against 35.1% at T8) and on
game validation (46.8% against 45.0% at T8). It has 2,368,422 parameters and
about twice v11's training speed.

Across v9 to v12 every model reaches its answer by the latest graded step and
at best holds it. No training example has ever rewarded steps taken after
the model could already answer, which is the behaviour being tested for.

## v13 steps after the answer (registered 2026-09-23, before any v13 training)

v12's model and game batches, with puzzle batches of 1 to 4 moves remaining,
each run for 4 steps per move (4, 8, 12, 16). Gradient tails are capped at 8
steps (`--puzzle-max-tail 8`). The rest runs first without gradients. A
3-move puzzle therefore runs 4 no-gradient steps and an 8-step tail. A
4-move puzzle reaches the model's own T8 state and must improve on it over 8
more trained steps. Random prefix 0 to 4 as in v11 and v12. Puzzles with 5 or
more moves stay untrained. `--steps 120000 --stop-after 10000`.

Gates, on the registered puzzle-depth report (depths extended with 20):

- F1: on the 4-move group, T16 beats T8 with a paired 95% interval above zero.
- F2: on the 5+ group, some T above 16 beats T16 with an interval above zero.
- E2 as before.

Early stop at step 5,000 if F1's difference is not above zero on the
precise report.

v13 at step 5,000 (registered puzzle-depth report, all test puzzles):

- F1 passes: on 4-move puzzles T16 beats T8 by 3.9 points (interval 2.8 to
  4.9). The same holds for 2 moves (+4.4, 3.8 to 4.9) and 3 moves (+3.6, 3.0
  to 4.1). This is the first Wombat model whose answers improve measurably
  with more steps.
- F2 not yet: on the untrained 5+ group T24 beats T16 by 1.1 points (interval
  -0.1 to 2.2). Against T8 the same group gains 1.5 at T16 and 2.6 at T24
  (0.8 to 4.3), so it improves up to its longest trained horizon and holds.
- 1-move puzzles lose 2.6 points from T8 to T16. Easy positions are
  overthought, so a stopping rule will be needed.
- On the 3+ group its best depth reaches 38.5% (T16) against v12's 36.0% at
  the same step. Training continues to step 10,000 by the registered rule.

### v13 outcome (2026-09-23): pilot complete at step 10,000

- F1 passes: T16 minus T8 is +4.6 (2 moves), +5.3 (3 moves) and +4.4 (4
  moves, interval 3.3 to 5.5). The untrained 5+ group also gains +3.4 (1.8
  to 5.0) from T8 to T16. On the 3+ group accuracy rose from 38.5% at step
  5,000 to 50.1% at T16.
- F2 fails: past T16 nothing improves (5+ group: T20 -1.0, T24 -1.1 and T32
  -1.6 against T16). The model improves up to its longest trained horizon.
- E2 passes: game-position Lichess accuracy 76.1 at T8 and 76.8 at T12
  against the feedforward baseline's 75.8 at step 10,000 (T12 difference
  +1.0, interval -0.2 to 2.3). Blunders 19.7% at T12. Game positions are
  overthought past their trained horizon of 12 (73.3 at T16, 68.9 at T24).
  Puzzle rating about 1,135 at T16 against the baseline's 805.

## v14 longer horizons everywhere (registered 2026-09-23, before any v14 training)

v13 with `--random-prefix 8` instead of 4. Every batch then continues from a
state the model reached without gradients for up to 8 extra steps. Game
positions are graded between steps 8 and 16, puzzles between 4r and 4r + 8
(up to 24 for 4 moves). Two questions: whether game positions stop being
overthought, and whether the longer trained horizons move the puzzle peak
beyond 16. Everything else as v13, pilot to step 10,000.

Gates at step 10,000:

- H-games: game Lichess accuracy at T16 is no more than 0.5 below its best T
  (paired interval), and the best T is at least as good as v13's 76.8.
- F1 as in v13. F2': on the 5+ group some T above 16 beats T16.
- Early stop at step 5,000 if game validation at T16 is still more than 3
  points below T8.

### v14 outcome (2026-09-23): pilot complete at step 10,000

- Games are no longer overthought: Lichess accuracy 75.9 at T8, 76.2 at T12,
  75.8 at T16 and 75.3 at T32 (T16 minus T12 -0.4, interval -0.9 to 0.1).
  v13 fell from 76.8 at T12 to 68.9 at T24. Its best T is 0.5
  below v13's (interval -1.4 to 0.4). H-games' second clause is not met
  on the point estimate, but the two are statistically tied.
- Puzzles never lose accuracy with more steps (1-move 64.2 at T8, 63.1 at
  T32). On the 3+ group T16 beats T8 by 2.9 (2.5 to 3.2) and T32 beats T16 by
  0.6 (0.3 to 0.8). On 4-move puzzles T32 beats T16 by 0.9 (0.3 to 1.5).
  F2' on the untrained 5+ group: T32 minus T16 +0.3 (-0.6 to 1.1), not met.
- The price of the longer prefixes is peak depth gain: 3+ best 48.4% against
  v13's 50.2%, and T16 minus T8 +2.9 against v13's +5.0.
- v14's recipe is the base for the next pilots and the long run. It keeps
  v13's strength within noise, and more steps never make it worse.

## v15 readout and neighbourhood pilots (registered 2026-09-23, before any v15 training)

Two 5,000-step pilots on v14's recipe (`--stop-after 5000`), each changing
one thing:

- v15a `--move-readout from_to`: a move's logit and value add a query-key
  score between its origin and destination square tokens. A move onto an
  attacked square can then see that square. Motivated by v7's blunder analysis
  (pieces moved onto attacked squares, defenders moved away).
- v15b `--local-neighbourhood king_knight`: local attention also covers
  knight jumps, so every piece's shortest reach is one block.

Each is compared with v14's step-5,000 snapshot on the registered strength
report and puzzle-depth report. A change goes into the long run under two
conditions. Its best-T Lichess accuracy is not below v14's (paired point
estimate). Its 3+ puzzle accuracy at best T is no more than 0.5 below v14's.

Amendment (2026-09-23, before v15b training): v15a's game validation led v14
by about 10 points at every step from 1,000 to 3,000 (51.5% against 41.9% at
T12, step 3,000). from_to is expected to pass and the remaining pilots
build on it. v15b is v15a's recipe with `--local-neighbourhood king_knight`.
v15c is v15a's recipe with `--board-features`, 31 per-square inputs (attacker
counts and cheapest attacker for both sides, pins for both sides, and check).
Each is compared with v15a at step 5,000 under the same rule.

Second amendment (2026-09-23, before v15c or v15d training): v15c is not run on
its own. A research review supports inputs that are global or long range
under local attention. It covered AlphaVile's +97 Elo input package, Stockfish
18's +27 Elo threat inputs, Giraffe's attack maps and DeepMind's play-time
repetition rule. v15d is v15a's recipe plus `--board-features` and
`--board-features-long-range` (sole defenders, checking squares, check lines
and discovered-check pieces, x-ray attackers, material per piece type). It adds
`--local-neighbourhood king_knight` if v15b passes, and is compared with v15a
at step 5,000 under the same rule. The two feature sets are tested as one
package. They are inputs the model can ignore, and they add under 1% of
parameters. This supersedes the rule under "Model" not to hand-encode attacks
or threats. Separately, play-time inference forces the halfmove bucket to 0
(training data never sets it). It also avoids stalemating or
threefold-repeating moves when the model's value says it is winning.

### v15a and v15b outcomes (2026-09-23), step 5,000

- v15a (from_to) is accepted. Lichess accuracy 77.6 and blunders 18.9% at
  T12 against v14's 71.0 and 26.3% at the same step (paired +6.6, interval
  5.3 to 7.9). It is +1.8 (0.6 to 3.1) above the feedforward baseline at step
  10,000. Puzzle rating about 1,268 against 896. On 3+ puzzles 49.9% at T8,
  53.8% at T16 and 54.6% at T32 against v14's 34.5, 36.4 and 36.9. T16 minus
  T8 +3.9 (3.6 to 4.3) and T32 minus T16 +0.8 (0.5 to 1.1). Games lose 2.1
  points from T12 to T32.
- v15b (king_knight on top of v15a) is not included. Every paired accuracy
  difference against v15a lies within 0.4 points with intervals spanning zero
  (best T 77.5 against 77.6). 3+ puzzle curves agree within 0.4 points. A
  knight jump is two king steps, which one recurrent step of two blocks
  already covers.
- The play fixes are committed (halfmove bucket 0 at play, a stalemate and
  repetition guard). v15c (v15a plus the 31
  basic board features) runs while the long-range features are built.
- v15c (v15a plus the 31 basic board features) is accepted. At step 5,000:
  Lichess accuracy 80.8 and blunders 14.4% at T12 (paired +3.2 over v15a,
  interval 2.2 to 4.2, blunders 4.5 points lower), puzzle rating about 1,430.
  3+ puzzles 55.1% at T8, 58.8% at T16 and 59.9% at T24 (about 5 points
  above v15a at every depth). T16 minus T8 +3.7 (3.4 to 4.1) and T32 minus
  T16 +0.9 (0.5 to 1.2), so the features keep the gain from more steps.
  Games lose 3.3 points from T12 to T32. Against the feedforward baseline's
  step-53,000 report it is still 4.1 points behind (interval 3.0 to 5.1)
  after a tenth of the steps.
- v15d (v15c plus the long-range board features) is accepted. At step 5,000:
  Lichess accuracy 81.5 and blunders 14.0% at T8 (paired +1.1 over v15c at
  T8, interval 0.3 to 1.8, and +1.6 at T32, 0.7 to 2.5). Puzzle rating about
  1,511 against 1,430, and 3+ puzzles 59.9% at T8 and 63.3% at T20 against
  v15c's 55.1% and 59.7%. T16 minus T8 +3.2 (2.8 to 3.5). Past T16 it holds
  (T32 minus T16 -0.2, -0.5 to 0.1). Games lose 2.4 points from T8 to T32.

## v16 long run (registered 2026-09-23, before training)

v15d's recipe on all 3,999,041 labelled positions (`train-labels`).
That is local attention without prelude or latent tokens, from_to readout, and
basic and long-range board features. Depth is fixed at 8 graded at the end,
with a random prefix of 0 to 8. A quarter of batches are puzzles with 1 to 4
moves at 4 steps per move and an 8-step gradient cap.
`--steps 120000 --stop-after 53000`, snapshots every 5,000 steps.

Evaluation at step 53,000, frozen now: the registered strength report and
puzzle-depth report, and 100 paired games against Stockfish 1320 with the
play-time fixes on, at the best T from the strength report.

Gates, paired 95% intervals on the 2,000 positions:

- L1: best-T Lichess accuracy beats the feedforward baseline at step 53,000
  (84.9).
- L2: on 3+ puzzles T16 beats T8.
- Reported: blunder rate, puzzle rating, Elo, and T32 against the best T.

Match rules corrected (2026-09-23): games against Stockfish ended
as soon as the side to move could claim a threefold repetition, including by
a move it had not played. The play-time guard only avoided repeating a
third time itself. Matches now draw only after a real third occurrence or at
fifty moves, as Cute Chess adjudicates. The guard also avoids moves that
let the opponent claim a draw. Earlier match numbers used the old rules.

Step-5,000 matches, 100 paired games against Stockfish 1320 at T8:

- v15d, old rules: +10 =41 -49, score 0.305, about 1,177 Elo (1,121 to
  1,227). 26 of the 41 repetition draws came with Wombat ahead in material.
- v15d, corrected rules: +32 =11 -57, 0.375, about 1,231 (1,158 to 1,297).
- v16, corrected rules: +43 =4 -53, 0.450, about 1,285 (1,215 to 1,353).
  v16 and v15d have the same position accuracy at step 5,000 (61.0 against
  60.9 on validation), so the match difference is within noise.

The long run then uses v14's recipe, the accepted changes and all 3,999,041
labelled positions (`train-labels`), with `--steps 120000 --stop-after
53000`. It is compared with the feedforward baseline at step 53,000 (84.9
accuracy, 10.5% blunders).

Step-12,000 ladder (v16, T8, corrected rules, Stockfish limited to 50,000
nodes per move instead of time), 100 paired games each. Against
UCI_Elo 1320 score 0.560 (about 1,362), 1500 score 0.280 (about 1,336), 1700
score 0.150 (about 1,399). Full-strength Stockfish at 50,000 nodes won all
100. The three anchors agree on roughly 1,350 to 1,400.

## v17 pilot: joint reply head (registered 2026-09-24, before training)

v16's play shows the reply prediction is weak. The square reply heads pick a
legal reply that matches Stockfish's about 21% of the time. The policy
itself on the position after the move gets 41%. v17 is v16's recipe plus the
joint reply readout (`--reply-readout pair --reply-weight 0.25`). It scores
every origin and destination pair, so it can only name real moves.
5,000 steps, compared with v16's step-5,000 copy.

Accepted if `evaluate-reply` legal agreement with Stockfish's reply rises
by at least 10 points. The strength report at the best T must not be worse
(paired interval above -0.5). Also reported: the T16 minus T8 gain on 3+
puzzles.

v17 result (step 5,000, against v16's step-5,000 reports): not accepted.
Legal reply agreement with Stockfish 25.1% (22.5 to 27.9) against 20.6%,
+4.5 points, short of +10. The policy asked on the position after the move
still gets 37.3%. Best Lichess accuracy 81.5 at T8 against 81.6. It holds up
better with more steps: paired +0.8 at T32 (0.05 to 1.55) with 1.2 points
fewer blunders. Puzzles +0.5 at T8, +0.7 at T16 and +0.9 at T32. T0 and
T1 rise by about 5.7 points. Kept as the fallback if depth training does
not stop the decline past T8.

## v18 pilot: move-graph attention (registered 2026-09-24, before training)

Local attention lets a square read only its king neighbours, so a bishop's
effect on a far square has to travel one square per block. v18 adds the
lines pieces actually act along. Square i also reads square j when the
piece on either attacks or can push to the other, stopping at the first
blocker (`--local-neighbourhood move_graph`). No new parameters. 5,000
steps against v16's recipe (v17's head too if v17 is accepted).

v18b (`--local-neighbourhood move_lines`) lets sliding pieces reach along
their whole lines through blockers. A bishop, the pinned piece and the
queen behind it are then one hop apart. It adds a learned per-head bias for
the number of pieces between the two squares (0, 1, 2+), since attention
between two squares cannot see the squares in between. It runs beside v18
from the same seed and step count.

Order and baseline (2026-09-24): v16 stopped at step 28,000 (resumable).
v18b runs first, alone, with v15d's exact command plus `--local-neighbourhood
move_lines`, so it pairs with v15d's step-5,000 reports. v15d and v16 differ
only in the label pool and matched on validation at step 5,000.

v18b is rejected. At step 5,000 its best Lichess accuracy is 81.6 at T12
against v15d's 81.5 at T8. Paired differences at T4, T8, T12, T16 and T32
are +0.2, -0.1, +0.3, +0.5 and +0.2, every interval spanning 0. Blunder
rates are within 0.5 points. On the puzzle test split (69,280 positions) it is
0.3 to 0.6 points lower overall and 1.0 lower on 2-move puzzles at T8 and
T16. 3+ puzzles are level. Lines through pieces add nothing on top of the
long-range board features, so v18, which sees strictly less, is not run.
v16's step-5,000 strength report (81.6 at T8, 13.9% blunders) is the
baseline for later pilots on the full label pool.

Each is accepted if best-T Lichess accuracy improves (paired interval above
0). Between them, the stronger one wins. Playing strength decides. Needing
fewer steps for the same answer is a gain, not a cost. Reported: the 3+ puzzle T16 minus T8 gain and the T at
which accuracy peaks.

## Goal and backlog (2026-09-24)

Goal: beat DeepMind's 9M searchless transformer (ChessBench, 2024) on the
measure that can be reproduced here. That is full-solution accuracy on 10,000
Lichess test puzzles, where every solver move must match, 88.9% for their
9M. Wombat at step 5,000 solves 59.9% at T8 and 62.3% at T16 on 10,084
test puzzles. Those are the puzzles whose every position is in the
puzzle-depth report (mean rating 1,529, a biased subset). Playing strength
against the Stockfish ladder at 50,000 nodes stays the second measure. Their
Lichess bot Elo (2,054) is not reproducible here. Each change below is
registered with its gate before its run, and accepted only on strength (best
T) or puzzles.

Why it is plausible: their 9M saw about 41 billion training examples and
checks each legal move with its own forward pass. Wombat has seen about 3.6
million and thinks in one pass. On puzzles, T1 to T16 lifts full-solution
accuracy from 34% to 62%, which is the compute they buy with one pass per
move.

Backlog, in order, one GPU job at a time with code written meanwhile:

1. Cooldown: v16 resumed from step 28,000 with a 6,000-step `1 - sqrt` decay
   (`runs/v16-cool`). Measures how much strength the never-annealed schedule
   hides and whether the plateau is capacity.
2. Depth training (Huginn): lognormal total depth, mean 16, max 48, last 8
   steps with gradients, with and without initial state noise, against v16's
   step-5,000 reports. Target: stop the 3-point decline from T8 to T32.
3. Move choice by per-move value (ChessBench action values): policy, value
   and a mix, on the strength report, with no training.
4. Full-solution puzzle evaluation on a fixed 10,000-puzzle sample of the
   test split, the goal's yardstick.
5. Won-position labels: every mate maps to exactly 1.0 and anything above
   about +15 pawns to 0.996, so progress is not rewarded. In the step-12,000
   games 13 of 99 positions held 5 or more points ahead were drawn by the
   ply cap, repetition or the fifty-move rule. Rank shorter mates and larger
   conversions higher, then relabel.
6. Adaptive stopping per position, once more steps no longer hurt. The oracle
   bound on v16's step-5,000 report is 13.9% blunders at T8 against 12.6%
   choosing between T8 and T16.
7. Model size at equal wall time, after the cooldown shows whether capacity
   limits.
8. Value heads as 128-bin HL-Gauss classification (ChessBench, Stop
   Regressing), most useful if item 3 moves choice onto values.
9. More labelled positions: 53,000 steps is about 1.3 passes over the
   4.0 million labels. Longer or larger runs need more.
10. The v17 reply loss, if item 2 does not stop the decline past T8.

Closed: move-graph and full-line attention (v18, v18b), king_knight
neighbourhood (v15b).

Backlog item 3 result (v16 step 5,000, same 2,000 positions, paired against
policy choice): choosing by per-move value loses 3.3 Lichess accuracy at T8
(-4.4 to -2.3) and about 180 puzzle rating points. It wins only at T0 to T2
(51.9 against 45.1 at T0). The mix at beta 12.5 is level from T4 to T16,
-0.8 at T8 (-1.7 to +0.1), and +0.8 at T32 with 1.7 points fewer blunders.
It is about 110 puzzle rating points lower. Neither beats the policy at its
best T, so play stays on the policy. The full ChessBench paper explains why.
With equal data their action-value and state-value policies tie (Table A4).
Their behavioral cloning loses because it trains on the best move only,
which Wombat's all-move policy target already avoids. Revisit only if item 8
(classification value heads) makes the values much stronger.

Correction from the full paper: 9,707 of ChessBench's 10,000 test puzzles
are in Wombat's puzzle training split. The goal is therefore measured on a
fixed 10,000-puzzle sample of Wombat's own test split. DeepMind's released 9M
runs on the same puzzles and the same 2,000 strength positions.

## v19 pilots: depth training (registered 2026-09-24, before training)

v16's recipe on the full label pool for 5,000 steps with
`--depth-distribution lognormal --depth-mean 16 --depth-max 48`. As in Huginn,
each batch's total depth comes from a log-normal Poisson, with the last 8 steps
carrying gradients, graded at the end. This replaces fixed depth 8 plus a 0 to
8 prefix. v19a has no noise. v19b adds `--initial-noise 0.5` on the square
tokens of H0 (their RMS is 1.21 and one step changes them by 0.64 on v16's
step-5,000 snapshot). Baseline: v16's step-5,000 reports.

Accepted if best-T Lichess accuracy is not worse (paired interval above
-0.5) and one of two things improves. Either it improves itself (interval
above 0), or 3+ puzzle accuracy at the best T improves (paired). Also
reported: T32 against the best T, wall time per 1,000 steps, and
full-solution puzzle accuracy.

Backlog item 1 result (cooldown): v16 resumed from step 28,000 with a
6,000-step `1 - sqrt` decay to 0. Strength reports on the same 2,000
positions: step 28,000 before the cooldown reaches 84.6 at T8 (10.5%
blunders, puzzle rating 1,741). After it, 85.6 at T12 (9.6% blunders,
1,799, rising to 1,844 at T32). Paired, the cooldown adds +0.8 at T8 (0.1
to 1.4), +1.4 at T12 (0.8 to 2.0) and +1.2 at T16, with 1.5 points fewer
blunders at T12. T32 is level. Most of the gain since step 5,000 (81.6) is
the extra training. The cooled model passes gate L1 (84.9, the feedforward
baseline at step 53,000, which never cooled either). From now on every run
that is evaluated as a finished model ends with a cooldown.

## Probe of v16 step 20,000 and the cooled model's blunders (2026-09-24)

`probe-steps` on v16's step-20,000 copy, 2,000 positions per sample. The
one-move tactical facts are linearly readable from the square tokens before
any step. At the input they reach AUROC 0.96 to 0.99 for "a piece moving here
loses the exchange", "the opponent has a free capture", and mate in one for
either side. They barely change with steps, so the board features supply them.
What the steps build is the move. On puzzles, the best move's destination
rises from 42% at the input to 62% at T16, and the reply's destination from
13% to 38%. The T4 to T16 gain is spread over the board. Swapping only the
move's origin and destination tokens from T16 into T4 reproduces T16's move
on 43% of changed positions, the same as swapping every other square.
Freezing squares more than one step from the move after T4 loses the gain.

The cooled v16 makes 192 blunders at T12 on the strength positions (15 or
more win-chance points lost). By what the move allows one move later: 29 put
the moved piece where it loses the exchange (the state holds that fact, so the
readout ignores it). 35 leave other material en prise, 6 allow mate in one,
and 122 lose nothing at once (deeper tactics or misjudged positions).

## v20 pilot: move consequences (registered 2026-09-24, before any code)

Named gap: 70 of 192 blunders (36%) are visible one move ahead, but nothing
tells the model what each candidate move allows. The input features
describe the current position only. For every legal move, compute four facts
on the position after it. They are the static exchange of the moved piece, the
opponent's best static-exchange gain on any square, whether the opponent then
has mate in one, and whether the move gives check. They are added to that
move's policy logit and move value. The weights come from the move's origin
and destination tokens, not fixed weights. The thinking state then decides per
move whether a lost exchange matters or the sacrifice works (a fixed penalty
would punish sound sacrifices, which puzzles are full of). Zero initialised.
Computed batched on the GPU from
the existing ray code. v16's recipe for 5,000 steps against v16's
step-5,000 reports. Accepted if best-T Lichess accuracy improves (paired
interval above 0) and 3+ puzzle accuracy at the best T does not fall
(paired). Blunder rate and the four blunder kinds are reported.

Protocol change (2026-09-24): a pilot is judged on its finished model. Each
pilot's step-5,000 checkpoint gets a 1,000-step `1 - sqrt` cooldown to 0
(`--resume --stop-after 6000 --cooldown-steps 1000` in a copy of its run
directory) before the strength and puzzle reports. The cooldown moved v16 by
more than a point, as much as the differences pilots are decided on, and
recipes may gain from it unequally. The baseline is v15d's step-5,000
checkpoint with the same cooldown (v16's recipe, since v16 kept no full
checkpoint at step 5,000). v17, v19a and v19b are re-judged this way. Their
step-5,000 verdicts stand only as provisional.

## ChessBench 9M on our held-out data (2026-09-24)

DeepMind's released 9M was ported to PyTorch and checked against their JAX
code (identical moves on 408 positions, log probabilities within 8.7e-5). It
runs on a fixed 10,000-puzzle sample of Wombat's puzzle test split
(`data/eval/external-benchmark`, rating median 1,407 like theirs) and the
2,000 strength positions. The same code scores both (`evaluate-full-puzzles`,
`evaluate-external-strength`):

| Model | Full-solution puzzles | Lichess accuracy | Blunders |
| --- | --- | --- | --- |
| ChessBench 9M | 87.4% (86.7 to 88.0) | 93.3 | 2.25% |
| v16 cooled, T16 | 71.0% | 85.4 | 9.8% |
| v16 cooled, T12 | 70.3% | 85.6 | 9.6% |
| v16 step 5,000, T16 | 55.0% | 80.6 | 15.1% |

The released 9M reproduces 86.2% on ChessBench's own 10,000 puzzles, not
the paper's 88.9%. The gap to Wombat grows with puzzle rating: level up to
800, 3 to 9 points from 1,000 to 1,400, 26 to 38 points from 1,600 to
2,200. The strength measure favours the 9M, which was trained on the same
judge's move values.

Goal restated on this yardstick: beat 87.4% full-solution accuracy on the
10,000 exported test puzzles, and 93.3 Lichess accuracy on the 2,000
strength positions.

## Overnight procedure and the long run (2026-09-24)

Each pilot: 5,000 steps from v16's recipe plus one change, then a
1,000-step cooldown (protocol above). Then come the strength report and the
full-solution puzzle report on `data/eval/external-benchmark`, judged
against the cooled v15d baseline (81.6 best Lichess accuracy). New ideas go
into the backlog with a named gap and a gate before they run.

When every backlog item has a verdict and no justified idea remains: the
long run. The best accepted recipe for about 100,000 steps ending in a
cooldown of about 20,000, on every training position once
`train-labels-rest` is finished and merged with `train-labels`, plus the
puzzle mix. Evaluated with the strength report, the full-solution puzzle
report at T8, T12, T16 and T32, and the ChessBench 9M on the same data.

## v21 pilot: width (registered 2026-09-24, before training)

Backlog item 7. v16 is 2.45M parameters (width 256, MLP 1,024, 2 recurrent
blocks). ChessBench's smallest is 9M and bigger was better on every metric
there. v21a is v16's recipe at width 384, 8 heads, MLP 1,536 (about 2.2
times the parameters), 5,000 steps plus the 1,000-step cooldown. It runs
against the cooled v15d baseline at equal steps, because the long run is
budgeted in steps. Accepted if best-T Lichess accuracy improves (paired
interval above 0) and its time per 1,000 steps is at most twice v16's. A
120,000 step long run then stays within about a day. Full-solution puzzles
reported.

Cooled re-judgement (step 6,000, each paired with the cooled v15d baseline
on the same 2,000 positions and 10,000 external-benchmark puzzles):

| Run | Best Lichess accuracy | Full-solution puzzles, best | T4 accuracy | T32 accuracy |
| --- | --- | --- | --- | --- |
| baseline (v15d) | 81.6 (T8 to T16) | 58.1% (T16) | 79.2 | 80.0 |
| v17 reply pair | 81.8 (T8) | not run | 79.1 | 79.9 |
| v19a depth | 81.5 (T8) | 58.0% (T16) | 81.1 | 81.4 |
| v19b depth + noise | 81.9 (T8) | 58.0% (T12) | 81.3 | 81.6 |

v17: level at every depth (T8 +0.3, interval -0.4 to 0.9). Its steadier
long thinking before the cooldown was an unfinished-model effect. Not
accepted, and no longer the fallback for the decline past T8.
v19a and v19b: level at the best T on both yardsticks (v19b +0.4 at T8,
-0.4 to 1.1, puzzles -0.1 and -0.2 at T16). They are better at T4 (+1.9 and
+2.1, intervals above 0) and at T32 (+1.4 and +1.6), with no decline past the
peak and puzzles +1.4 at T8. Not accepted under the gate, which asks for a
higher best. Recorded as equal strength at half the steps. Noise adds
nothing measurable over v19a.

v21a result (cooled, step 6,000, paired with the cooled baseline): Lichess
accuracy +0.6 at T8 (-0.1 to 1.3), +0.5 at T12 and T16, +0.9 at T4 (0.1 to
1.7), +1.8 at T32 (0.9 to 2.6). It makes 0.5 to 1.8 points fewer blunders at
every depth, best 82.1 against 81.6. Full-solution puzzles +0.9 at T8 (0.3
to 1.5), +0.2 at T16, +1.5 at T32, best 58.4% against 58.1%. Training ran
at about 760 positions per second, faster than v16's pilots. Gate not met
at 6,000 steps (the best-T interval touches 0). Every depth leans the
same way, and ChessBench's width advantage grew with training length.

## v22: width at 20,000 steps (registered 2026-09-24, before training)

The long run's model size is the largest open choice, and 6,000 steps is
too short to separate widths. v22a is v21a (width 384) and v22b is v16's
recipe (width 256). Each runs 20,000 steps plus a 2,000-step cooldown on the
full label pool, same seed. The long run uses width 384 if v22a's best-T
Lichess accuracy beats v22b's with a paired interval above 0. It also uses
width 384 if they tie there and v22a's full-solution puzzle accuracy at its
best T is higher (paired interval above 0). Otherwise it uses width 256.

## Backlog additions from the reference sources (2026-09-24)

A survey of reference sources adds the items below, in order of expected gain
per cost. It covered lczero-training (and Monroe's fork), Maia-3/Chessformer,
searchless_chess, Huginn, TRM, Deep Thinking and the Leela look-ahead studies.

12. Surprise-weighted sampling (Leela diff focus, KataGo policy surprise
    weighting). Half the draws are uniform, half weighted by the cooled
    model's KL to the target policy plus value error, capped. Puzzles are
    weighted by failures. Gap: hard tactics are where the policy disagrees
    most.
13. Direction-aware attention (Shaw relative key and value vectors per king
    step offset, Monroe's fork): +1.8 policy accuracy over absolute encoding
    in CF-6M. Gap: signals travel one king step per block without knowing
    their direction.
14. Board-dependent attention bias (smolgen, Chessformer GAB), recomputed
    from the state every step.
15. Stopping rules first without training (top move unchanged for k steps,
    successive-policy KL), then a learned head (TRM halting, Leela value
    error), on existing reports. The oracle bounds them (13.9% to 12.6%
    blunders between T8 and T16).
16. Warm start between a puzzle's solver moves (Huginn continuous thinking).
    Carry the previous move's thinking into the next board. Test zero shot
    on the external puzzles first.
17. Optimistic policy head (lczero-training). It needs item 15's value-error
    head.
18. Whole-board input embedding (lczero-training), and difficulty-matched
    horizons for game positions using item 12's difficulty scores.

Ranked lower: mirror augmentation (under one pass of data), a separate
answer state (TRM, and v8 says Wombat ignores unforced state), soft policy
head (Wombat's target is already all-move), moves-left head (item 5 covers
conversion).

## Stopping rules and warm starts on the cooled v16, zero shot (2026-09-24)

Backlog items 15 and 16 without training, on `runs/v16-cool/checkpoint.pt`
(`evaluate-stopping`, `evaluate-warm-start`, reports in `runs/v16-cool/stopping`
and `runs/v16-cool/warm-start`). Every position is stepped to 32 once. A rule
picks the step whose argmax is played. Rules with settings are tuned on the
even rows (1,000 strength positions, 5,000 external puzzles) and reported on
the odd rows. Fixed T on all rows reproduces the stored reports exactly
(85.62 at T12, 71.03% at T16). Test half, paired with T12 on games and T16 on
puzzles:

| Rule | Lichess accuracy | Blunders | Steps | Puzzles | Steps |
| --- | --- | --- | --- | --- | --- |
| T8 | 85.92 (-0.31, -1.00 to 0.38) | 9.3% | 8 | 68.48% (-2.22, -3.02 to -1.42) | 8 |
| T12 | 86.23 | 8.9% | 12 | 69.96% (-0.74, -1.31 to -0.17) | 12 |
| T16 | 86.22 (-0.01, -0.52 to 0.49) | 9.1% | 16 | 70.70% | 16 |
| argmax held 6 steps, from step 2 | 85.83 (-0.40, -0.91 to 0.11) | 9.6% | 10.3 | 69.88% (-0.82, -1.50 to -0.14) | 8.1 |
| KL, tuned per set | 85.96 (-0.26, -0.52 to -0.01) | 9.1% | 10.7 | 70.38% (-0.32, -0.97 to 0.33) | 21.4 |
| oracle | 91.73 (+5.50, 4.56 to 6.45) | 4.5% | 3.0 | 83.88% (+13.18, 12.24 to 14.12) | 4.3 |

The KL rule tuned to 0.003 nats for 3 steps on games and 0.0003 for 3 steps
on puzzles. The agreement rule tuned to its largest window on both. No rule
beats the fixed depth it is paired with, and none matches it at fewer steps.
Each lands on the fixed-T curve at its mean step count. Stability of the
argmax or the policy does not tell a finished answer from a stuck one. The
oracle's gain is the spread of moves across 32 steps, not a reachable bound
for a rule that sees only the policy. A learned halting or value-error head
(the rest of item 15) is the remaining test.

Warm start, T8 and T16 on the 10,000 puzzles (8,524 with more than one solver
move), paired with the fresh start:

| Variant | T8 | T16 |
| --- | --- | --- |
| fresh | 69.01% | 71.03% |
| squares, alpha 0.25 | +0.01 (-0.17 to 0.19) | -0.09 (-0.24 to 0.06) |
| squares, alpha 0.5 | +0.18 (-0.08 to 0.44) | -0.35 (-0.62 to -0.08) |
| squares, alpha 1 | -1.10 (-1.50 to -0.70) | -1.19 (-1.55 to -0.83) |
| state token | 0 (identical) | 0 (identical) |

Copying the state token changes nothing by construction. Under local
attention square queries cannot read H's state token, and the square readout
never reads it. Carrying square thinking zero shot is level at small alpha
and harmful at alpha 1. Item 16 would need training with the carried state.

Verdicts: item 15's training-free rules are closed. Its learned head stays
in the backlog behind items 12 to 14, since the rules show the policy alone
does not signal a finished answer. Item 16 is closed. There is no zero-shot
signal at any alpha, so a trained carry-over is not justified.

## v23 pilot: surprise-weighted sampling (registered 2026-09-24, before training)

Backlog item 12. Surprise scores come from the cooled v16 at depth 8
(`runs/v16-cool/surprise`), capped at the 99th percentile. For game positions
the score is the KL from the training target to the policy plus squared value
error. For puzzles it is the solution cross-entropy. v23a is v16's recipe with
`--surprise-weights runs/v16-cool/surprise --surprise-share 0.5` for 5,000
steps plus the 1,000-step cooldown, against the cooled v15d baseline.
Accepted if best-T Lichess accuracy or best-T full-solution puzzle accuracy
improves (paired interval above 0) and neither falls (interval above -0.5).
Caveat: high surprise partly means "not yet trained on", since the scorer
saw 81% of the positions at most once.

## 04:28 memory stop (2026-09-24)

Every background job was stopped at 04:28 because the machine was
critically low on RAM. A virtual machine outside this project held about
11.7 GB of the 32 GB. The project's jobs were labelling with 12 engines,
surprise scoring, v20 training, an evaluation and the visualizer. Nothing
was corrupted, and each resumes from disk when restarted:

- v20: killed at step 1,000 (validation T8 55.0 against v16's 43.6 at the
  same step).
- Labelling `train-labels-rest20k`: 122 of about 1,219 chunks (610,000
  positions) at 117 positions per second.
- Surprise scoring: 3,604,480 of 3,999,041 game positions, no puzzles.
- Visualizer: stopped.

v22a finished before the
stop, at step 22,000 with its cooldown. Its best Lichess accuracy is 84.8 at
T12 (10.6% blunders, puzzle rating 1,741). Full-solution puzzles are 67.7,
68.3, 68.7 and 68.9% at T8, T12, T16 and T32. v22b, its width-256 pair, is a
new job and runs alone.

v22 result (step 22,000 with its cooldown, paired on the same data):
v22a (width 384) minus v22b (width 256), Lichess accuracy +0.9 at T4 (0.1
to 1.7), +0.3 at T8 (-0.3 to 0.9), +0.4 at T12 (-0.2 to 1.0), +0.6 at T16
(0.0 to 1.2), +1.8 at T32 (1.1 to 2.6). Fewer blunders at every depth,
best 84.8 against 84.4. Full-solution puzzles +2.3 at T8 (1.7 to 2.9), +1.9
at T12, +1.2 at T16 and T32, best 68.9% against 67.7%, +1.2 (0.6 to 1.8).
Games tie at the best T and puzzles improve. By the registered rule the
long run uses width 384 (8 heads, MLP 1,536, 5.3M parameters).

Baseline change: later pilots use width 384, judged against v21a. v21a is
width 384, 5,000 steps plus the 1,000-step cooldown in one run, with best
Lichess accuracy 82.1 and full-solution puzzles 58.4% at T32. v20b is v20 at
width 384 (v21a's flags plus `--move-consequences`), with v20's gate.
Best-T Lichess accuracy must go up (paired interval above 0). 3+ puzzle
accuracy, read here as full-solution puzzles at the best T, must not go down
(interval above -0.5).

## v24 pilot: direction-aware attention (registered 2026-09-24, before training)

Backlog item 13 (`--relative-attention`). It learns key and value vectors per recurrent block and head for
the 9 king-step offsets (plus one for the board memory's state token), zero
initialised. That is 15,360 parameters at width 384, with about 13% slower
graphed training. v24a is v21a's flags plus the option (6,000 steps with the
1,000-step cooldown), against v21a. Accepted if best-T Lichess accuracy
improves (paired interval above 0) or best-T full-solution puzzles improve
(interval above 0), and neither falls (interval above -0.5).

v20b strength (step 6,000 with its cooldown, paired with v21a on the same
2,000 positions): Lichess accuracy +3.2 at T4 (2.3 to 4.1), +1.2 at T8 (0.3
to 2.0), +1.1 at T12, +1.2 at T16 (0.3 to 2.0), +1.5 at T32 (0.6 to 2.3).
It makes 1.6 to 3.9 points fewer blunders, best 83.3 (flat from T4) against
82.1, and 79.3 already at T1. The games half of v20's gate passes. The puzzle
half waits for the full-solution report, which runs after v24a because
both do not fit on the GPU together. v24a's first try ran out of memory
beside a report and was restarted alone as `runs/v24a-r`.

## v25 pilot: state-generated attention bias (registered 2026-09-24, before training)

Backlog item 14 (`--state-attention-bias`).
A smolgen-style per-head 64x64 bias is generated from H's square tokens in
every block at every step, zero initialised. It has 188,096 parameters at
width 384, of which about 72,500 can learn under the king mask. v25a is
v21a's flags plus the option, against v21a, with v24's gate. It runs after
v24a and v20b's puzzle report.

v20b puzzles (10,000 external-benchmark puzzles, paired with v21a): +1.9
at T8 (1.2 to 2.6), +1.3 at T12, +0.9 at T16 (0.1 to 1.6), +0.7 at T32
(-0.1 to 1.4). Best 59.1% against 58.4%, reached from T8 on. **v20b is
accepted**: best-T Lichess accuracy +1.2 (0.3 to 2.0) with fewer blunders at
every depth, and puzzles do not fall. Move consequences join the recipe.

v24a result (step 6,000 with its cooldown, paired with v21a): Lichess
accuracy +1.8 at T4 (1.0 to 2.5), +0.6 at T8 (0.0 to 1.3), +0.7 at T16
(0.0 to 1.4), +1.0 at T32 (0.3 to 1.6), best 82.8 against 82.1. Puzzles +2.4
at T8 (1.8 to 3.0), +1.8 at T12, +1.4 at T16, +1.3 at T32 (0.7 to 1.9), best
59.6% against 58.4%. **v24a is accepted** on puzzles (best-T interval above
0) with games not down. Direction-aware attention joins the recipe. v26
checks it together with move consequences.

## v26 pilot: accepted changes together (registered 2026-09-24, before training)

v26a is v21a's flags plus `--move-consequences --relative-attention`
(6,000 steps with the 1,000-step cooldown). It is paired against v20b and
against v24a. It enters the long-run recipe if its best-T Lichess accuracy and
best-T full-solution puzzles are each not worse than either single change
(intervals above -0.5). Otherwise the long run takes the single change with
the better best-T puzzles. v25 joins the same way if it is accepted.

v25a result (step 6,000 with its cooldown, paired with v21a): Lichess
accuracy +2.1 at T4 (1.4 to 2.9), +0.5 at T8 (-0.2 to 1.2), +0.5 at T12,
+0.6 at T16, +0.8 at T32 (0.2 to 1.5), best 82.7 against 82.1. Puzzles +2.8
at T8 (2.1 to 3.4), +2.4 at T12, +1.9 at T16, +1.7 at T32, best 60.2%
against 58.4%, +1.8 (1.1 to 2.4). **v25a is accepted** on puzzles with games
not down. v26a therefore combines all three accepted changes: v21a's flags
plus `--move-consequences --relative-attention --state-attention-bias`,
under v26's rule against each single change.

## v27 pilot: HL-Gauss value heads (registered 2026-09-24, before training)

Backlog item 8 (`--value-bins`). Position
value and per-move values become 64-bin distributions trained with HL-Gauss
(sigma 0.75 bins), read out as expected values on the old -1..1 scale. That
adds 73,535 parameters. Gap: the per-move values are weak (choosing by them
cost 3.3 accuracy). v27a is the recipe v26 selects plus `--value-bins 64`,
against that recipe's pilot. Accepted if best-T Lichess accuracy or best-T
full-solution puzzles improve (paired interval above 0) and neither falls
(interval above -0.5). Also reported: the strength report with
`--select value` and `--select mix`, to see whether the values became usable.

v26a result (step 6,000 with its cooldown): best Lichess accuracy 84.2
(10.8% blunders, flat from T4, 81.4 at T1) and full-solution puzzles 61.0%
(from T8). Paired at T8 and at each run's best puzzle depth:

| v26a minus | Lichess accuracy, T8 | Puzzles, best T |
| --- | --- | --- |
| v21a (baseline) | +2.0 (1.2 to 2.8) | +2.6 (1.9 to 3.3) |
| v20b (move consequences) | +0.8 (0.3 to 1.4) | +1.9 (1.3 to 2.5) |
| v24a (relative attention) | +1.4 (0.5 to 2.2) | +1.4 (0.6 to 2.1) |
| v25a (state attention bias) | +1.5 (0.7 to 2.3) | +0.8 (0.1 to 1.6) |

The three changes add up and v26a beats each alone on both yardsticks, so
the recipe is now v26a's: width 384, move consequences, relative attention
and the state attention bias. v27a runs on it.

## v28 pilot: more puzzle batches (registered 2026-09-24, before training)

Gap: on the goal's yardsticks Wombat trails ChessBench's 9M by about 26
points on full-solution puzzles (61.0% against 87.4%). On Lichess accuracy
the gap is 9 (84.2 against 93.3). Puzzle batches are a quarter of training.
v28a is v26a's flags with `--puzzle-share 0.4`, against v26a. Accepted if
best-T full-solution puzzles improve (paired interval above 0) and best-T
Lichess accuracy does not fall (interval above -0.5).

v27a result (paired with v26a): Lichess accuracy +0.3 at T8 to T32 (about
-0.2 to 0.8), best 84.5 against 84.2, and 64.0 against 61.9 at T0.
Full-solution puzzles +1.4 at T4 (0.9 to 1.9), +1.0 at T8, +1.2 at T16
(0.7 to 1.7), +1.1 at T32, best 62.1% against 61.0%. **v27a is accepted**
on puzzles with games not down, and value bins join the recipe. Choosing moves
by the policy and value mix still loses to the policy (-0.6 at T8, -0.7 at
T16, -1.4 to -0.1), so play stays on the policy. v28a now runs on v27a's
recipe (plus `--puzzle-share 0.4`) against v27a.

v28a result (paired with v27a): full-solution puzzles +1.5 at T8 and T16
(1.0 to 2.0), +1.6 at T32, best 63.6% against 62.1%. Lichess accuracy -1.0
at T8 (-1.7 to -0.4), -1.2 at T16 and T32, with about 1.1 points more
blunders. **Rejected**: games fall past the -0.5 limit. More puzzle batches
trade game strength for puzzles one for one. The share stays 0.25.

## v29: the recipe at 20,000 steps (registered 2026-09-24, before training)

Early gains have vanished with longer training before (v17). v29a is
v27a's recipe for 20,000 steps plus a 2,000-step cooldown. It is paired
against v22a (width 384 alone, same steps and cooldown: 84.8 best Lichess
accuracy, 68.9% puzzles). The long run keeps the recipe if v29a beats v22a on
both best-T Lichess accuracy and best-T puzzles (paired intervals above 0).
Otherwise the added parts are re-examined one by one at this length.

Backlog status at 08:07 (2026-09-24):

- Accepted into the recipe: width 384 (v22), move consequences (v20b),
  relative attention (v24a), state attention bias (v25a), value bins
  (v27a), together as v26a/v27a. Cooldown at the end of every run (item 1).
- Rejected or closed: depth training (v19a/b: equal peak), move choice by
  value or mix (item 3, again on v27a), stopping rules and warm start
  (items 15 training-free, 16), more puzzle batches (v28a), move-graph
  attention (v18/v18b), reply head (v17).
- Interrupted by the 04:28 memory stop: labelling the remaining 6.1M positions (item 9, needed for the long
  run) and surprise scoring (item 12, v23).
- Deferred without a pilot, because no named gap on the goal's yardsticks
  supports them now:
  - won-position labels (item 5). It concerns converting won games, which
    neither yardstick measures.
  - the learned halting head (item 15). The recipe's curve is flat from T4, so
    stopping earlier saves compute but cannot add strength.
  - the optimistic policy head (item 17), with evidence only inside tree
    search.
  - whole-board input embedding and difficulty-matched horizons (item 18).
    The evidence is weak, and v18b showed extra reach adds little on top of
    the board features.
- Running: v29a, the recipe at 20,000 steps, the last check before the long
  run.

v29a result (step 22,000 with its 2,000-step cooldown, paired with v22a):
Lichess accuracy +2.1 at T4 (1.3 to 3.0), +1.3 at T8 (0.5 to 2.1), +1.2 at
T12, +1.3 at T16, +1.3 at T32, with 1.4 to 2.0 points fewer blunders. Best
86.0 at T8 with 9.2% blunders (puzzle rating 1,807 to 1,829). Full-solution
puzzles +3.0 at T8 (2.3 to 3.7), +2.7 at T16, +2.4 at T32, best 71.3%
against 68.9%, +2.4 (1.7 to 3.1). The recipe keeps its lead at 20,000
steps, so the long run uses it. At 22,000 steps it already matches the
cooled v16 at 34,000 (85.6, 71.0%, paired +0.4 at T12, -0.4 to 1.1).

Against the goal: 86.0 against 93.3 Lichess accuracy and 71.3% against 87.4%
full-solution puzzles. The long run is about 100,000 steps plus a
20,000-step cooldown on every labelled position. It is ready to start once
labelling of the remaining 6.1M positions, stopped at 04:28, is restarted and
merged.

09:50: labelling (`train-labels-rest20k`, now 14 engines) and surprise
scoring resumed from disk after the 04:28 stop. The long run starts when
labelling is merged. Until then the GPU runs surprise scoring and then v23.

v23 rebased: surprise scoring finished (games and 13.7M puzzle positions,
`runs/v16-cool/surprise`). v23a is v27a's recipe plus `--surprise-weights
runs/v16-cool/surprise --surprise-share 0.5`, against v27a, with v23's gate.

## Long run L1 (registered 2026-09-24, before training)

v27a's recipe for 120,000 steps, the last 20,000 a `1 - sqrt` cooldown, in
`runs/long1`. The recipe is width 384, move consequences, relative attention,
state attention bias and 64 value bins. Depth is fixed at 8 graded at the
end, with a prefix of 0 to 8 and a puzzle share of 0.25. It starts on the
4.0M labelled positions. It is resumed on the merged `train-labels-all`
(about 10.1M) as soon as labelling finishes. Surprise sampling is switched on
at that resume only if v23a is accepted. Evaluated at the end with the
strength report, the full-solution puzzle report at T4 to T32, the ChessBench
9M on the same data, and a Stockfish ladder at 50,000 nodes. Snapshots every
5,000 steps.

v23a result (paired with v27a): Lichess accuracy -0.6 at T8 (-1.2 to 0.0),
-0.7 at T16 and T32 (about -1.3 to -0.1), with 0.9 points more blunders.
Full-solution puzzles level (62.0% against 62.1%, -0.1 to +0.6 intervals
around 0). **Rejected**: games fall past the -0.5 limit and puzzles do not
move. The weights came from the cooled v16, not the current recipe. A
rescored version is not rerun because nothing here suggests a gain.

L1 started at 14:00 on the 4.0M labelled positions, without surprise
sampling.

## Rented machine and the current state (2026-09-24, 15:10)

A rented server joins the work: RTX 5090 32 GB, AMD EPYC 9554 with 32
cores, Ireland, $0.689/h. Setup: the code from `git archive HEAD`,
Stockfish 19 (Linux universal build), torch 2.14 cu130 in a virtual
environment, and the data (train split, train-labels, puzzles, validation,
test, external benchmark). An earlier instance in Taiwan was dropped
because its route to Europe ran at about 80 KB/s.

Running at 15:10:

- L1 at home (`runs/long1`, v27a's recipe, 120,000 steps, 20,000-step
  cooldown, started 14:00 on the 4.0M labels, about 400 positions/s).
- Labelling of the remaining positions into `train-labels-rest20k` from
  both ends: at home forward (14 engines), on the server backward
  (`--reverse`, 32 engines). A chunk that exists is
  skipped, so the two meet without repeating work. When all 1,220 exist,
  `label-moves` runs once more locally to finalize, then
  `merge-move-labels --inputs train-labels train-labels-rest20k --output
  train-labels-all --allow-mixed-nodes`, and resume L1 with `--labels
  data/processed/games-v1/train-labels-all --resume`.
- Batch-size throughput benchmark on the 5090 (batches 128, 512 and
  1,024, 400 steps each).

Next, registered here before running: v30, the recipe at the largest batch
that runs well on the 5090. The learning rate is scaled by the square root of
the batch ratio. It sees the same number of training positions as a
6,000-step batch-128 pilot and is paired against v27a. The long run moves to
the 5090 with that batch if v30's best-T Lichess accuracy and puzzles are not
worse (intervals above -0.5). The home L1 then continues only as a backup.

## Backlog item 19: more game positions (2026-09-24, before any code)

Gap: the 5090 trains the recipe at about 1,560 positions per second at
batch 128 (3.4 times home), and more at larger batches. A day on it
passes over our 10.3M labelled positions many times, while the ChessBench
9M learned from about 530M. Server labelling at 20,000 nodes runs at about
1,080 positions per second (13 chunks a minute with 32 engines, same label
statistics as home). 40M more positions take about ten hours.

Plan: build `games-v2` from the source records after the ones games-v1
read, with no position from games-v1 or the evaluation sets. Label it on
the server, and train on both corpora. Gate: a long run on games-v1 plus
games-v2 is judged against the same-length run on games-v1 alone at equal
steps (paired, best-T Lichess accuracy and full-solution puzzles). More
data is accepted if either improves with its interval above 0 and neither
falls below -0.5.

## v30 result: batch 512 on the 5090 (2026-09-24)

Throughput of the recipe on the 5090 (400 steps each, while 32 labelling
engines ran): 1,559 positions per second at batch 128, 2,060 at batch 512.
Batch 1,024 runs out of its 32 GB. Training already runs in bf16 with CUDA
graphs, so the card is near full at batch 128.

v30a (on the server): batch 512, learning rate 1e-3,
warmup 250, EMA 0.996, 1,250 steps plus a 250-step cooldown, the same
768,000 positions as v27a. At its end validation T8 is 61.9% against
v27a's 64.2%, and full-solution puzzles at T16 are 57.6% against 62.1%.
Rejected: the larger batch learns less per position by far more than its
1.3 times throughput buys. The strength report was not run, since the puzzle
gate already fails. Long runs on the 5090 use batch 128 (3.4 times home).

## Evaluation overlap with training (2026-09-24)

Exact board, side to move, castling and legal en passant, streamed over
games-v1 train and puzzles-v1 train. 191 of the 10,000 external benchmark
puzzles (416 of 23,063 scored positions) have a scored position in
training, from other puzzles or games with the same position. No puzzle id
is shared. The strength report has 0 of 2,000 game positions in games
training (1 in puzzle training) and 23 of its 1,000 puzzles. On the clean
9,809 puzzles the 9M scores 87.49% and the cooled v16 70.96% at T16, so the
gap moves by about 0.2 points. Lists are in `data/eval/external-benchmark`
(`contaminated-ids.txt` and two strength lists). games-v2 excludes all
puzzle validation and test positions.

## Long run L2 on the 5090 (registered 2026-09-24, before training)

L1's recipe (v27a, batch 128) on all of games-v1's labels
(`train-labels-all`, 10.3M positions) and the puzzles, trained on the 5090.
It runs 400,000 steps with a 60,000-step cooldown (51M positions, about five
passes over the games), snapshots every 50,000. It answers item 19's baseline
question too. L3 is the same run with games-v2 added (`--extra-train`),
judged against L2 at equal steps by item 19's gate. L2 and L3 are each
evaluated at the end on the strength report (T4 to T32), full-solution
puzzles (T4 to T32) and the 9M comparison. L1 at home stops when L2 is
running, freeing the home GPU for pilots.

L2 started at 15:40 on the 4.0M labels (resumed onto `train-labels-all`
once merged), 1,420 positions per second. At step 5,000 its validation
matches L1, v29a and v27a at the same step within noise (T8 63.6% against
63.2 to 63.3%). L1 stopped at step 23,000, since L2 is the same recipe at
3.4 times the speed. The home GPU tracks L2. Each 50,000-step snapshot is
copied home and gets full-solution puzzles at T8 to T16 and the strength
report, paired with v29a. L1's step-23,000 checkpoint (not cooled) is
evaluated the same way first. That measures what the cooldown adds at this
recipe (v29a is the same recipe cooled at 24,000).

L1's step-23,000 checkpoint (not cooled, its cosine spans 120,000 steps)
paired with v29a (cooled at 24,000): Lichess accuracy -0.3 at T8 (-0.9 to
0.3), -0.5 at T12 (-1.1 to 0.1), -0.4 at T16, best 85.6 against 86.0.
Full-solution puzzles +0.7 at T8 (0.1 to 1.3), +0.2 at T16 (-0.3 to 0.8),
best 71.6%. An uncooled snapshot of the recipe reads about 0.4 lower on
games and level on puzzles. L2's snapshots are read with that in mind.

## v31 pilot: width 512 (registered 2026-09-24, before training)

Backlog item 7 in the long-run regime. Gap: L2 and L3 give the recipe 51M
positions and more, and the 9M has 9M parameters to our 5.6M. Width 256 to
384 gave +1.2 puzzles at 20,000 steps (v22). v31a is v29a's command with
`--d-model 512 --heads 8 --mlp-width 2048` (22,000 steps with a 2,000-step
cooldown), at home. It is paired with v29a on the strength report (T4 to
T16) and full-solution puzzles (T8 to T16). Accepted if best-T Lichess
accuracy or puzzles improve with the interval above 0 and neither falls below
-0.5. Then the final long run after L3 uses width 512. L2's snapshots are
evaluated at home between v31's end and the next job.

State at 16:25: labelling of the remaining positions finished. The home
run stopped at 16:10 so the PC stays free, and the server finished the rest.
`train-labels-all` on the server holds 10,092,005 positions (4.0M at
50,000 nodes, 6.1M at 20,000). L2 resumes on it from its step-20,000
checkpoint. Width 512 needs 6.0 GiB of allocated VRAM at batch 128 (width
384: 4.6 GiB), more than the home card can spare. v31a runs on the
5090 beside L2 with a 10 GiB cap. The 5090 is shared with another job
from 16:15, so L2's steps per hour are lower from then on.

games-v2 built on the server (16:30): 39,201,654 train positions from
source records 26.9M to 165.5M, none in games-v1 (14,756 repeats rejected)
or the evaluation and puzzle validation/test sets. Labelling started at
16:33 (`data/processed/games-v2/train-labels`, 20,000 nodes, 32 engines,
about ten hours). L2 resumed at step 20,000 on all 10.1M games-v1 labels
at 16:31. v31a is at step 2,000 at 730 positions per second beside it.

v31a stopped at step 4,000 (16:45). It diverged between steps 2,000 and
3,000. Training loss went from 4.42 to 5.00 and validation T8 from 55.9% to
50.2%. T1 equalled T8 afterwards, so the steps stopped contributing. v29a at
width 384 reached 62.1% at step 4,000 on the same schedule. Width 512 is
unstable at learning rate 5e-4. Not a verdict on width. A retry at 3.75e-4
(the rate scaled by 384/512) waits until the 5090 has room, since running
beside L2 halves L2's speed. L2's validation at step 20,000 (69.3% T8,
puzzles r2 80.3%) matches L1 and v29a at that step. Its gain from the new
labels shows only from here on.

### Why v31a diverged, and v31b/v31c (registered 2026-09-24, before results)

Attention logit growth. Largest |q.k| logit on 256 validation positions
over 8 steps: v31a at step 4,000 reaches 162 (EMA weights) and 241 (raw),
with a mean per-call maximum of 117 and 157. The healthy width-384 models
stay at 14 to 21 (v29a step 5,000: 19, v30a: 21, L2 step 25,000: 14).
The additive biases (square and state bias) stay at 8 to 15 in all of them,
and the hidden RMS at 0.9 to 1.2. The growth is in the query and key
projections (|Wq|, |Wk| 21 to 23 at width 512 against 16 to 20). An eager
rerun logged the logit climbing from 2.8 at step 50 to 11.3 at step 550,
still in warmup. Saturated softmax attention stops the gradient through
attention, and the steps stop contributing (T1 equals T8 after step
3,000).

Fix: `--qk-norm` RMS-normalizes each
head's queries and keys with a learned gain before their product
(ViT-22B, OLMo 2), which bounds the logits. Two pilots on v29a's schedule
(22,000 steps plus the 2,000-step cooldown, 4.0M labels), on the 5090
beside L2, both paired with v29a on the strength report and full-solution
puzzles:

- v31b: width 512 with `--qk-norm`. Width is accepted for the final run if
  v31b beats v31c on best-T Lichess accuracy or puzzles with the interval
  above 0 and neither falls below -0.5.
- v31c: width 384 with `--qk-norm`, the recipe plus QK-norm alone. QK-norm
  joins the recipe if v31c is not worse than v29a on either yardstick
  (intervals above -0.5), since it removes a failure mode at no cost.

L2 step 25,000 (not cooled, 5,000 steps on the full labels), paired with
v29a (cooled at 24,000): full-solution puzzles 71.6% at T8 (+0.9, 0.3 to
1.4), 71.9% at T12 (+0.6, 0.1 to 1.1), 72.0% at T16 (+0.7, 0.2 to 1.2), a
new best. Lichess accuracy 85.4 at T4 (+0.5, -0.2 to 1.1), 85.9 at T8
(-0.1, -0.7 to 0.6), 85.6 at T12, 85.8 at T16, level. The 9M: 87.4% and
93.3. Reports in `runs/long2/*-step-025000`. With v31b, v31c and another
job sharing the 5090, L2 runs at about 430 positions per second.

Sharing the 5090 costs throughput. With L2, v31b and v31c together (and
another job at times) L2 ran at 432 positions per second and v31c at
189, about 750 in total against 1,400 for L2 alone. v31b and v31c were
stopped at 17:20 (v31c at step 1,000, level with v29a, v31b before its
first evaluation). They run one at a time after L2, before L3, which waits
for the games-v2 labels anyway. One training job on the 5090 at a time.

L2 step 50,000 (snapshot, EMA weights, not cooled), paired with v29a:
full-solution puzzles 75.1% at T8 (+4.4, 3.8 to 5.0), 75.7% at T12
(+4.4, 3.8 to 5.1), 75.8% at T16 (+4.5, 3.8 to 5.1). Lichess accuracy
85.8 at T4 (+0.8, 0.1 to 1.6), 86.6 at T8 (+0.7, -0.1 to 1.4), 86.6 at
T12, 86.5 at T16, blunders 8.3% at T8 against 9.2%. Both are new bests,
30,000 steps after the switch to all 10.1M labels. Snapshots are evaluated
automatically on the server.

## Backlog item 20: calculation depth (2026-09-24, before any run)

Where the 9M leads, from the stored full-solution reports (L2 step 50,000
at T16 against the 9M on the same 10,000 puzzles):

| Puzzles | 9M | L2 50k | Gap |
| --- | --- | --- | --- |
| 1 solver move (1,476) | 99.7% | 99.8% | +0.1 |
| 2 solver moves (5,140) | 92.5% | 81.1% | -11.3 |
| 3 solver moves (2,581) | 79.2% | 62.1% | -17.1 |
| 4+ solver moves (803) | 58.4% | 41.7% | -16.7 |
| rating below 1,000 (2,319) | 99.3% | 98.6% | -0.6 |
| 1,000 to 1,500 (3,239) | 96.1% | 94.0% | -2.1 |
| 1,500 to 2,000 (2,541) | 87.9% | 69.5% | -18.4 |
| 2,000 to 2,500 (1,510) | 64.5% | 30.3% | -34.2 |
| 2,500 and up (391) | 29.2% | 6.6% | -22.5 |

The gap is entirely in deep, multi-move tactics. Thinking helps most there
(v29a: T4 to T16 adds 4.0 points on 4+ move puzzles and 0.2 on one-move
ones). L2's step-to-step validation shows the thinking gain growing
with training (T1 to T8: +7.2 at step 5,000, +16.1 at 60,000, with T1 flat
at 56%). But every curve is flat from T8 or T12 to T32. The recurrence has
learned the depth it was trained at and does not extrapolate.

Gap: calculation depth on hard puzzles. Pilots, each on v29a's schedule
(22,000 steps plus the 2,000-step cooldown, 4.0M labels), paired with v29a
on full-solution puzzles (by band and overall) and the strength report:

- v32a: training depth 16 instead of 8 (`--fixed-depth 16`, random prefix
  0 to 16), so the loop is trained to use more steps. Evaluated to T32.
- v32b: longer puzzle thinking (`--puzzle-steps-per-move 8
  --puzzle-max-tail 16`), more steps per solver move on multi-move lines.

Accepted if best-T full-solution puzzles improve (interval above 0), with
the 1,500+ bands carrying the gain, and best-T Lichess accuracy does not
fall below -0.5. Order on the 5090 after L2: v31c, v31b, v32a, v32b, then
L3 on the recipe they leave (L3 starts sooner with partial games-v2 labels
if the pilots run long).

Revision (18:25): v32a and v32b are withdrawn before running. They guess at
a fix. The mechanism comes first: what the thinking steps do on hard puzzles
L2 misses and the 9M solves. That means the per-step trajectory of the solver
move, probes for the continuation, attention toward the line's later squares
and state convergence, on L2's step-50,000 snapshot, CPU only. The change to
test follows from that analysis. L2 stops at step 100,000. A cooled branch
(`runs/long2-cool`, resumed from step 100,000, cooldown to 115,000) gives
the finished model and L3's baseline at equal steps. It frees the 5090
about six hours sooner than the full 400,000 steps.

L2 step 100,000 (snapshot, not cooled), paired with v29a: full-solution
puzzles 78.0% at T8 (+7.3, 6.6 to 8.0), 78.7% at T12, 78.8% at T16 (+7.5,
6.8 to 8.2). Lichess accuracy 86.3 at T4 (+1.3), 87.9 at T8 (+1.9, 1.1 to
2.7), 87.7 at T12, 87.5 at T16, blunders 7.3% at T8. L2 stopped there at
18:53, and `runs/long2-cool` cools it to step 115,000.

L2 cooled (`runs/long2-cool`, step 115,000: step 100,000 plus a 15,000-step
cooldown), paired with v29a and with the 9M on the same puzzles:

| T | Puzzles | vs v29a | vs 9M (87.36%) | Lichess accuracy | Blunders | vs v29a |
| --- | --- | --- | --- | --- | --- | --- |
| 4 | 74.3% | +4.7 (4.0 to 5.4) | -13.0 | 86.2 | 8.9% | +1.3 (0.4 to 2.1) |
| 8 | 79.1% | +8.4 (7.7 to 9.1) | -8.3 | 88.5 | 6.3% | +2.6 (1.7 to 3.4) |
| 12 | 80.1% | +8.8 (8.0 to 9.5) | -7.3 | 88.8 | 6.5% | +2.8 (1.9 to 3.6) |
| 16 | 80.6% | +9.3 (8.5 to 10.0) | -6.8 | 88.3 | 7.0% | +2.4 (1.5 to 3.2) |
| 32 | 80.8% | +9.5 (8.7 to 10.2) | -6.6 (-7.4 to -5.8) | 86.9 | 8.5% | |

Best: 80.8% full-solution puzzles (the 9M 87.4%) and 88.8 Lichess
accuracy (the 9M 93.3). After the cooldown the puzzle curve still rises
from T8 to T32 (+1.7, against +0.6 for v29a), while games peak at T12 and
fall by T32. L3 is judged against this model.

## Mechanism of the thinking steps, and backlog item 21 (2026-09-24)

Analysis of the cooled L2 on 1,300 hard puzzles (rating 1,500+, 2+ solver
moves). Group A is where L2 solves, B where L2 fails and the 9M solves, and C
where both fail. The full report is in
`runs/analysis/reasoning-l2-cool/REPORT.md`. The loop settles to a fixed point
rather than calculating. The per-step state change halves every step or two,
identically on easy and hard puzzles. The steps sharpen the first move and
make the reply to the favoured move decodable (probe 0.14 to 0.42). The
solver's second move (0.12 to 0.23) and the line's material outcome (r 0.3 to
0.4, flat) never become decodable. In B the solution is in the top 3 in about
85% of failing positions. Its probability stays flat near 0.2 from T2 to T32
(A climbs 0.39 to 0.74). B's solutions are quieter (46% against 33%) and more
often sacrifices. L2's wrong moves are greedy captures and checks that lose a
median 54 points and are refuted within 2 plies. B is no deeper than A for
Stockfish (median depth 1). The knowledge is there. After the first move and
reply, L2 plays the rest of B's lines correctly 90% of the time. Its own value
head applied after each legal move picks B's solution in 38 to 46% of its
failures. The fault is the move readout, which scores moves on a hidden state
that holds only the present position.

Item 21, registered before any code: child evaluation at move choice. For
the policy's top k moves (k = 2 and 3), play the move, run L2 on the
resulting position at the same depth, and re-rank by the negated child
value. Also test the blend of log-probability plus beta times child value.
This is what the 9M does with one pass per legal move. Evaluation only, on the
cooled L2, paired with its policy choice on full-solution puzzles and the
strength report at T8 and T16. Accepted if best-T puzzles or Lichess
accuracy improve (interval above 0) and neither falls below -0.5. If it
passes, a training version teaches the policy the child-value choice so
play needs one pass again.

v31c result (width 384 with QK-norm, 22,000 steps plus cooldown, paired
with v29a): full-solution puzzles 71.3% at T16 (-0.01, -0.55 to 0.53),
-0.8 at T4, level from T8. Lichess accuracy 85.7 at T12 (-0.28, -0.95 to
0.39), -0.3 to -0.5 from T8 to T32. Level, but both intervals dip just
below -0.5, so by its gate QK-norm does not join the width-384 recipe. It
stays tied to width 512 (v31b, running).

Item 21, first results on the cooled L2 (full benchmark, paired with its
own policy choice): child choice among the top 2 moves 74.4% at T8 (-4.7,
-5.5 to -3.9) and 74.8% at T16 (-5.8, -6.6 to -5.0). Among the top 3,
73.0% (-6.1) and 72.9% (-7.7). Picking by the child value alone breaks
more puzzles than it rescues, as the analysis predicted for group A. The
blend (child-mix, beta 4) is running.

## v33 and v34: learning to calculate, after the reasoner project (registered 2026-09-24, before any run)

Why. The mechanism analysis shows the loop settles to a fixed point that
re-renders the present position. Three properties of our training produce
that. Only the final state is graded. Every batch starts from a random
no-gradient prefix, so extra steps must change nothing (a fixed point is
the optimum). One 65-token state must hold both the present and any
imagined future, so the lookahead heads collapse to copying the present.
The reasoner project is built the other way. It has a
persistent memory M read by address and written by the step, a small
workspace H that carries the computation, and one shared operator. A
canonical-state loss pulls the state after hop k toward the encoding of the
problem that remains. That let it extend 2-hop training to 6 hops. Its forked
control also tested surplus-step invariance introduced before the exact-depth
algorithm has formed. That cut unseen-depth accuracy from 84% to 42%. Surplus
steps belong in a late stabilisation stage. Wombat's `--random-prefix 8`
applies them from the first step.

v33a (the training order alone): v29a's command with the random prefix
only from step 18,000 of 24,000 (`--random-prefix-from 18000`). The
first 18,000 steps grade exactly 8 steps (puzzles at their own horizon).
Paired with v29a on full-solution puzzles (overall and the 1,500+ bands)
and the strength report, T4 to T32. Accepted if best-T puzzles or Lichess
accuracy improve with the interval above 0 and neither falls below -0.5.
Mechanism check alongside: the per-step state change and the T8 to T32
gain, against v29a.

v34 (the reasoner's structure on chess), design to build:
- M: the encoded present board as memory, writable by the step, so an
  imagined move is a write that turns the present board's memory into its
  child's.
- H: a small workspace of a few tokens that reads the board by global
  attention and carries the plan (candidate, place in the line, running
  evaluation). The answer is read from H after the line is visited.
- Canonical-state supervision per ply: the state after the step that plays
  a move is pulled toward the encoder's representation of the resulting
  position (stop-gradient). This runs along Stockfish's 4-ply main line where
  it is labelled, and along the model's own chosen moves otherwise (the
  relwalk "self" target). No new labels.
- Exact depth first, surplus steps last.
- One line per candidate. Conditional branching is left out, since the
  reasoner never learned it (condwalk).
Gate for the v34 pilot (v29a's schedule, paired with v29a and v33a):
best-T puzzles improve with the interval above 0 and the gain sits in the
1,500+ bands. Lichess accuracy is not below -0.5. Probes read the
solver's second move from the state clearly above L2's 0.12 to 0.23.

Item 21, blend: child-mix (top 3, beta 4) on the cooled L2 scores 81.1% at
T8 (+2.0, 1.5 to 2.6) and 82.0% at T16 (+1.4, 0.9 to 2.0) against its own
policy choice. It passes the puzzle half of its gate. But it is an
inference-time patch on an existing model and costs extra passes per
move. It is recorded as evidence that per-candidate evaluation is the
missing ingredient. The work goes into training that ingredient
(v33, v34) rather than into this patch.

v31b result (width 512 with QK-norm, stable, 22,000 steps plus cooldown):
best-T puzzles 70.5% against v31c's 71.3% (-0.8 at T16, -1.4 to -0.3)
and v29a's 71.3%. Lichess accuracy 85.8 at T8 (+0.2 against v31c, -0.4 to
0.9, and -0.2 against v29a). Width 512 does not beat width 384 at this
training length, so width stays 384. The widths are compared again only
at long-run scale if data grows past what width 384 can use. v33a (exact
depth first) is training on the 5090 from 18:34.

v34 as built (8.05M parameters):
- M is a copy of the present board memory, reset at the start of each line.
- 12 workspace tokens persist across lines and attend globally to B, M and
  themselves every step.
- At each ply the model plays its own line-policy choice (exact tensor chess
  rules, checked against python-chess) and marks the move on M. It is pulled
  toward the stop-gradient encoding of the resulting position.
- The reference line is a warm start that fades to 0 by 30% of training and
  never forces a line's first move.
- Each batch draws 1 to 3 lines of 1 to 3 plies at 2 steps per ply (2 to 18
  steps, exact), with surplus steps only from step 18,000.
- The root answer reads H through one readout block into the existing heads.
Evaluation takes any T (lines of `--line-plies` plies), so budgets far past
training are swept (T up to 128). v34a is queued after v33a on the 5090.

Games ladder (requested 2026-09-24): the cooled L2 plays v16's ladder
(Stockfish 19 at 50,000 nodes, 50 pairs per level, seed-0 openings, T8).
It plays at UCI_Elo 1320, 1500, 1700 (v16's anchors: about 1,362, 1,336 and
1,399) plus 2000, 2300, 2600 and full strength, on the server's CPU. The 9M
port is being wired to play the identical ladder.

v33a result (exact depth until step 18,000, surplus after, paired with
v29a): full-solution puzzles 67.4% at T4 (-2.3), 69.7% at T8 (-1.0),
71.0% at T12, 71.3% at T16, 71.8% at T32 (+0.5, -0.1 to 1.1). 1,500+
puzzles level at T16 (+0.1). Lichess accuracy 85.8 at T8 (-0.1, -0.8 to
0.5), level to T16, then 84.6 at T32 (-1.4, -2.2 to -0.6). Validation
peaks at T8 to T12 and declines to T32 (67.4% against v29a's flat 69.8%).
Verdict: rejected by its gate (no improvement with the interval above 0).
What changed is the dynamics, not the result. Without the prefix from the
start, the state no longer settles (puzzles still rise to T32, games fall
past T12), but the extra steps are not calculation either. The ordering
alone does not make the current core calculate. v34 tests the structure
the reasoner pairs it with.

v34a stopped at step 4,000 (21:28): validation T8 60.7% against v29a's
62.1% at the same step. It was flat across budgets at every evaluation (T2
60.5%, T32 60.6%, steps 1,000 to 4,000 the same). The imagined lines
added nothing measurable to the root answer. Stopped by hand.

L3 started at 21:30 on the 5090. It runs L2's command (400,000-step schedule,
60,000-step cooldown, snapshots every 50,000) on games-v1's 10.1M labels
plus games-v2's first 14.8M (`train-labels-snap1`, 2,966 finished chunks).
That is 24.9M training positions in all. It resumes onto larger games-v2
snapshots as labelling continues. Judged by item 19's gate against L2 at equal
steps (snapshots evaluated automatically, a cooled branch at 100,000 as for
L2).

### Games ladder result (2026-09-24, 22:10)

Stockfish 19 at 50,000 nodes per move, 100 paired games per level from the
same seed-0 openings, draw rules as in v16's ladder. Wombat plays at T8 with
its 0.3 draw guard. The 9M plays by ChessBench's own play rule, one pass per
legal move and its repetition scoring. Score for the model (+wins =draws
-losses):

| Stockfish level | v16 step 12,000 | L2 cooled | ChessBench 9M |
| --- | --- | --- | --- |
| 1320 | 0.560 | 0.890 (+88 =2 -10) | 0.995 (+99 =1 -0) |
| 1500 | 0.280 | 0.645 (+62 =5 -33) | 0.995 (+99 =1 -0) |
| 1700 | 0.150 | 0.365 (+34 =5 -61) | 0.960 (+93 =6 -1) |
| 2000 | | 0.250 (+19 =12 -69) | 0.865 (+82 =9 -9) |
| 2300 | | 0.180 (+14 =8 -78) | 0.705 (+63 =15 -22) |
| 2600 | | 0.030 (+1 =4 -95) | 0.515 (+40 =23 -37) |
| full strength | 0.000 | 0.000 (+0 =0 -100) | 0.010 (+0 =2 -98) |

Implied ratings per anchor: v16 about 1,340 to 1,400. L2 about 1,600
(1500 and 1700 anchors), 1,810 (2000) and 2,040 (2300). The 9M about 2,450
(2300) and 2,610 (2600). The UCI_Elo anchors are not consistent under a
node limit, so the per-level scores are the comparison, not the implied
Elo. L2 improves on v16 by about 250 points on the common anchors. The
9M leads L2 by roughly 700 to 1,000 points of play. That is far more than the
puzzle-rating gap (2,237 against 2,047) or the Lichess accuracy gap (93.3
against 88.8) suggest. The 9M's two draws against full strength are
perpetual checks in rook endings. Why L2 loses games is being analysed
move by move with Stockfish (per-move win-chance loss, phase, error type).

## Backlog item 22: ChessBench's own training data (registered 2026-09-24, before any run)

Gap: the games ladder puts the 9M 700 to 1,000 points above L2 in play,
far more than the static yardsticks show. Our positions come from the
Lichess evaluation database (positions people asked the cloud engine
about). It over-represents openings and sharp middlegames and
under-represents the ordinary course of a game (quiet play, long endgames,
converting an advantage). ChessBench's released action-value data holds
about 530M positions sampled from 10M real Lichess games. Every legal move is
labelled by Stockfish 16 (1.1 TB in 2,148 shards of about 250,000
positions). Download costs about $0.003 per GB on the rented server.

Plan: an importer streams shards and converts them to our corpus and label
format. It drops every evaluation position (games-v1 and puzzles-v1
validation and test, the external benchmark lines) and duplicates, and
deletes each shard after conversion. First 10M positions (40 shards),
then up to 50M. Gate: a run with ChessBench positions added is judged
against L3 at equal steps on the ladder (per-level scores at 1500 to 2300,
paired openings) as well as the two static yardsticks. It is accepted if
ladder scores rise with the static yardsticks not falling below -0.5. The
games analysis (per-move error by phase and position type) decides how much
weight game positions get.

L3 step 50,000 paired with L2 step 50,000 (both snapshots, not cooled):
full-solution puzzles 76.0% at T16 (+0.2, -0.3 to 0.8). Lichess accuracy
87.1 at T12 (+0.5, -0.1 to 1.0), 87.0 at T8 (+0.4, -0.2 to 1.0). Level so
far, leaning positive on games. Each L3 position has been seen 2.5 times
less often than L2's at this step, so the data effect is expected later.

games-v2 labelling stopped at 22:37 with 3,713 of 7,840 chunks finished
(about 18.6M positions, and the run resumes from them if ever wanted). It
comes from the same evaluation-database source as games-v1. Its
remaining ten hours of 32 engines are worth less than ChessBench's
already-labelled game positions (item 22). The CPU goes to that import.

Item 22 correction and start (23:05): ChessBench shuffled individual
action-value records across all 2,148 shards (1.23 TB, about 17B records
over 578M positions). A position's moves are spread over about 29
shards, and no subset of shards holds complete positions. The importer
(`import-chessbench`) therefore streams every
shard and keeps a hash-chosen 5.2% of positions (about 30M). It deletes each
shard after reading it and joins the moves per position at the end. It keeps
only positions whose labelled moves are exactly their legal moves.
Download cost about $3.30. Their win_prob is the mover's win chance, the
same scale as our label_win. Move order is identical on 3,000 test
positions, and the best move is within 0.02 win chance of Stockfish 19 at
400,000 nodes in 27 of 30. Evaluation positions from games-v1, games-v2 and
puzzles-v1 validation and test and the external benchmark are excluded.

## Why L2 loses games: move-by-move analysis (2026-09-24)

All 43,082 positions of the ladder games so far (L2 at 1320 to 2000, the
9M at 2000 to 2600) were evaluated by Stockfish 19 at 250,000 nodes. Both
models' moves are scored on every live position (win chance 10 to 90).
Tables are in `runs/analysis/games-l2-vs-9m`.

- Not a distribution shift: L2's rate of live moves losing over 20 win
  points is 5.0% in games and 5.2% on the static set, the 9M's 1.6% and
  1.5%. On the same positions L2 errs over 20 on 5.0% against the 9M's
  1.4% (over 40: 1.1% against 0.1%). That is about a 3.5 times
  decisive-error rate, worth about 1,000 ladder Elo. Each model scores near
  50% where Stockfish's own error rate matches its own. Average accuracy
  hides it.
- After the first imbalance L2 collapses. At 2000, games where L2 errs
  first score 0.00 (the 9M 0.38). Where Stockfish errs first L2 scores
  0.45, the 9M 0.96. From a won position (80+) L2 scores 0.44 to 0.94
  by level, the 9M 0.72 to 0.94. L2's error rate in that phase is 8.6 to
  9.0%. It takes 47 to 81 plies to mate from +10 (the 9M 22 to 31). It plays
  an available mate in one 18% of the time in games (74% on held-out
  positions, the 9M 90.5%), and drew a KR vs K by the fifty-move rule.
- Error types (333 errors over 20): 36% lose material within 4 plies
  (missed opponent tactics, hung pieces), 23% are slips from winning
  positions, 19% are quiet. L2 hangs 2+ points on 1.44% of live moves, the
  9M on 0.47%.
- L2 cannot see its errors. Its value rates the position after its
  blunder within 5 points of the best move's in 70% of cases. Its move
  values prefer the blunder 77% of the time. After hanging a piece it
  values the result at 65.9 (Stockfish 21.2) until the capture is on the
  board. More thinking steps do not fix it (the same error at T16 to T64
  in 74 to 81% of cases).

Implications, each a registered item below:
- (a) Our labels come from one search per position over all moves together
  (20,000 to 50,000 nodes), so non-candidate moves get shallow scores.
  ChessBench searches every move separately, which is what teaches a model
  that a hanging move is bad.
- (b) Mate labels are all 1.0, so mate in 1 and mate in 20 are equal (item
  5). The halfmove clock is always zero in the input.
- (c) Gate on the error tail (live over-20 and over-40 rates) as well as
  average accuracy.

Item 23 (registered before any run): label quality. Relabel a sample of
20,000 games-v1 train positions with a separate search per legal move
(ChessBench style) and compare with the stored labels. The questions are how
often the stored labels rank a move that loses 20+ points (by the per-move
search) above the per-move best, and the rank correlation. If the stored
labels misrank blunders often, two things take priority over further model
changes. They are the ChessBench import (per-move labels) and a per-move
relabelling of our own data.

Item 23 result (3,000 games-v1 train positions, 90,922 moves, a separate
20,000-node search per legal move): 44,468 moves lose 20+ win points. The
stored labels put them a median 41.2 points below the stored best (true
42.5), and only 1.3% within 10 points of it. The stored top move is such a
blunder in 1.1% of positions, and a blunder outranks the per-move best in
2.2%. Best-move agreement is 53% (disagreement among close moves), median
rank correlation 0.88. Verdict: the one-search labels mark blunders clearly.
They are not why L2 misses them. The failure is in what the model learns from
these targets. Its move values prefer the blunder in 77% of game errors
although the labels do not. The next tests are on the model and the loss,
not the labels.

Fit or generalisation (L2 against the per-move truth, T8): on the 3,000
relabelled train positions the policy's top move is a 20+ blunder in 5.2%.
The move values put 5.5% of blunders within 10 points of the best. On
3,000 held-out validation positions the two numbers are 5.1% and 6.1%. Not
overfitting: L2 is as blind on positions it trained on as on new ones.
Across L2's checkpoints (held-out top-move blunder rate): 7.4% at 25,000,
6.8% at 50,000, 6.2% at 100,000, 5.1% cooled. That is about 0.6 points per
doubling of training plus the cooldown. Reaching the 9M's rate (about 1.4% in
its games) by training alone would take 50 to 100 times more. The limit is
structural. All moves are scored from one shared state of the present
position, while the 9M gives every move a full pass.

Item 22 paused (23:30) after 201 of 2,148 shards, because its two reasons are
gone. The games analysis finds no distribution shift (L2 errs at the same
rate in games as on its static positions). Item 23 finds the labels
mark blunders clearly. The fit check shows L2 underfits blunders equally
on positions it trained on, so more data is not the bottleneck. Complete
per-move positions need every shard (1.23 TB) because the records are
shuffled across all of them, which is not worth it now. If data becomes
the bottleneck, ChessBench's state-value (36 GB) and best-move (34 GB)
files hold complete records per position and can be sampled cheaply. The
imported records stay on the server, and the import resumes if needed.

## Item 24: per-move evaluation through child positions (registered 2026-09-24, before any code)

Gap: L2 rates the position after hanging a piece at 65.9 (Stockfish 21.2).
Its top choice is a 20+ blunder in 5.1% of held-out positions, the 9M's
about 1.4%. The loop refines one picture of the present position and
every move is scored from it. The per-move labels already hold the value
of every child position (a move's win chance is the child's value from
the other side). There are about 30 per labelled position, blunders
included, and training never shows those positions to the value head.

Proof before the full design: `--child-value-share S` replaces a share S
of each game batch with child positions. A child is a labelled position with
one of its labelled moves played, by exact rules from the v34 line code. Its
only target is the value, 1 minus the move's win chance for the side to move.
P24a: L2 cooled fine-tuned 5,000 steps with S = 0.5 (learning rate at
L2's cooldown start, a 1,000-step cooldown). It is paired with L2 cooled on
three measures:
- (a) The per-move truth sets: the share of 20+ blunders whose child the
  value head rates within 10 points of the best child, and the value head's
  top child being a blunder.
- (b) Child-mix move choice on the full benchmark (L2's 82.0% at T16).
- (c) Policy-only puzzles and strength, which must not fall below -0.5.
A is worth building in full (child evaluation of the top K moves at play
time, trained end to end) under two conditions. (a) improves clearly (blunder
children within 10 points falling by half), and (b) improves with the
interval above 0.

L3 step 100,000 paired with L2 step 100,000 (snapshots): full-solution
puzzles 79.4% at T16 (+0.6, 0.1 to 1.1), 79.2% at T12 (+0.5, 0.0 to 1.1).
Lichess accuracy 87.9 at T16 (+0.4, -0.3 to 1.0), level from T4 to T12.
2.5 times the positions buys about half a puzzle point at equal steps and
nothing measurable on games. That is consistent with the fit check (L2 is not
data-limited). The cooled branch of L3 is running for the final pairing.

P24a baseline (L2 cooled's value head on the child of every legal move, T8,
3,000 held-out positions with per-move truth): 8.1% of 20+ blunders are
valued within 10 points of the best child (median gap 0.299). The
best-valued child is such a blunder in 10.2% of positions, worse than the
policy's 5.1%. P24a passes (a) if the first number falls to 4% or less.
Code: `--child-value-share` and `--init-from`.
P24a is queued after L3's cooled branch.

Item 19 verdict (L3 cooled against L2 cooled, both 100,000 steps plus the
15,000-step cooldown): full-solution puzzles 81.1% at T16 (+0.5, -0.1 to
1.0), 79.9% at T8 (+0.8, 0.2 to 1.3), best 81.1% against 80.8%. Puzzle
rating 2,055 against 2,047. Lichess accuracy 88.5 at T8 (0.0, -0.6 to
0.6), best 88.5 against 88.8. Rejected by its gate: 2.5 times the
positions at equal steps gives under a puzzle point and nothing on games.
That is the same verdict as the fit check (L2 is not data-limited). L3 cooled
is kept as the best puzzle model. L2 cooled stays the reference.

P24a result (L2 cooled fine-tuned 5,000 steps, child-value share 0.5):
- (a) Blunder children valued within 10 points of the best: 4.95% against
  8.1% (median gap 0.389 against 0.299, true 0.427). Best-valued child a
  blunder: 5.4% against 10.2%. The gate asked for 4% or less.
- (b) Child-mix puzzles 80.3% at T16 against L2's 82.0% (-1.7, -2.2 to
  -1.2).
- (c) Policy puzzles 79.2% (-1.4, -1.9 to -0.9), Lichess accuracy -0.5 at
  T8, -1.2 at T12 (-1.8 to -0.6), policy top-move blunder rate 5.9% against
  5.1%.
Fails its gate as registered. Confound: re-heating a cooled model to the full
rate and cooling it again in 5,000 steps may cost this much by itself. P24b
(the same fine-tune with child share 0) runs as the control. The child
effect is P24a against P24b.

P24b control (the same fine-tune, child share 0) against L2 cooled:
policy puzzles -1.1 at T8, -1.2 at T16, so the fine-tune schedule itself
costs about a point. Child effect, P24a against P24b: the value head's
blunder blindness halves. Children of 20+ blunders within 10 points of the
best are 5.0% against 10.1%, and the best child is a blunder in 5.4% against
11.7%. Policy puzzles are level (-0.2 at T16, -0.7 to 0.3). Child-mix puzzles
-0.8 at T16 (-1.3 to -0.3) and -1.2 at T8, Lichess accuracy -0.8 at T12
and T16 (-1.3 to -0.3). The value head learns what item 24 aimed at, but
the blend does not turn it into better moves. Half of P24a's batches also
carried no policy target. Pure child choice (top 3) on both is running to
see whether the better values choose better moves on their own.

Pure child choice (top 3) on the full benchmark: P24a 71.4% at T8 and
71.9% at T16, P24b 71.8% and 71.7% (-0.4, -1.1 to 0.3, and +0.2, -0.5 to
0.9). On puzzles the better value head does not choose better moves. The
verdict on item 24 is held until the error-tail measure below exists. Its
target is blunders, which the puzzle and accuracy yardsticks measure poorly.

## Measuring playing strength (2026-09-25)

Puzzle and Lichess-accuracy averages are too coarse to show whether the
model plays better. L2 and the 9M differ by 4.5 accuracy points but about
1,000 ladder Elo. The difference lives in the error tail (live moves
losing 20+ or 40+ win points). Two layers from here on:
1. Screen: `evaluate-game-errors` on ChessBench's held-out test games
   (62,038 real-game positions with every legal move labelled by
   Stockfish, converted by the importer). It reports the rate of moves
   losing 5, 10, 20 and 40+ points on live positions, per depth, paired
   per position between runs. Calibrated first on v29a, L2, L3 and the 9M
   against the ladder order.
2. Ground truth: paired head-to-head matches (candidate against baseline,
   balanced openings, 400 to 1,000 games with a sequential stop) and the
   Stockfish ladder at the level where a model scores near 50%.
Earlier experiments rejected on accuracy (v17 to v33) are re-scored on the
screen against their registered baselines. Any that cut the error tail
get game matches.

### Error-tail screen: built and calibrated (2026-09-25)

`prepare-game-errors` copied the imported test corpus to
`data/eval/chessbench-test` with each position's full FEN. 7,417 of its
62,038 positions are in our training sets (games-v1 train 6,431, games-v2
train 972, puzzles-v1 train 28, 94% of them within the first ten moves).
They are left out, so the screen scores 54,621 positions, 42,242 of them
live. `evaluate-game-errors` plays the argmax policy over legal moves at
each depth, or the 9M by its own rule without game history. It saves
per-position losses. `compare-game-errors` pairs two runs (reports
in `runs/game-errors`). Live positions, share
of moves losing over 10, 20 and 40 win points, with 95% intervals for
the tail:

| Model | T | Mean loss | Over 10 | Over 20 | Over 40 | Accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| v29a | 4 | 3.83 | 11.15% | 4.28% (4.09 to 4.48) | 0.80% (0.72 to 0.89) | 87.6 |
| v29a | 8 | 3.39 | 9.61% | 3.42% (3.25 to 3.60) | 0.66% (0.59 to 0.74) | 88.8 |
| v29a | 16 | 3.44 | 9.75% | 3.58% (3.40 to 3.76) | 0.71% (0.63 to 0.79) | 88.7 |
| L2 cooled | 4 | 3.09 | 8.53% | 2.97% (2.82 to 3.14) | 0.56% (0.49 to 0.63) | 89.7 |
| L2 cooled | 8 | 2.53 | 6.53% | 2.14% (2.00 to 2.28) | 0.39% (0.34 to 0.46) | 91.3 |
| L2 cooled | 16 | 2.57 | 6.66% | 2.29% (2.15 to 2.44) | 0.43% (0.37 to 0.49) | 91.2 |
| L3 cooled | 8 | 2.47 | 6.32% | 2.01% (1.88 to 2.15) | 0.37% (0.32 to 0.43) | 91.5 |
| P24a | 8 | 2.70 | 7.09% | 2.38% (2.24 to 2.53) | 0.45% (0.39 to 0.52) | 90.8 |
| ChessBench 9M | | 1.03 | 1.73% | 0.38% (0.33 to 0.45) | 0.03% (0.02 to 0.06) | 96.1 |

The order matches the ladder on every column. The 9M is far ahead, and L2
well ahead of v29a (paired at T8: over 20 -1.29 points, -1.45 to -1.12, over
40 -0.27, -0.34 to -0.19). L2's over-20 rate is 5.6 times the 9M's here
against 3.6 times in the ladder games. The 9M learned this judge's labels
on positions from the same kind of games, which favours it on this set.
Every Wombat model peaks at T8 and is slightly worse at T16. Precision at
42,242 live positions: one run's over-20 rate is known to about ±0.14 to
0.17 points and its over-40 rate to ±0.06 to 0.08. Paired between two
Wombat runs the intervals are about ±0.11 to 0.17 and ±0.05 to 0.08
points. The paired Lichess accuracy is known to ±0.1 to 0.16 (the
2,000-position strength report: ±0.6 to 0.8).

L3 against L2 at T8: over 20 -0.13 (-0.24 to -0.01), over 40 -0.02
(-0.07 to 0.02), accuracy +0.16 (0.05 to 0.28). The gain is in endgames
(4 or fewer pieces: -0.36, -0.62 to -0.10). P24a against L2 at T8: over
20 +0.24 (0.14 to 0.35), over 40 +0.05 (0.01 to 0.10), accuracy -0.5, and at
T16 +0.32 and +0.10. Child-value training does not cut the policy's
error tail. P24b's checkpoint is not at home, so the child effect alone
(P24a against P24b) is not measured here.

Earlier experiments re-scored against their registered baselines at
equal steps (live positions, candidate minus baseline in points, 95%
paired intervals). Every run's best depth on the screen is T8 (v30a:
T4), so the pairs are best T against best T:

| Candidate (verdict then) | Baseline | Over 20 | Over 40 | Accuracy |
| --- | --- | --- | --- | --- |
| v17 reply pair (rejected) | v15d cooled | -0.18 (-0.35 to -0.02) | -0.04 (-0.13 to 0.05) | +0.22 |
| v18b move lines (rejected) | v15d step 5,000 | -0.20 (-0.38 to -0.02) | -0.03 (-0.13 to 0.07) | +0.15 |
| v19a depth (not accepted) | v15d cooled | -0.03 (-0.19 to 0.14) | -0.03 (-0.12 to 0.06) | +0.01 |
| v19b depth, noise (not accepted) | v15d cooled | +0.05 (-0.12 to 0.22) | -0.05 (-0.14 to 0.04) | -0.07 |
| v21a width 384 (not accepted at 6,000) | v15d cooled | -0.29 (-0.46 to -0.13) | -0.12 (-0.21 to -0.03) | +0.30 |
| v23a surprise (rejected) | v27a | +0.26 (0.11 to 0.40) | +0.08 (0.01 to 0.14) | -0.28 |
| v28a puzzle share (rejected) | v27a | +0.74 (0.59 to 0.88) | +0.15 (0.08 to 0.22) | -1.07 |
| v30a batch 512 (rejected) | v27a | +0.44 (0.28 to 0.59) | +0.07 (0.00 to 0.14) | -0.88 |
| v31c QK-norm (rejected) | v29a | +0.02 (-0.11 to 0.15) | +0.01 (-0.04 to 0.07) | -0.01 |
| v31b width 512 (rejected) | v31c | +0.16 (0.02 to 0.30) | +0.01 (-0.05 to 0.07) | -0.24 |
| v33a exact depth first (rejected) | v29a | -0.26 (-0.39 to -0.13) | -0.12 (-0.18 to -0.06) | +0.30 |
| L3 more data (item 19, rejected) | L2 cooled | -0.13 (-0.24 to -0.01) | -0.02 (-0.07 to 0.02) | +0.16 |
| P24a child values (failed its gate) | L2 cooled | +0.24 (0.14 to 0.35) | +0.05 (0.01 to 0.10) | -0.50 |

For comparison, accepted changes on the same screen: v20b (move
consequences) -1.25 and -0.65 against v21a, v26a -1.41 and -0.69, v24a
-0.38 and -0.04, v25a -0.34 and -0.02, v29a against v22a -0.81 and
-0.29. v27a (value bins) -0.07 and +0.02 against v26a and v22a (width
384) +0.00 and -0.03 against v22b are level on the tail, as they were on
games. v31a (diverged) and v20 (killed at step 1,000) have no equal-step
baseline. v34a has no checkpoint at home.

Two rejected experiments cut the error tail on both rates with the interval
clear of 0. v33a does (over 20 -0.26, over 40 -0.12, accuracy +0.30 at T8,
where the strength report had it level). v21a does too, and its width is in
the recipe since v22. Three cut only the over-20 rate, with the upper end
within 0.02 of 0. They are v18b, v17 and L3, the first to fall away among
13 comparisons. v33a is the one to take to game matches
against v29a. Its tail rises past T8 (T16 +0.16, 0.02 to 0.29), so it
plays at T8.

## State at wrap-up (2026-09-25, 01:00)

- Running on the server: 400-game matches at Stockfish 1700 (50,000
  nodes, 200 paired openings, T8) for L2 cooled, L3 cooled, P24a and
  P24b, about 75% done. They decide item 19 and
  item 24 on games. Nothing queued after them.
- Being built at home: `evaluate-game-errors`, the error-tail screen on
  ChessBench's 62,038 held-out real-game positions, calibrated against
  the ladder, then used to re-score earlier rejected experiments.
- Open, proposed and not started:
  1. Data balance. Our training set is 61% balanced positions, while games
     spend much time clearly ahead, where L2 errs on 13.5% of live moves.
     The fixes are re-weighting our 28M labelled positions toward
     unbalanced ones (cheapest) and resuming the ChessBench import (201 of
     2,148 shards done, resumable, about $3 and 5 hours for complete
     real-game positions). L3 only tested more of the same skewed data, not
     different data.
  2. Head-to-head model matches and thinking-budget matches (T4 against T8
     against T16), which need a model-against-model match mode.
  3. Self-play labelled by Stockfish as a training source.
- Best models: L2 cooled (`runs/long2-cool`, ladder about 1,600 to 2,000,
  puzzles 80.8%, accuracy 88.8) and L3 cooled (puzzles 81.1%).

Games at Stockfish 1700 (50,000 nodes, 200 paired openings, 400 games
each, T8):

| Model | Score (+ = -) | Implied Elo (95%) |
| --- | --- | --- |
| L2 cooled | 0.420 (+153 =30 -217) | about 1,644 (1,609 to 1,678) |
| L3 cooled | 0.485 (+178 =32 -190) | about 1,690 (1,656 to 1,723) |
| P24a | 0.412 (+146 =38 -216) | about 1,639 (1,608 to 1,669) |
| P24b | 0.465 (+172 =28 -200) | about 1,676 (1,641 to 1,710) |

Revised verdicts: item 19 (more game positions) helps in play by about 45
Elo although the static yardsticks showed it level. Its rejection on
puzzles and accuracy is withdrawn, and L3 cooled is the best player. Item
24's child training loses about 37 Elo against its own control and is closed.
The control's own gain over L2 (about 30 Elo) is unexplained and wants a
head-to-head. The lesson for every earlier verdict: decisions go on games
and the error-tail screen, not on puzzle and accuracy averages.

Correction to the verdicts above, from the paired openings (the same 200
openings, both colours, per-opening score difference). L3 against L2:
better on 84 openings, worse on 64, level on 52, +0.065 per game (95%
-0.004 to +0.134). P24a against P24b: -0.052 (-0.116 to +0.011). P24b
against L2: +0.045 (-0.027 to +0.117). L3 is probably the stronger player
but not proven at 400 games. Item 24's child training is probably
worse, also not proven. Both need more games (a head-to-head or 800+ per
side) or the error-tail analysis of these games before a verdict. Item
19's rejection stays withdrawn pending that.

## Overnight at home (2026-09-25, registered before running)

v35a, data balance: v29a's command sampled by balance weights
(`runs/balance-weights`, `--surprise-weights`, share 1.0). The weight per
best-win bin is the real-game share (ChessBench held-out games) over the
training share. Training has 58% of positions at best win 0.4 to 0.6 against
36% in real games, so the weights run from 0.57 to 1.86. Home GPU, 6 GiB cap,
idle priority. Judged against v29a on the error-tail screen at T8 (live
over-20 and over-40, and over-20 in positions with best win 0.75 to 0.9).
Accepted if over-20 falls with the interval clear of 0 and over-40 does not
rise.

Games at Stockfish 1700 (50,000 nodes, T8, CPU players): v33a and v29a on
seed-0's 200 openings (400 games each). L3 and L2 on 200 new openings
(seed 1), doubling their sample. v33a joins the recipe if its paired
per-opening score beats v29a with the interval above 0. L3 against L2 is
decided on the 400 paired openings of seeds 0 and 1 together.

Overnight stop (2026-09-25, about 04:10): all five home jobs were stopped
for low system memory. On the 32 GB machine a virtual machine held 4.4 GB and
other programs were running too. Saved and resumable: v35a at its
step-15,000 checkpoint (`--resume`). The 1700 matches stand at 205 (v33a),
234 (v29a), 216 (L3, seed 1) and 195 (L2, seed 1) of 400 games, and
`play-stockfish` resumes them from the PGNs.

Second pause (2026-09-25, late morning): v35a resumed with the v33a and
v29a matches. All three were paused by hand when available memory fell to
0.7 GB as another program on the machine grew to 8.7 GB. Saved: v35a's
checkpoint at step 18,000 (verified loadable), the v33a and v29a match
PGNs. v35a alone needs about 10 GB of RAM (private), each match about 1.5 GB.

v35a result (balance-weighted sampling, v29a's schedule, error-tail screen
at T8 against v29a, 42,242 live real-game positions): moves losing 20+
3.85% against 3.42% (+0.43, 0.29 to 0.57), 40+ 0.76% against 0.66% (+0.10,
0.04 to 0.17), accuracy -0.43. By balance the 20+ rate rises in every
bucket, including the one aimed at (one side better: 4.73% against 4.40%,
won: 0.82% against 0.79%). Rejected: sampling more unbalanced positions
does not teach the model to handle them and costs everywhere. The
coverage question stays open only for genuinely new game positions
(ChessBench's), not for reweighting the ones we have.

v33a at Stockfish 1700 (seed-0 openings, 400 games): +74 =51 -275, score
0.249 (about 1,508). The paired verdict against v29a waits for v29a's
match on the same openings.

v33a verdict (paired per opening against v29a, Stockfish 1700, seed-0
openings, 400 games each): v29a 0.245, v33a 0.249. v33a is better on 67
openings, worse on 66, level on 67, +0.004 per game (-0.049 to +0.056).
Level in play, so it does not join the recipe. Its error-tail gain (-0.26
points on 20+) is real on the screen but too small to show in 400 games
(on the order of 10 Elo). That sets the scale for the screen. A
difference of a few tenths of a point in the 20+ rate needs thousands of
games to confirm, and only larger screen gains are worth a match.

L3 against L2 in games, provisional (the seed-1 matches were stopped by
low system memory at 395 and 287 of 400 games, saved and resumable). On
the 343 openings both finished (200 of seed 0, 143 of seed 1, 686 games
each), L3 is better on 136, worse on 121, level on 86. That is +0.042 per
game (95% -0.010 to +0.093), about +30 Elo with an interval touching 0. With
the error-tail screen (-0.13 on 20+, -0.24 to -0.01) L3 is likely slightly
stronger, by a margin too small to confirm with a few hundred games. The
extra games-v2 data is a small gain, not the lever to close the gap to
the 9M.

## Scaling check (2026-09-26): error tail against training volume

Live over-20 rate on the error-tail screen (T8), the current recipe:
- L1 (4M labels, not cooled): 4.90% at 0.6M samples, 4.20% at 1.3M, 3.89%
  at 1.9M, 3.68% at 2.6M, 3.65% at 2.9M.
- L2 (10.1M labels, not cooled): 3.47% at 3.2M, 2.96% at 6.4M, 2.51% at
  12.8M, cooled 2.14% at 14.7M.
- L3 (24.9M, not cooled): 2.80% at 6.4M, 2.34% at 12.8M, cooled 2.01% at
  14.7M.
- v27a 4.60% (0.8M, cooled), v29a 3.42% (3.1M, cooled), the 9M 0.38%.

The curve is a straight line in log-log with no flattening. It falls about
x0.85 per doubling of training samples (exponent about -0.23). The cooldown
is worth another x0.85. More unique positions shift it down about 5 to 7% at
equal steps. Reaching 0.38% on this line needs about ten more doublings
(about 15B samples, roughly the 9M's own training scale). The gap is
therefore mainly training compute with unique data to match, not a worse
architecture per sample. Projections (extrapolated, so uncertain): 10 times
the training (about a day on one 5090) gives about 1.2%, 100 times about
0.7%. Caveats: the 9M is favoured on this set (its own data and labeller). A
larger model is likely more sample-efficient. Beyond about 100 times, the
unique positions run out without ChessBench-scale data.

L3 against L2 in games, final (Stockfish 1700, 400 paired openings over
seeds 0 and 1, 800 games each): seed 0 +0.065 per game (-0.004 to +0.134),
seed 1 -0.006 (-0.073 to +0.061). Together L3 is better on 155 openings,
worse on 143, level on 102, +0.029 per game (95% -0.019 to +0.078). Level
in play. The first 200 openings' edge was mostly chance. That is consistent
with the scaling check. games-v2 shifts the error tail about 6%, worth on the
order of 15 Elo, below what 800 games resolve. Item 19 stands as a small
real gain on the screen with no measurable game effect.

## Items 25 to 27: ideas from Stockfish's NNUE (registered 2026-09-26, before any code)

Source: the nnue-pytorch documentation and Stockfish 18's SFNNv10 (threat
inputs). NNUE itself is a fast shallow evaluator whose tactics come from
search. What can carry over to a searchless model is its input design.

- Item 25, attack pairs (after SFNNv10's threat inputs, "piece X on a
  attacks piece Y on b"). Gap: 36% of L2's big game errors lose material
  within 4 plies (hung pieces, missed opponent tactics). Wombat's board
  features give per-square summaries (attacked by N), not who attacks whom.
  Full-line attention (v18b), the nearest earlier change, re-scores at -0.20
  (-0.38 to -0.02) on the error tail. Change: a learned bias per head on the
  attention logit between two square tokens when the piece on one attacks
  or defends the piece on the other. The bias depends on attacker and target
  piece type. It is computed from the board and added through the square
  attention bias path.
- Item 26, phase heads (after NNUE's 8 layer stacks by material). Gap:
  L2's error rate differs by phase (live 20+: opening 3.6%, middlegame
  7.3%, endgame 5.3% in games) and conversion errors dominate lost wins.
  Change: the value, move-value and policy head output layers become 4
  buckets by non-pawn material, sharing everything below.
- Item 27, king-relative encoding (after HalfKA). Gap: king safety and
  endgame king activity, with weaker evidence. Change: each square token adds
  a learned embedding of its offset (clipped to 7 per axis) from its own
  side's king and from the enemy king.

Protocol (early answers, not long runs): each is v27a's command plus the
one change, 5,000 steps plus a 1,000-step cooldown on the home GPU (6 GiB
cap, idle priority). It is judged on the error-tail screen at T8 paired
against v27a (live over-20 4.60%). A pilot passes if the live over-20 rate
falls with the interval clear of 0 and over-40 does not rise. The gain is
also stated as an equivalent training multiplier (x0.85 per doubling on the
scaling curve). Passing pilots are confirmed at 22,000 steps against v29a
before anything joins the recipe.

## Item 28: L2 cooled at T = 64 against itself at T = 1 (registered 2026-11-03, before the run)

Gap: every game result so far is at T8 against Stockfish, and the plan's
thinking-budget evidence is on puzzles and the error tail. Whether extra
thinking itself wins games was never played. The v1 gate asks for T = 64
beating T = 1 by 100 Elo in 1,000 paired games on two seeds. This is the
first measurement of that comparison, on one checkpoint and one seed, so it
cannot meet the gate.

Tooling: `wombat evaluate-match` (Cute Chess, `wombat uci --checkpoint
--device`). Two changes made a run possible. The engine now reports its value
as `info ... score cp` because Cute Chess applies `-maxmoves` only when an
engine reports a score (without it two of six dry-run games ran past 300
plies, one to 718). The command also runs the current interpreter instead of
a hardcoded executable name.

Design (fixed before the run):
- Players: L2 cooled twice, one process each, `ReasoningSteps` 64 against 1,
  both on the GPU (RTX 3070 Ti), the same weights and the same move rule
  (highest legal logit, the 0.3 draw guard of the ladders). The move choice
  is deterministic.
- Budget limits: there is no clock or node limit, because the engine ignores
  `go` arguments and thinks exactly T steps (`tc=inf`). The thinking budget
  is T. A game ends at mate, stalemate, insufficient material, the fifty-move
  rule, a third repetition, or as a draw after 150 full moves (300 plies, the
  cap of `play-stockfish`).
- Openings: the first 150 of `select_openings` on `games-v1/test` with seed 0
  (balanced, 26+ pieces). They are the same openings, in the same order, as
  the first 150 of the 200 used in the ladders and the 1700 matches. Each
  opening is played twice with the colours swapped, 150 pairs, 300 games.
- Size: the ladders used 100 games per level and the 1700 checks 400 to 800.
  A dry run of 6 openings (12 games) scored 0.833 with pair interval 0.739
  to 0.928, which at 150 pairs gives a half-width near 0.03 (about 25 Elo).
  300 games take about 19 s each at concurrency 3 (12 games in 231 s), 1.6
  hours. The 1,000 games of the gate would take 5.3 hours at 1.4 GB per
  engine process, so they are not played here.
- Concurrency 3 (memory bound), seed 0.

Measure: score of T = 64 per game and its 95% interval from the per-opening
pair variance (`pair_statistics`, as in the Stockfish matches), and the Elo
difference with that interval.

Pass marks (fixed now):
- Extra thinking adds in play: the lower end of the score interval is above
  0.500.
- It reaches the gate's size: the lower end is at least 0.640 (100 Elo).
- Otherwise, if the interval contains 0.500 the README keeps "no measured
  gain", and if the upper end is below 0.500 the extra steps hurt and it
  says so.

### Item 28 result (2026-11-03)

T = 64 scored 0.928 per game (+260 =37 -3 over 300 games, 150 openings with
both colours), 95% interval 0.906 to 0.951, about +445 Elo (+393 to +516).
Both pass marks are met: the lower end is above 0.500 and above 0.640. The
run took 45 minutes, about half the 1.6 hours registered, because games
averaged 88 plies and 20 of 300 hit the cap. One checkpoint and one opening
set, so the v1 gate's 1,000 games on two seeds is still open, and T = 64
was not played against T = 8.

## Item 29: L2 cooled at T = 64 and T = 8 in games (registered 2026-11-03, before the runs)

Gap: item 28 showed that T = 64 beats T = 1 (0.928 per game), which mostly
shows that more steps than one help. The training ran the network at about
T = 8 and the earlier evaluations found gains stop at the trained depth
(Lichess accuracy peaks near T = 12 and falls by T = 32). Two matches were
not played: whether compute past the trained depth adds playing strength or
hurts, and how much of item 28's gain is already there at the trained depth.

Design (the same method as item 28, fixed before the runs): L2 cooled for
both players, one process each on the GPU, `ReasoningSteps` set per player,
the same move rule, the first 150 openings of `select_openings` on
`games-v1/test` with seed 0, both colours (300 games), 300-ply cap, no
adjudication, concurrency 3, one match at a time.

- Match A: T = 64 (scored) against T = 8.
- Match B: T = 8 (scored) against T = 1.

Measure: score of the scored engine per game with its 95% interval from the
per-opening pair variance, and the Elo difference with that interval.

Pass marks (fixed now). Match A:
- Compute past the trained depth adds strength: the lower end of the score
  interval is above 0.500.
- It hurts: the upper end is below 0.500.
- Otherwise no measured difference at this size. Then T = 16 and T = 32 are
  each played against T = 8 the same way (matches C and D) to see where the
  curve turns. If A is decided either way, C and D are not played.

Match B:
- The trained depth beats one step: the lower end is above 0.500.
- Most of item 28's gain is already there at T = 8: the point estimate of the
  Elo difference is at least half of item 28's +445 (+223, score 0.78).
  Below that, steps past 8 contribute the larger part.
- A consistency check, not a pass mark: Elo(A) + Elo(B) against item 28's
  +445. A large mismatch means Elo from different pairings is not additive
  here and the three matches are read separately.

Not shown whatever the outcome: one checkpoint, one opening set, 300 games
per match, a deterministic move rule, so no statement about other
checkpoints or seeds and not the v1 gate (1,000 paired games on two seeds).

### Item 29 result (2026-11-03)

Match A, T = 64 against T = 8: T = 64 scored 0.305 per game (+77 =29 -194
over 300 games, 150 openings with both colours), 95% interval 0.260 to
0.350, about -143 Elo (-182 to -107). The upper end is below 0.500, so
compute past the trained depth hurts in play. C and D were not played, as
registered. 59.5 minutes, 111 plies per game on average.

Match B, T = 8 against T = 1: T = 8 scored 0.968 per game (+281 =19 -0),
95% interval 0.954 to 0.982, about +594 Elo (+528 to +699). The lower end is
above 0.500 and the point estimate is above +223, so all of item 28's gain
is there at the trained depth. T = 8 beat T = 1 by more than T = 64 did
(0.968 against 0.928, intervals not overlapping). 13 minutes, 76 plies per
game on average. The Elo at a score near 0.97 is imprecise, and the scores
are the firmer numbers.

Consistency check: -143 + 594 = +451 against item 28's +445, so Elo adds
across the three pairings here.

Limits as registered: one checkpoint, one opening set, 300 games per match,
a deterministic move rule. The v1 gate (T = 64 beats T = 1 by 100 Elo over
1,000 paired games on two seeds) is met in sign and size by item 28 but not
in game count or seeds.
