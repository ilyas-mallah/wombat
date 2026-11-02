# What L2's thinking steps do on hard puzzles

Models: the cooled L2 (`runs/long2-cool/checkpoint.pt`, step 115,000) is the subject; L2 step
50,000 is the comparison. Puzzles: the 10,000 benchmark puzzles with rating 1,500+ and 2+ solver
moves (4,357). Groups use each model's own stored verdict at its decision depth (T16 cooled, T8
50k) and the 9M's.

| Group | Meaning | Pool cooled | Pool 50k | Sampled |
| --- | --- | --- | --- | --- |
| A | L2 solves | 2,557 | 2,137 | 500, matched to B on solver moves (cap 4) and rating |
| B | L2 fails, 9M solves | 1,058 | 1,403 | 500 |
| C | both fail | 742 | 817 | 300 |

All solver positions traced T0 to T32. "Fail@1" means the first wrong move is solver move 1 (B:
295 cooled, 317 at 50k); "fail@2+" later (205, 181). A mating alternative counts as correct. The
fp32 rerun reproduces the stored verdicts on all but 1 of 1,300 puzzles per model. Scripts:
session scratchpad `reasoning/` (`WOMBAT_MODEL=cool` selects the cooled model); JSON outputs in
`out/` and `out_cool/` there.

## Mechanism

The loop settles; it does not search. Each step refines one picture of the current position: it
sharpens the choice among moves proposed at T1 and makes the reply to the favoured move
decodable, but it never represents the position two or more plies ahead, and its per-step change
shrinks geometrically, the same on easy and hard puzzles. Moves are scored from origin and
destination tokens plus one-move-later static features, so quiet moves and sacrifices lose to
moves that win something at once. In B the solution stays a runner-up: top 3 in about 85% of
cases, probability flat near 0.2 from T2 to T32. The knowledge exists: once the first move and
reply are played, L2 plays the remaining solver moves correctly 90% of the time, and its own
value head, applied after each legal move, finds the solution in 38 to 46% of B failures.

## 1. State dynamics

Per-square change of H in one step, and distance to H32 (A first moves; B and C within 3% of A
at every step):

| Step | 1 | 2 | 3 | 4 | 6 | 8 | 12 | 16 | 17 | 24 | 32 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 50k change | 10.3 | 5.1 | 2.9 | 1.8 | 0.98 | 0.55 | 0.21 | 0.08 | 0.07 | 0.02 | 0.01 |
| 50k distance to H32 (T0: 12.0) | 6.0 | 4.3 | 3.4 | 2.9 | 2.0 | 1.4 | 0.56 | 0.21 | | | 0 |
| cooled change | 5.3 | 2.7 | 1.7 | 1.15 | 0.60 | 0.34 | 0.19 | 0.22 | 0.225 | 0.15 | 0.06 |
| cooled distance to H32 (T0: 6.1) | 3.7 | 2.8 | 2.5 | 2.3 | 2.1 | 2.0 | 1.6 | 1.1 | | | 0 |

At 50k the state is 89% of the way to its fixed point by T8 (policy KL per step 0.005): hence
T8 = T32. The cooled model has a second slow phase peaking at T17 and T18, just past T16, the
deepest total the loop saw in training (8 prefix plus 8 graded steps); this is where its puzzle
accuracy still moves (79.1% at T8, 80.8% at T32). Difficulty does not change the dynamics: B's
choices settle later (last argmax change at step 9.1 against 4.6 for A, cooled), but the loop
does no extra work on them.

## 2. Trajectory at the failing position (cooled)

| Step | 0 | 1 | 2 | 4 | 8 | 12 | 16 | 32 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A first: P(solution) | 0.08 | 0.39 | 0.51 | 0.61 | 0.61 | 0.63 | 0.68 | 0.74 |
| B fail@1: P(solution) | 0.06 | 0.20 | 0.23 | 0.22 | 0.20 | 0.20 | 0.20 | 0.24 |
| B fail@1: top 1 | 0.13 | 0.23 | 0.26 | 0.23 | 0.13 | 0.07 | 0 | 0.17 |
| B fail@1: top 3 | 0.31 | 0.64 | 0.76 | 0.84 | 0.83 | 0.85 | 0.87 | 0.88 |
| C fail@1: P(solution) | 0.06 | 0.15 | 0.17 | 0.16 | 0.15 | 0.15 | 0.17 | 0.20 |

(50k: A rises 0.09, 0.42, 0.52, 0.63 by T4, then flat; B fail@1 flat at 0.17 to 0.20 from T2.)

| Failure class | cooled B (500) | cooled C (300) | 50k B (498) |
| --- | --- | --- | --- |
| Found then abandoned | 53% | 41% | 40% |
| Found too late (after the decision depth) | 6% | 3% | 5% |
| Runner-up, never top 1 | 37% | 50% | 47% |
| Never top 3 | 4% | 7% | 7% |

In cooled B the wrong move became the favourite for good at T0 to T1 in 30% of cases, T2 to T3
16%, T4 to T5 15%, T6 to T8 18%, T9 to T16 22%; 63 of 500 become correct by T32. Groups are
selected on the answer at the decision depth, so "found then abandoned" is partly noise; the
robust signal is the flat probability with top-3 membership near 85%.

## 3. Solutions, wrong moves, and engine depth

Cooled, with 50k in brackets:

| | A first (solution) | B fail@1 solution | B fail@1 wrong move | C fail@1 solution |
| --- | --- | --- | --- | --- |
| Quiet | 33% (31) | 46% (47) | 39% (38) | 53% (50) |
| Sacrifice (static exchange below 0) | 19% (15) | 23% (30) | 19% (13) | 29% (32) |
| Wins material at once (static exchange above 0) | 20% (22) | 13% (10) | 15% (22) | 6% (6) |
| Capture | 41% | 37% | 41% | 26% |
| Check | 39% | 26% | 35% | 31% |
| Opponent's legal replies after the solution | 18.5 | 24.0 | | 20.8 |

Stockfish (depth 14, single-thread engines):

| | A first | B first | C first | B fail@1 | C fail@1 |
| --- | --- | --- | --- | --- | --- |
| Depth from which the solution stays best (median) | 1 | 1 | 3 | 1 | 4 |
| Needed depth 4 or less | 80% | 81% | 64% | 79% | 54% |
| Wrong move win-chance loss (median) | | 54 | 43 | 54 | 43 |
| Wrong move loses over 20 points | | 98% | 96% | 98% | 96% |
| Wrong move fine at depth 2, losing at depth 14 | | 27% | 28% | 27% | 28% |

B is not deeper than A for a searcher; C is. L2's wrong moves turn a won position into a lost
one and are mostly refuted within 2 plies.

## 4. The knowledge is present

| Later solver moves correct, after failing move 1 | cooled | 50k |
| --- | --- | --- |
| B | 90% (469) | 92% (490) |
| C | 81% (313) | 80% (361) |

One ply with L2's own value head: play every legal move, read the child position at the decision
depth, choose the move worst for the opponent.

| | Policy | One ply, child at T1 | One ply, child at decision depth | Solution in child-value top 3 |
| --- | --- | --- | --- | --- |
| cooled A first (300) | 99.7% | 37% | 68% | 91% |
| cooled B fail@1 (295) | 0% | 22% | 38% | 83% |
| cooled B fail@2+ (205) | 0% | 24% | 46% | 89% |
| cooled C fail@1 (163) | 0% | 17% | 25% | 74% |
| 50k B fail@1 | 0% | 20% | 37% | 77% |
| 50k B fail@2+ | 0% | 20% | 39% | 80% |
| 50k C fail@1 | 0% | 19% | 18% | 62% |

Hybrids (cooled):

| Selection | A first | B fail@1 | B fail@2+ | C fail@1 |
| --- | --- | --- | --- | --- |
| Policy | 99.7% | 0% | 0% | 0% |
| Policy top 2 re-ranked by child value | 81% | 45% | 44% | 23% |
| Policy top 3 re-ranked by child value | 75% | 45% | 48% | 28% |
| log p + 1 x child value | 96% | 14% | 14% | 5% |
| log p + 2 x child value | 93% | 21% | 22% | 10% |
| log p + 4 x child value | 91% | 31% | 30% | 15% |
| log p + 8 x child value | 86% | 38% | 40% | 17% |

Each hybrid trades A for B. The groups are selected on the policy, so these rates bound the
whole-benchmark effect only loosely; a full benchmark run of the blend is the test. The 9M
scores every legal move with its own forward pass, which is the capability this points at.

## 5. Probes for the continuation (cooled)

Linear square probes (one score per square token, softmax over squares), trained on 3,000
test-split first moves (1,500+, 2+ moves, none in the benchmark). Columns: A / B first move right
(205) / B first move wrong (295) / C wrong (163). Majority-square baselines 0.01 to 0.06.

| Target | T0 | T2 | T4 | T8 | T16 | T32 |
| --- | --- | --- | --- | --- | --- | --- |
| Solution destination | .29 .41 .24 .17 | .53 .66 .31 .22 | .64 .76 .29 .26 | .71 .84 .29 .23 | .75 .83 .29 .22 | .73 .82 .30 .23 |
| Reply origin | .39 .30 .29 .31 | .43 .45 .40 .34 | .49 .51 .43 .36 | .51 .59 .44 .36 | .52 .56 .42 .35 | .54 .55 .43 .36 |
| Reply destination | .14 .13 .15 .08 | .32 .38 .24 .19 | .36 .44 .26 .19 | .42 .52 .24 .25 | .44 .52 .25 .20 | .43 .49 .25 .24 |
| Second move origin | .19 .27 .17 .20 | .24 .26 .23 .23 | .25 .26 .22 .21 | .26 .27 .23 .18 | .29 .31 .23 .20 | .27 .28 .21 .23 |
| Second move destination | .12 .09 .12 .14 | .18 .12 .20 .18 | .22 .11 .19 .17 | .21 .15 .20 .24 | .23 .14 .20 .20 | .23 .15 .22 .22 |
| Material at line end, r | .30 .40 .34 .43 | .38 .38 .28 .39 | .41 .37 .36 .29 | .41 .38 .34 .30 | .41 .35 .30 .30 | .41 .36 .31 .34 |
| Line ends in mate, AUROC | .72 .94 .64 .87 | .84 .83 .85 .85 | .91 .91 .90 .87 | .94 .92 .93 .85 | .94 .93 .94 .87 | .94 .91 .95 .86 |

The model's own reply head reaches reply-destination accuracy 0.42 (A) and 0.51 (B right) by T8,
flat after. Its future-board heads get 15 to 31% of changed squares, never above their T0 value.
At 50k the reply saturates by T4, second-move destination 0.12 to 0.18 in A, material r flat
near 0.4. The loop computes the move and the reply for its favoured move, and little beyond.

## 6. Not informative

- Attention: king-local by construction, so attention to distant line squares is zero. Gradient
  attribution of the solution's logit is identical across A, B and C (at T16: 14% on the move's
  own squares, 33% one king step away, 24% two, 29% three or more; line squares 11 to 12%
  against a uniform 5% in every group).
- Line spread on the board: 4.8 king steps (A), 5.0 (B), 4.8 (C).
- Move-value head as a chooser: 17% on B fail@1, 84% on A; it shares the policy's blind spot.
- Root value rises from about -0.05 to 0.1 to 0.3 in every group and never reads these positions
  as won.

## 7. What to change

The fault is in the move readout (`_square_policy`, `_move_value_scores`): it scores a move on a
hidden state that only holds the present position.

1. Explicit child evaluation (high confidence): L2's own value head, applied after each move,
   recovers 38 to 46% of B failures. Test the blend at inference first, then teach the policy
   the child-value choice.
2. Imagined positions in the loop and supervision past the reply (moderate, untested): scratch
   boards seeded with the model's top candidates; second-move and line-outcome targets from
   puzzle lines.
3. Deeper training alone (v32a) is unlikely to close B (moderate): the solution's probability is
   flat from T2 to T32.

## Caveats

Groups are selected on the answer at the decision depth, so per-step top-1 rates at that depth
are 0 or 1 by construction. Traces ran in fp32 while the stored verdicts came from bf16 runs (1
of 1,300 puzzles differs per model). Probes are linear. Stockfish at depth 14; "needed depth"
includes its capture search.
