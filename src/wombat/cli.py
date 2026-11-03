"""Command-line entry points."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import psutil


def _doctor() -> int:
    import chess
    import numpy
    import torch
    import yaml
    import zstandard
    from wombat.settings import load_executables

    disk = psutil.disk_usage(Path.cwd().anchor)
    report = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": {
            "torch": torch.__version__,
            "numpy": numpy.__version__,
            "python_chess": chess.__version__,
            "pyyaml": yaml.__version__,
            "zstandard": zstandard.__version__,
        },
        "cuda": {
            "available": torch.cuda.is_available(),
            "runtime": torch.version.cuda,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "memory_gb": round(psutil.virtual_memory().total / 2**30, 1),
        "disk_free_gb": round(disk.free / 2**30, 1),
        "executables": {"git": shutil.which("git"), **load_executables()},
    }
    print(json.dumps(report, indent=2))
    return 0 if report["cuda"]["available"] else 1


def _eval_smoke(output: Path, device_name: str, seed: int) -> int:
    import chess
    import torch

    from wombat.eval import PositionCase, PositionManifest, evaluate_agreement
    from wombat.model import WombatNetwork, count_parameters
    from wombat.model.adapter import TorchPolicyProvider

    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    after_e4 = chess.Board()
    after_e4.push_uci("e2e4")
    manifest = PositionManifest(
        name="wombat-smoke-v1",
        positions=(
            PositionCase("start", chess.STARTING_FEN, "e2e4", ("opening",)),
            PositionCase("after-e4", after_e4.fen(), "c7c5", ("opening",)),
            PositionCase(
                "promotion",
                "4k3/P7/8/8/8/8/8/4K3 w - - 0 1",
                "a7a8q",
                ("promotion",),
            ),
        ),
    )
    manifest.validate()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    manifest.write(manifest_path)

    model = WombatNetwork()
    checkpoint_path = output / "untrained.pt"
    torch.save({"seed": seed, "model": model.state_dict()}, checkpoint_path)
    checkpoint_hash = sha256(checkpoint_path.read_bytes()).hexdigest()
    provider = TorchPolicyProvider(model, device)
    evaluations = []
    for steps in (1, 2, 4, 8, 16, 32, 64, 128):
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        result = evaluate_agreement(provider, manifest, steps)
        if device.type == "cuda":
            torch.cuda.synchronize()
        evaluations.append(
            {
                **asdict(result),
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
            }
        )

    report = {
        "schema": "wombat.evaluation-report.v1",
        "checkpoint_sha256": checkpoint_hash,
        "manifest_sha256": manifest.sha256,
        "seed": seed,
        "device": str(device),
        "parameter_count": count_parameters(model),
        "evaluations": evaluations,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


def _regret_smoke(output: Path, nodes: int) -> int:
    import chess

    from wombat.eval.manifest import PositionCase, PositionManifest
    from wombat.eval.regret import StockfishRegretEvaluator, evaluate_regret
    from wombat.policy import FixturePolicy
    from wombat.settings import load_executables

    executable = load_executables()["stockfish"]
    if executable is None:
        raise SystemExit("Stockfish is not configured; set WOMBAT_STOCKFISH or configs/local.yaml")
    board = chess.Board()
    white_mate = chess.Board("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1")
    black_mate = chess.Board("8/8/8/8/8/6k1/5q2/7K b - - 0 1")
    manifest = PositionManifest(
        name="regret-smoke-v1",
        positions=(
            PositionCase("same-move", board.fen(), "e2e4"),
            PositionCase("white-mate", white_mate.fen(), "f7f8"),
            PositionCase("black-mate", black_mate.fen(), "f2f1"),
        ),
    )
    provider = FixturePolicy(
        {board.fen(): "e2e4", white_mate.fen(): "f7f8", black_mate.fen(): "f2f1"}
    )
    output.mkdir(parents=True, exist_ok=True)
    with StockfishRegretEvaluator(Path(executable), nodes, output / "cache") as evaluator:
        report = evaluate_regret(provider, manifest, evaluator, reasoning_steps=1)
        report["engine"] = evaluator.engine_label
        report["nodes"] = nodes
    mate_rows = [row for row in report["positions"] if str(row["id"]).endswith("mate")]
    if any(int(row["reference_cp"]) <= 0 for row in mate_rows):
        raise RuntimeError("score perspective check failed: winning root moves must score positive")
    (output / "regret_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return 0 if report["mean_regret_cp"] == 0 else 1


def _evaluate_regret_curve(args: argparse.Namespace) -> int:
    """Evaluate checkpoint moves with a fixed-node Stockfish oracle."""

    import statistics
    import torch

    from wombat.eval.manifest import PositionManifest
    from wombat.eval.regret import StockfishRegretEvaluator, evaluate_regret
    from wombat.model import NetworkConfig, WombatNetwork
    from wombat.model.adapter import TorchPolicyProvider
    from wombat.settings import load_executables

    executable = args.stockfish or load_executables()["stockfish"]
    if executable is None:
        raise SystemExit("Stockfish is not configured")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = WombatNetwork(NetworkConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model"], strict=False)
    provider = TorchPolicyProvider(model, args.device)
    source_manifest = PositionManifest.read(args.manifest)
    manifest = PositionManifest(
        name=f"{source_manifest.name}-first-{args.limit}",
        positions=source_manifest.positions[: args.limit],
    )
    depths = tuple(int(item) for item in args.depths.split(","))
    result: dict[str, object] = {
        "schema": "wombat.stockfish-regret-curve.v1",
        "checkpoint": str(args.checkpoint),
        "nodes": args.nodes,
        "positions": len(manifest.positions),
        "reports": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with StockfishRegretEvaluator(
        Path(executable), args.nodes, args.cache_dir
    ) as evaluator:
        result["engine"] = evaluator.engine_label
        for depth in depths:
            report = evaluate_regret(provider, manifest, evaluator, depth)
            regrets = [int(row["regret_cp"]) for row in report["positions"]]
            report["median_regret_cp"] = statistics.median(regrets)
            report["mean_regret_cp_clipped_10k"] = statistics.fmean(
                min(10_000, max(0, value)) for value in regrets
            )
            report["exact_reference_rate"] = sum(value == 0 for value in regrets) / len(regrets)
            result["reports"].append(report)
            args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            print(
                json.dumps(
                    {
                        "reasoning_steps": depth,
                        "median_regret_cp": report["median_regret_cp"],
                        "mean_regret_cp_clipped_10k": report[
                            "mean_regret_cp_clipped_10k"
                        ],
                        "exact_reference_rate": report["exact_reference_rate"],
                    }
                ),
                flush=True,
            )
    return 0


def _match_smoke(output: Path) -> int:
    from wombat.eval.matches import run_paired_smoke
    from wombat.settings import load_executables

    executable = load_executables()["cutechess_cli"]
    if executable is None:
        raise SystemExit("Cute Chess CLI is not configured")
    output.mkdir(parents=True, exist_ok=True)
    result = run_paired_smoke(
        Path(executable),
        Path(sys.executable),
        Path.cwd(),
        output / "games.pgn",
    )
    summary = asdict(result)
    (output / "match_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(result.stdout)
    return 0


def _evaluate_match(args: argparse.Namespace) -> int:
    import chess

    from wombat.eval.games import select_openings
    from wombat.eval.matches import checkpoint_match_command, summarize_pgn
    from wombat.settings import load_executables

    if args.first_steps == args.second_steps:
        raise SystemExit("--first-steps and --second-steps must differ")
    cutechess = load_executables()["cutechess_cli"]
    if cutechess is None:
        raise SystemExit("Cute Chess CLI is not configured")
    args.output.mkdir(parents=True, exist_ok=True)
    fens = select_openings(args.openings, args.pairs, seed=args.seed)
    openings = args.output / "openings.epd"
    openings.write_text("\n".join(chess.Board(fen).epd() for fen in fens) + "\n", encoding="utf-8")
    pgn = args.output / "games.pgn"
    command = checkpoint_match_command(
        Path(cutechess),
        Path(sys.executable),
        Path.cwd(),
        args.checkpoint.resolve(),
        openings.resolve(),
        pgn.resolve(),
        first_steps=args.first_steps,
        second_steps=args.second_steps,
        pairs=args.pairs,
        device=args.device,
        max_moves=args.max_moves,
        concurrency=args.concurrency,
    )
    with (args.output / "cutechess.log").open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=Path.cwd(),
            check=False,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=args.timeout,
        )
    if completed.returncode != 0:
        raise RuntimeError(f"Cute Chess failed, see {args.output / 'cutechess.log'}")
    summary = summarize_pgn(pgn, f"Wombat-H{args.first_steps}", f"Wombat-H{args.second_steps}")
    report = {
        "schema": "wombat.paired-compute-match.v1",
        "checkpoint": str(args.checkpoint),
        "first_steps": args.first_steps,
        "second_steps": args.second_steps,
        "device": args.device,
        "seed": args.seed,
        "openings": str(args.openings),
        "command": command,
        **summary,
    }
    (args.output / "match_summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return 0


def _trace_smoke(output: Path, steps: int, device_name: str) -> int:
    import chess
    import torch

    from wombat.eval.tracing import trace_positions
    from wombat.model import WombatNetwork

    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    model = WombatNetwork().to(device_name).eval()
    boards = [chess.Board(), chess.Board("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1")]
    summary = trace_positions(model, boards, steps, output)
    print(json.dumps({key: value for key, value in summary.items() if key != "steps"}, indent=2))
    return 0


def _build_puzzles(source: Path, output: Path, limit: int) -> int:
    from wombat.data.build import build_puzzle_manifest

    manifest = build_puzzle_manifest(source, output, limit)
    print(
        json.dumps(
            {
                "output": str(output),
                "positions": len(manifest.positions),
                "sha256": manifest.sha256,
            },
            indent=2,
        )
    )
    return 0


def _build_corpus(source: Path, output: Path, limit: int, puzzle_manifest: Path) -> int:
    from wombat.data.build import build_evaluation_corpus
    from wombat.eval.manifest import PositionManifest

    puzzles = PositionManifest.read(puzzle_manifest)
    report = build_evaluation_corpus(
        source,
        output,
        limit,
        excluded_fens=(case.fen for case in puzzles.positions),
    )
    print(json.dumps(report, indent=2))
    return 0 if report["written_records"] == limit else 1


def _build_puzzle_corpus(args: argparse.Namespace) -> int:
    from wombat.data.puzzle_corpus import build_puzzle_corpus
    from wombat.eval.manifest import PositionManifest

    frozen = PositionManifest.read(args.puzzle_manifest)
    manifest = build_puzzle_corpus(
        args.source,
        args.output,
        excluded_puzzle_ids=(case.id for case in frozen.positions),
        excluded_fens=(case.fen for case in frozen.positions),
        limit=args.limit,
        workers=args.workers,
    )
    print(json.dumps(manifest, indent=2))
    return 0


def _build_game_corpus(args: argparse.Namespace) -> int:
    import numpy as np

    from wombat.data.game_corpus import build_game_corpus
    from wombat.eval.manifest import PositionManifest

    frozen = PositionManifest.read(args.puzzle_manifest)
    excluded = [case.fen for case in frozen.positions]
    for split in args.exclude_splits:
        excluded.extend(str(fen) for fen in np.load(split / "fens.npy"))
    manifest = build_game_corpus(
        args.source,
        args.output,
        limit=args.limit,
        excluded_fens=excluded,
        skip_records=args.skip_records,
        workers=args.workers,
    )
    print(json.dumps(manifest, indent=2))
    return 0


def _import_chessbench(args: argparse.Namespace) -> int:
    import numpy as np

    from wombat.data.chessbench import (
        TRAIN_URL,
        fen_keys,
        import_chessbench,
        parse_shards,
        puzzle_line_fens,
        shard_name,
    )
    from wombat.data.puzzle_corpus import lower_priority
    from wombat.eval.manifest import PositionManifest

    lower_priority()
    exclusion: dict[str, object]
    if args.exclusion_keys is not None:
        excluded = np.load(args.exclusion_keys)
        exclusion = {"keys": str(args.exclusion_keys)}
    else:
        fens = [case.fen for case in PositionManifest.read(args.puzzle_manifest).positions]
        for split in args.exclude_splits:
            fens.extend(str(fen) for fen in np.load(split / "fens.npy"))
        for puzzles in args.exclude_puzzles:
            fens.extend(puzzle_line_fens(puzzles))
        for positions in args.exclude_positions:
            fens.extend(line for line in positions.read_text(encoding="utf-8").splitlines() if line)
        excluded = fen_keys(fens)
        exclusion = {
            "puzzle_manifest": str(args.puzzle_manifest),
            "splits": [str(split) for split in args.exclude_splits],
            "puzzles": [str(path) for path in args.exclude_puzzles],
            "positions_files": [str(path) for path in args.exclude_positions],
            "fens": len(fens),
        }
    shards = parse_shards(args.shards)
    sources = [(shard_name(index), TRAIN_URL.format(index=index)) for index in shards]
    sources += [(bag.stem, str(bag)) for bag in args.bags]
    manifest = import_chessbench(
        sources,
        args.output,
        work_dir=args.work_dir,
        fraction=args.fraction,
        excluded=excluded,
        exclusion=exclusion,
        workers=args.workers,
        join=args.join,
        partitions=args.partitions,
        delete_records=args.delete_records,
    )
    if manifest is not None:
        summary = {key: value for key, value in manifest.items() if key != "shards"}
        print(json.dumps(summary, indent=2))
    return 0


def _label_moves(args: argparse.Namespace) -> int:
    from wombat.data.move_labels import label_positions
    from wombat.settings import load_executables

    stockfish = args.stockfish or load_executables()["stockfish"]
    if stockfish is None:
        raise SystemExit("Stockfish is not configured")
    manifest = label_positions(
        args.split,
        args.output,
        count=args.count,
        nodes=args.nodes,
        engine_command=[str(stockfish)],
        workers=args.engines,
        exclude=args.exclude,
        reverse=args.reverse,
    )
    print(json.dumps(manifest, indent=2))
    return 0


def _merge_move_labels(args: argparse.Namespace) -> int:
    from wombat.data.move_labels import merge_labels

    manifest = merge_labels(args.inputs, args.output, allow_mixed_nodes=args.allow_mixed_nodes)
    print(json.dumps(manifest, indent=2))
    return 0


def _train_puzzles(args: argparse.Namespace) -> int:
    from wombat.model import NetworkConfig
    from wombat.puzzle_training import PuzzleTrainingConfig, train_puzzles

    normalized = args.stability == "normalized"
    model_config = NetworkConfig(
        d_model=args.d_model,
        attention_heads=args.heads,
        mlp_width=args.mlp_width,
        prelude_blocks=args.prelude_blocks,
        recurrent_blocks=args.recurrent_blocks,
        latent_tokens=args.latent_tokens,
        policy_readout="square",
        move_readout=args.move_readout,
        normalize_state=normalized,
        max_residual_update=None if normalized else 0.25,
        use_update_gate=not normalized,
        square_attention_bias=args.square_bias,
        lookahead_heads=args.lookahead_weight > 0 or args.reply_weight > 0,
        reply_readout=args.reply_readout,
        scratch_boards=args.scratch_boards,
        flat_scratch=args.flat_scratch,
        refresh_board=args.refresh_board,
        local_attention=args.local_attention,
        local_neighbourhood=args.local_neighbourhood,
        board_features=args.board_features,
        board_features_long_range=args.board_features_long_range,
        move_consequences=args.move_consequences,
        relative_attention=args.relative_attention,
        state_attention_bias=args.state_attention_bias,
        state_attention_bias_width=args.state_attention_bias_width,
        qk_norm=args.qk_norm,
        value_bins=args.value_bins,
        attack_pairs=args.attack_pairs,
        phase_heads=args.phase_heads,
        king_relative=args.king_relative,
        core=args.core,
        workspace_tokens=args.workspace_tokens,
        steps_per_ply=args.steps_per_ply,
        line_plies=args.line_plies,
        readout_blocks=args.readout_blocks,
    )
    config = PuzzleTrainingConfig(
        corpus=str(args.corpus),
        output=str(args.output),
        task=args.task,
        labels=None if args.labels is None else str(args.labels),
        extra_train=tuple((str(corpus), str(labels)) for corpus, labels in args.extra_train or ()),
        temperature=args.temperature,
        move_value_weight=args.move_value_weight,
        value_weight=args.value_weight,
        lookahead_weight=args.lookahead_weight,
        reply_weight=args.reply_weight,
        supervise_every_step=args.supervise_every_step,
        ema_decay=args.ema_decay or None,
        recompute_steps=args.recompute_steps,
        snapshot_every=args.snapshot_every,
        steps=args.steps,
        stop_after=args.stop_after,
        cooldown_steps=args.cooldown_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        max_depth=args.max_depth,
        max_gradient_steps=args.max_gradient_steps,
        fixed_depth=args.fixed_depth,
        progressive_share=args.progressive_share,
        min_gradient_steps=args.min_gradient_steps,
        random_prefix=args.random_prefix,
        random_prefix_from=args.random_prefix_from,
        depth_distribution=args.depth_distribution,
        depth_mean=args.depth_mean,
        depth_max=args.depth_max,
        initial_noise=args.initial_noise,
        puzzle_corpus=None if args.puzzle_corpus is None else str(args.puzzle_corpus),
        puzzle_share=args.puzzle_share,
        puzzle_remaining=tuple(int(item) for item in args.puzzle_remaining.split(",")),
        puzzle_steps_per_move=args.puzzle_steps_per_move,
        puzzle_max_tail=args.puzzle_max_tail,
        surprise=None if args.surprise_weights is None else str(args.surprise_weights),
        surprise_share=args.surprise_share,
        child_value_share=args.child_value_share,
        init_from=None if args.init_from is None else str(args.init_from),
        lines_min=args.lines_min,
        lines_max=args.lines_max,
        line_plies_min=args.line_plies_min,
        line_plies_max=args.line_plies_max,
        lines_max_final=args.lines_max_final,
        line_plies_max_final=args.line_plies_max_final,
        budget_ramp_from=args.budget_ramp_from,
        budget_ramp_until=args.budget_ramp_until,
        canonical_weight=args.canonical_weight,
        line_policy_weight=args.line_policy_weight,
        reference_line_share=args.reference_line_share,
        reference_line_until=args.reference_line_until,
        eval_every=args.eval_every,
        eval_depths=tuple(int(item) for item in args.eval_depths.split(",")),
        max_vram_gb=args.max_vram_gb or None,
        cuda_graphs=args.cuda_graphs,
        seed=args.seed,
        device=args.device,
    )
    train_puzzles(config, model_config, resume=args.resume)
    return 0


def _evaluate_puzzle_corpus(args: argparse.Namespace) -> int:
    from wombat.eval.puzzle_corpus_eval import evaluate_corpus_curve

    report = evaluate_corpus_curve(
        args.checkpoint,
        args.split,
        (int(item) for item in args.depths.split(",")),
        output=args.output,
        limit=args.limit,
        batch_size=args.batch_size,
        device_name=args.device,
        max_vram_gb=args.max_vram_gb or None,
    )
    summary = [
        {key: row[key] for key in ("reasoning_steps", "accuracy", "low", "high")}
        for row in report["curve"]  # type: ignore[attr-defined]
    ]
    print(json.dumps(summary, indent=2))
    return 0


def _evaluate_puzzle_depth(args: argparse.Namespace) -> int:
    from wombat.eval.puzzle_corpus_eval import evaluate_puzzle_depth

    report = evaluate_puzzle_depth(
        args.checkpoint,
        args.split,
        (int(item) for item in args.depths.split(",")),
        output=args.output,
        per_group=args.per_group,
        seed=args.seed,
        batch_size=args.batch_size,
        device_name=args.device,
        max_vram_gb=args.max_vram_gb or None,
        select=args.select,
        beta=args.select_beta,
    )
    groups = report["groups"]
    assert isinstance(groups, dict)
    summary = {
        name: {
            depth: [round(row["accuracy"], 4), round(row["versus_t8"]["difference"], 4)]
            for depth, row in group["depths"].items()
        }
        for name, group in groups.items()
    }
    print("accuracy and paired difference against T8, per solver-moves-remaining group")
    print(json.dumps(summary, indent=2))
    return 0


def _evaluate_strength(args: argparse.Namespace) -> int:
    from wombat.eval.strength import evaluate_strength
    from wombat.settings import load_executables

    stockfish = args.stockfish or load_executables()["stockfish"]
    if stockfish is None:
        raise SystemExit("Stockfish is not configured")
    report = evaluate_strength(
        args.checkpoint,
        games=args.games,
        puzzles=args.puzzles,
        output=args.output,
        depths=tuple(int(item) for item in args.depths.split(",")),
        stockfish=Path(stockfish),
        positions=args.positions,
        puzzle_count=args.puzzle_count,
        nodes=args.nodes,
        engines=args.engines,
        device_name=args.device,
        select=args.select,
        beta=_select_beta(args),
        top_k=args.select_top_k,
        line_plies=args.line_plies,
    )
    keys = ("reasoning_steps", "lichess_accuracy", "good_rate", "blunder_rate", "puzzle_rating")
    print(json.dumps([{key: row[key] for key in keys} for row in report["curve"]], indent=2))  # type: ignore[attr-defined]
    return 0


def _evaluate_reply(args: argparse.Namespace) -> int:
    from wombat.eval.reply import evaluate_reply
    from wombat.settings import load_executables

    stockfish = args.stockfish or load_executables()["stockfish"]
    if stockfish is None:
        raise SystemExit("Stockfish is not configured")
    report = evaluate_reply(
        args.checkpoint,
        split=args.split,
        engine_command=[str(stockfish)],
        positions=args.positions,
        nodes=args.nodes,
        steps=args.steps,
        engines=args.engines,
        cache_dir=args.cache_dir,
        device_name=args.device,
        max_vram_gb=args.max_vram_gb or None,
        seed=args.seed,
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


def _play_stockfish(args: argparse.Namespace) -> int:
    from wombat.eval.games import play_stockfish
    from wombat.external.chessbench import EXTERNAL_PLAYERS
    from wombat.settings import load_executables

    stockfish = args.stockfish or load_executables()["stockfish"]
    if stockfish is None:
        raise SystemExit("Stockfish is not configured")
    if (args.external is None) != (args.external_weights is None):
        raise SystemExit("--external and --external-weights go together")
    reports = play_stockfish(
        args.checkpoint,
        [str(stockfish)],
        levels=args.elo,
        steps=args.steps,
        pairs=args.pairs,
        movetime=None if args.nodes else args.movetime,
        nodes=args.nodes,
        max_plies=args.max_plies,
        output=args.output,
        split_dir=args.openings,
        seed=args.seed,
        draw_guard=args.draw_guard,
        select=args.select,
        beta=args.select_beta,
        external=args.external,
        external_weights=args.external_weights,
        device_name=args.device,
    )
    name = EXTERNAL_PLAYERS[args.external].name if args.external else f"Wombat T{args.steps}"
    for report in reports:
        opponent = report["opponent"]
        assert isinstance(opponent, dict)
        line = (
            f"{opponent['label']}: +{report['wins']} ={report['draws']} -{report['losses']}"
            f" score {report['score']:.3f}"
        )
        if "elo_difference" in report:
            line += (
                f" Elo diff {report['elo_difference']:+.0f}"
                f" [{report['elo_low']:+.0f}, {report['elo_high']:+.0f}]"
                f" pentanomial {report['pentanomial']}"
            )
        if "wombat_elo" in report:
            line += f" {name} ~ {report['wombat_elo']:.0f}"
        print(line)
    print(f"games and match.json in {args.output}")
    return 0


def _compare_strength(args: argparse.Namespace) -> int:
    from wombat.eval.strength import compare_strength

    result = compare_strength(args.first, args.first_depth, args.second, args.second_depth)
    print(json.dumps(result, indent=2))
    return 0


def _compare_puzzle_runs(args: argparse.Namespace) -> int:
    from wombat.eval.puzzle_corpus_eval import compare_correctness

    print(
        json.dumps(
            compare_correctness(args.first, args.first_depth, args.second, args.second_depth),
            indent=2,
        )
    )
    return 0


def _prepare_game_errors(args: argparse.Namespace) -> int:
    from wombat.eval.game_errors import prepare_error_set

    manifest = prepare_error_set(args.source, args.output, args.training)
    print(json.dumps({key: value for key, value in manifest.items() if key != "sha256"}, indent=2))
    return 0


def _evaluate_game_errors(args: argparse.Namespace) -> int:
    from wombat.eval.game_errors import THRESHOLDS, evaluate_game_errors, rate_name

    if (args.checkpoint is None) == (args.external is None):
        raise SystemExit("give exactly one of --checkpoint and --external")
    if (args.external is None) != (args.external_weights is None):
        raise SystemExit("--external and --external-weights go together")
    report = evaluate_game_errors(
        args.error_set,
        args.output,
        checkpoint=args.checkpoint,
        depths=(int(item) for item in args.depths.split(",")),
        external=args.external,
        external_weights=args.external_weights,
        batch_size=args.batch_size,
        device_name=args.device,
        max_vram_gb=args.max_vram_gb or None,
        limit=args.limit,
    )
    print(f"{report['positions']} positions, {report['live']} live; live rates in percent")
    results = report["results"]
    assert isinstance(results, dict)
    for label, result in results.items():
        live = result["live"]
        rates = " ".join(
            f"{rate_name(threshold)} {100 * live[rate_name(threshold)]['rate']:.2f}"
            for threshold in THRESHOLDS
        )
        print(
            f"{label}: mean loss {100 * live['mean_loss']['mean']:.2f} {rates}"
            f" accuracy {live['lichess_accuracy']['mean']:.2f}"
        )
    return 0


def _compare_game_errors(args: argparse.Namespace) -> int:
    from wombat.eval.game_errors import compare_game_errors

    result = compare_game_errors(args.first, args.second, args.depth, args.second_depth)
    print(json.dumps(result, indent=2))
    return 0


def _export_external_benchmark(args: argparse.Namespace) -> int:
    from wombat.eval.external_strength import export_strength_fens
    from wombat.eval.full_puzzles import export_puzzle_csv

    summary = export_puzzle_csv(
        args.split, args.source, args.output / "puzzles.csv", count=args.puzzles, seed=args.seed
    )
    summary["strength_positions"] = export_strength_fens(
        args.games, args.output / "positions.txt", args.positions
    )
    print(json.dumps(summary, indent=2))
    return 0


def _evaluate_full_puzzles(args: argparse.Namespace) -> int:
    from wombat.eval.full_puzzles import evaluate_full_puzzles, score_external_puzzles

    if (args.checkpoint is None) == (args.moves is None):
        raise SystemExit("give exactly one of --checkpoint and --moves")
    if args.moves is not None:
        report = score_external_puzzles(args.moves, args.puzzles, output=args.output)
    else:
        report = evaluate_full_puzzles(
            args.checkpoint,
            args.puzzles,
            (int(item) for item in args.depths.split(",")),
            output=args.output,
            batch_size=args.batch_size,
            device_name=args.device,
            max_vram_gb=args.max_vram_gb or None,
            select=args.select,
            beta=_select_beta(args),
            top_k=args.select_top_k,
            line_plies=args.line_plies,
        )
    results = report["results"]
    assert isinstance(results, dict)
    keys = ("accuracy", "low", "high", "exact_accuracy", "puzzle_rating")
    print(json.dumps({name: {key: row[key] for key in keys} for name, row in results.items()}))
    return 0


def _evaluate_stopping(args: argparse.Namespace) -> int:
    from wombat.eval.stopping import evaluate_stopping, held_out_summary
    from wombat.settings import load_executables

    stockfish = args.stockfish or load_executables()["stockfish"]
    if stockfish is None:
        raise SystemExit("Stockfish is not configured")
    report = evaluate_stopping(
        args.checkpoint,
        puzzles=args.puzzles,
        games=args.games,
        output=args.output,
        stockfish=Path(stockfish),
        positions=args.positions,
        nodes=args.nodes,
        engines=args.engines,
        cache_dir=args.cache_dir,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        device_name=args.device,
        max_vram_gb=args.max_vram_gb or None,
    )
    print(json.dumps(held_out_summary(report), indent=2))
    return 0


def _evaluate_warm_start(args: argparse.Namespace) -> int:
    from wombat.eval.warm_start import evaluate_warm_start

    report = evaluate_warm_start(
        args.checkpoint,
        args.puzzles,
        (int(item) for item in args.depths.split(",")),
        (float(item) for item in args.alphas.split(",")),
        output=args.output,
        batch_size=args.batch_size,
        device_name=args.device,
        max_vram_gb=args.max_vram_gb or None,
    )
    results = report["results"]
    assert isinstance(results, dict)
    summary = {
        depth: {
            name: [row["accuracy"], row["versus_fresh"]["low"], row["versus_fresh"]["high"]]
            for name, row in rows.items()
        }
        for depth, rows in results.items()
    }
    print("accuracy and the paired interval against the fresh start")
    print(json.dumps(summary, indent=2))
    return 0


def _score_surprise(args: argparse.Namespace) -> int:
    from wombat.surprise import score_surprise

    manifest = score_surprise(
        args.checkpoint,
        args.labels,
        args.output,
        corpus=args.corpus,
        puzzle_corpus=args.puzzle_corpus,
        depth=args.depth,
        batch_size=args.batch_size,
        device_name=args.device,
        max_vram_gb=args.max_vram_gb or None,
    )
    print(json.dumps(manifest, indent=2))
    return 0


def _evaluate_external_strength(args: argparse.Namespace) -> int:
    from wombat.eval.external_strength import evaluate_external_strength
    from wombat.settings import load_executables

    stockfish = args.stockfish or load_executables()["stockfish"]
    if stockfish is None:
        raise SystemExit("Stockfish is not configured")
    report = evaluate_external_strength(
        args.moves,
        games=args.games,
        output=args.output,
        stockfish=Path(stockfish),
        positions=args.positions,
        nodes=args.nodes,
        engines=args.engines,
    )
    print(json.dumps(report, indent=2))
    return 0


def _explain_move(args: argparse.Namespace) -> int:
    import chess
    import torch

    from wombat.eval.puzzle_corpus_eval import load_checkpoint
    from wombat.interpretability.explain import explain_move, format_explanation

    model = load_checkpoint(args.checkpoint, torch.device("cpu"))
    result = explain_move(
        model,
        chess.Board(args.fen),
        chess.Move.from_uci(args.move),
        compare=chess.Move.from_uci(args.compare) if args.compare else None,
        steps=args.steps,
        depths=tuple(int(item) for item in args.depths.split(",")),
    )
    print(json.dumps(result, indent=2) if args.json else format_explanation(result))
    return 0


def _probe_steps(args: argparse.Namespace) -> int:
    from wombat.interpretability.steps import format_summary, probe_steps

    labels = args.labels if args.labels is not None and args.labels.exists() else None
    report = probe_steps(
        args.checkpoint,
        args.output,
        games=args.games,
        puzzles=args.puzzles,
        labelled_games=args.labelled_games,
        labels=labels,
        strength=args.strength,
        positions=args.positions,
        depths=tuple(int(item) for item in args.depths.split(",")),
        early=args.early,
        late=args.late,
        device_name=args.device,
        max_vram_gb=args.max_vram_gb or None,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    print(format_summary(report))
    print(f"report: {args.output / 'report.json'}")
    return 0


def _annotate_continuations(args: argparse.Namespace) -> int:
    from wombat.data.continuations import ContinuationConfig, annotate_continuations

    report = annotate_continuations(
        ContinuationConfig(
            source=args.source,
            destination=args.output,
            stockfish=args.stockfish,
            nodes=args.nodes,
            multipv=args.multipv,
            plies=args.plies,
            threads=args.threads,
            hash_mb=args.hash_mb,
            limit=args.limit,
            field_name=args.field,
        )
    )
    print(json.dumps(report, indent=2))
    return 0


def _train_baseline(args: argparse.Namespace) -> int:
    from wombat.training import TrainingConfig, train_supervised
    from wombat.model import NetworkConfig

    config = TrainingConfig(
        dataset=str(args.dataset),
        output=str(args.output),
        examples=args.examples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        move_value_weight=args.move_value_weight,
        continuation_weight=args.continuation_weight,
        policy_temperature=args.policy_temperature,
        target_kind=args.target_kind,
        reasoning_steps=args.reasoning_steps,
        seed=args.seed,
        device=args.device,
    )
    report = train_supervised(
        config,
        NetworkConfig(
            prelude_blocks=args.prelude_blocks,
            recurrent_blocks=args.recurrent_blocks,
            use_update_gate=args.use_update_gate,
        ),
    )
    print(
        json.dumps(
            {
                "checkpoint": report["checkpoint"],
                "parameters": report["parameters"],
                "final": report["metrics"][-1],
            },
            indent=2,
        )
    )
    return 0


def _evaluate_checkpoint(args: argparse.Namespace) -> int:
    from wombat.training import evaluate_checkpoint

    result = evaluate_checkpoint(
        args.checkpoint,
        args.dataset,
        reasoning_steps=args.reasoning_steps,
        batch_size=args.batch_size,
        device_name=args.device,
    )
    print(json.dumps(result, indent=2))
    return 0


def _train_recurrent(args: argparse.Namespace) -> int:
    from wombat.training import RecurrentTrainingConfig, train_recurrent

    report = train_recurrent(
        RecurrentTrainingConfig(
            dataset=str(args.dataset),
            output=str(args.output),
            initial_checkpoint=str(args.initial_checkpoint),
            examples=args.examples,
            epochs=args.epochs,
            batch_size=args.batch_size,
            effective_batch_size=args.effective_batch_size,
            learning_rate=args.learning_rate,
            move_value_weight=args.move_value_weight,
            continuation_weight=args.continuation_weight,
            policy_temperature=args.policy_temperature,
            initial_max_steps=args.initial_max_steps,
            final_max_steps=args.final_max_steps,
            minimum_steps=args.minimum_steps,
            proposal_auxiliary_weight=args.proposal_auxiliary_weight,
            disagreement_weight=args.disagreement_weight,
            early_auxiliary_weight=args.early_auxiliary_weight,
            progressive_weight=args.progressive_weight,
            max_gradient_steps=args.max_gradient_steps,
            improvement_weight=args.improvement_weight,
            improvement_margin=args.improvement_margin,
            improvement_fraction=args.improvement_fraction,
            improvement_eligibility_epsilon=args.improvement_eligibility_epsilon,
            contraction_weight=args.contraction_weight,
            contraction_ratio=args.contraction_ratio,
            use_scratchpad_backup=args.use_scratchpad_backup,
            use_future_workspaces=args.use_future_workspaces,
            future_workspace_weight=args.future_workspace_weight,
            future_board_weight=args.future_board_weight,
            future_value_weight=args.future_value_weight,
            max_residual_update=args.max_residual_update,
            reset_recurrent_to_identity=args.reset_recurrent_to_identity,
            seed=args.seed,
            device=args.device,
        )
    )
    print(json.dumps({"checkpoint": report["checkpoint"], "final": report["metrics"][-1]}, indent=2))
    return 0


def _evaluate_puzzles(args: argparse.Namespace) -> int:
    from wombat.eval.puzzles import evaluate_puzzle_curve, write_puzzle_curve

    depths = tuple(int(item) for item in args.depths.split(","))
    report = evaluate_puzzle_curve(
        args.checkpoint,
        args.manifest,
        depths,
        batch_size=args.batch_size,
        device_name=args.device,
        limit=args.limit,
    )
    write_puzzle_curve(report, args.output)
    print(json.dumps(report, indent=2))
    return 0


def _evaluate_quality(args: argparse.Namespace) -> int:
    from wombat.eval.quality_curve import evaluate_quality_curve, write_quality_curve

    report = evaluate_quality_curve(
        args.checkpoint,
        args.dataset,
        tuple(int(item) for item in args.depths.split(",")),
        value_weight=args.value_weight,
        move_value_weight=args.move_value_weight,
        continuation_weight=args.continuation_weight,
        policy_temperature=args.policy_temperature,
        batch_size=args.batch_size,
        device_name=args.device,
        limit=args.limit,
    )
    write_quality_curve(report, args.output)
    print(json.dumps(report, indent=2))
    return 0


def _add_training_arguments(parser: argparse.ArgumentParser, *, corpus: str, steps: int) -> None:
    parser.add_argument("--corpus", type=Path, default=Path(corpus))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=steps)
    parser.add_argument(
        "--stop-after", type=int, help="stop at this step; the schedule still spans --steps"
    )
    parser.add_argument(
        "--cooldown-steps",
        type=int,
        default=0,
        help="decay the learning rate to 0 over this many steps before the last one (1 - sqrt)",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--warmup-steps", type=int, default=1_000)
    parser.add_argument("--max-depth", type=int, default=16)
    parser.add_argument("--max-gradient-steps", type=int, default=8)
    parser.add_argument("--fixed-depth", type=int)
    parser.add_argument(
        "--progressive-share",
        type=float,
        default=0.0,
        help="with --fixed-depth, share of batches that continue a random no-gradient prefix",
    )
    parser.add_argument("--min-gradient-steps", type=int, default=1)
    parser.add_argument(
        "--random-prefix",
        type=int,
        default=0,
        help="up to this many extra no-gradient steps before every batch's graded tail",
    )
    parser.add_argument(
        "--random-prefix-from",
        type=int,
        default=0,
        help="training step from which --random-prefix applies; before it every batch runs"
        " its exact depth",
    )
    parser.add_argument(
        "--depth-distribution",
        choices=("fixed", "lognormal"),
        default="fixed",
        help=(
            "lognormal draws each batch's total steps from a log-normal Poisson, graded at the"
            " end, in place of --fixed-depth and --random-prefix"
        ),
    )
    parser.add_argument(
        "--depth-mean",
        type=float,
        default=16.0,
        help="mean Poisson rate of a lognormal depth, which adds 1",
    )
    parser.add_argument("--depth-max", type=int, default=48, help="longest lognormal depth")
    parser.add_argument(
        "--initial-noise",
        type=float,
        default=0.0,
        help="std of Gaussian noise on the square tokens of H0, in training only",
    )
    parser.add_argument("--eval-every", type=int, default=1_000)
    parser.add_argument("--eval-depths", default="0,1,2,4,8,16,32")
    parser.add_argument("--snapshot-every", type=int, default=0)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--mlp-width", type=int, default=1024)
    parser.add_argument("--prelude-blocks", type=int, default=1)
    parser.add_argument("--recurrent-blocks", type=int, default=2)
    parser.add_argument("--latent-tokens", type=int, default=16)
    parser.add_argument("--stability", choices=("normalized", "capped"), default="normalized")
    parser.add_argument(
        "--move-consequences",
        action="store_true",
        help=(
            "add each move's exchange, the reply's best capture, a mate-in-one reply and check to"
            " its logit and move value, weighted from its origin and destination tokens (v20)"
        ),
    )
    parser.add_argument(
        "--relative-attention",
        action="store_true",
        help=(
            "learned key and value vectors per king step offset between recurrent square tokens"
            " (Shaw relative positions); needs --local-attention"
        ),
    )
    parser.add_argument(
        "--state-attention-bias",
        action="store_true",
        help=(
            "square pair attention logits generated from H's square tokens at every recurrent"
            " step, per block through templates shared by all blocks (smolgen); needs the square"
            " bias"
        ),
    )
    parser.add_argument(
        "--state-attention-bias-width",
        type=int,
        default=32,
        help="compressed board width and templates per head of --state-attention-bias",
    )
    parser.add_argument(
        "--qk-norm",
        action="store_true",
        help="RMS-normalize each head's queries and keys, bounding attention logits",
    )
    parser.add_argument(
        "--value-bins",
        type=int,
        default=0,
        help=(
            "predict the position value and every move value as this many win-chance bins"
            " trained with the HL-Gauss loss (ChessBench); 0 keeps the tanh value heads"
        ),
    )
    parser.add_argument(
        "--attack-pairs",
        action="store_true",
        help=(
            "learned attention logits between the squares of two pieces where one attacks or"
            " defends the other, by both pieces; local attention also unmasks those pairs (v36)"
        ),
    )
    parser.add_argument(
        "--phase-heads",
        type=int,
        default=0,
        help=(
            "copies of the last policy, move value and value layers, one per phase bucket by"
            " non-pawn pieces (4: 0-3, 4-7, 8-11, 12+), starting equal; 0 keeps one (v37)"
        ),
    )
    parser.add_argument(
        "--king-relative",
        action="store_true",
        help=(
            "add learned embeddings of each square's file and rank offset from its own side's"
            " king and from the enemy king to its token, from zero (v38)"
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--init-from",
        type=Path,
        help=(
            "checkpoint whose raw and averaged weights start the run with a fresh optimizer, step"
            " count and schedule; the model flags must match it, and --resume of this run's own"
            " checkpoint takes precedence"
        ),
    )
    parser.add_argument(
        "--no-cuda-graphs", dest="cuda_graphs", action="store_false", help="launch kernels eagerly"
    )
    _add_gpu_sharing(parser)


def _add_workspace_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--core",
        choices=("recurrent", "workspace"),
        default="recurrent",
        help=(
            "workspace: a writable board memory and a small workspace that imagine lines of the"
            " model's own moves from the present position, each ply's memory pulled toward the"
            " encoding of the position its move produced (v34)"
        ),
    )
    parser.add_argument("--workspace-tokens", type=int, default=12, help="workspace tokens")
    parser.add_argument("--steps-per-ply", type=int, default=2, help="steps per imagined ply")
    parser.add_argument(
        "--line-plies",
        type=int,
        default=2,
        help="plies per line when inference and validation map T steps to lines",
    )
    parser.add_argument(
        "--readout-blocks",
        type=int,
        default=1,
        help="blocks in which the board tokens read the workspace before the heads",
    )
    for name, low, high in (("lines", 1, 3), ("line-plies", 1, 3)):
        parser.add_argument(
            f"--{name}-min", type=int, default=low, help=f"fewest {name} a batch draws"
        )
        parser.add_argument(
            f"--{name}-max", type=int, default=high, help=f"most {name} a batch draws"
        )
        parser.add_argument(
            f"--{name}-max-final",
            type=int,
            help=f"--{name}-max reached at the end of the budget ramp; unset keeps it fixed",
        )
    parser.add_argument("--budget-ramp-from", type=int, default=0, help="step the ramp starts")
    parser.add_argument("--budget-ramp-until", type=int, default=0, help="step the ramp ends")
    parser.add_argument(
        "--canonical-weight",
        type=float,
        default=1.0,
        help="weight of the memory's squared error from the encoding of each ply's position",
    )
    parser.add_argument(
        "--line-policy-weight",
        type=float,
        default=0.25,
        help="weight of the line policy's cross-entropy against reference moves",
    )
    parser.add_argument(
        "--reference-line-share",
        type=float,
        default=1.0,
        help="share of batches whose lines follow the reference after a matching first move,"
        " falling linearly to 0 at --reference-line-until",
    )
    parser.add_argument(
        "--reference-line-until",
        type=int,
        help="step at which reference following ends; default 30%% of the last step",
    )


def _add_line_plies(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--line-plies",
        type=int,
        help="workspace core: plies per line the depths' steps run in; default the checkpoint's",
    )


def _add_gpu_sharing(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--max-vram-gb", type=float, default=3.5, help="VRAM ceiling in GiB; 0 disables the cap"
    )


class _GivenBeta(argparse.Action):
    """Stores ``--select-beta`` and notes it was given, so child-mix can keep its own default."""

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: object,
        option_string: str | None = None,
    ) -> None:
        setattr(namespace, self.dest, values)
        namespace.select_beta_given = True


def _select_beta(args: argparse.Namespace) -> float:
    from wombat.selection import default_beta

    return float(args.select_beta) if args.select_beta_given else default_beta(args.select)


def _add_move_selection(parser: argparse.ArgumentParser, *, children: bool = False) -> None:
    from wombat.selection import CHILD_MIX_BETA, CHILD_MODES, CHILD_TOP_K, MIX_BETA, SELECT_MODES

    help_text = (
        "choose moves by policy logits, per-move values, or policy log probability plus "
        "beta times value; value and mix need a checkpoint trained with move values"
    )
    if children:
        help_text += (
            "; child and child-mix play each top-k policy move and rank it by the model's "
            "value of the resulting position at the same depth, alone or plus the policy log "
            "probability"
        )
    parser.add_argument(
        "--select",
        choices=SELECT_MODES + CHILD_MODES if children else SELECT_MODES,
        default="policy",
        help=help_text,
    )
    beta_help = (
        "weight of the -1..1 move value in mix; the default counts it once more at the "
        "scale of the policy target"
    )
    if children:
        beta_help += (
            f"; in child-mix the weight of the negated child value, default {CHILD_MIX_BETA:g}"
        )
    parser.add_argument(
        "--select-beta", type=float, default=MIX_BETA, action=_GivenBeta, help=beta_help
    )
    parser.set_defaults(select_beta_given=False)
    if children:
        parser.add_argument(
            "--select-top-k",
            type=int,
            default=CHILD_TOP_K,
            help="policy moves whose children the child modes evaluate",
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="wombat")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("doctor", help="report runtime and external-tool readiness")
    eval_smoke = subparsers.add_parser(
        "eval-smoke", help="evaluate one untrained checkpoint at all v1 reasoning depths"
    )
    eval_smoke.add_argument("--output", type=Path, default=Path("runs/smoke"))
    eval_smoke.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    eval_smoke.add_argument("--seed", type=int, default=0)
    regret_smoke = subparsers.add_parser(
        "regret-smoke", help="verify fixed-node Stockfish scoring and cache semantics"
    )
    regret_smoke.add_argument("--output", type=Path, default=Path("runs/regret-smoke"))
    regret_smoke.add_argument("--nodes", type=int, default=256)
    regret_curve = subparsers.add_parser(
        "evaluate-regret", help="measure fixed-node Stockfish regret across recurrent depths"
    )
    regret_curve.add_argument("--checkpoint", type=Path, required=True)
    regret_curve.add_argument(
        "--manifest", type=Path, default=Path("data/eval/lichess-puzzles-50k.json")
    )
    regret_curve.add_argument("--depths", default="1,2,4,8,16")
    regret_curve.add_argument("--limit", type=int, default=200)
    regret_curve.add_argument("--nodes", type=int, default=20_000)
    regret_curve.add_argument("--stockfish", type=Path)
    regret_curve.add_argument(
        "--cache-dir", type=Path, default=Path("data/cache/stockfish-regret")
    )
    regret_curve.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    regret_curve.add_argument(
        "--output", type=Path, default=Path("runs/regret-curve/report.json")
    )
    match_smoke = subparsers.add_parser(
        "match-smoke", help="run a two-game color-swapped Cute Chess UCI match"
    )
    match_smoke.add_argument("--output", type=Path, default=Path("matches/smoke"))
    match = subparsers.add_parser(
        "evaluate-match",
        help="play one checkpoint against itself at two thinking budgets in Cute Chess",
    )
    match.add_argument("--checkpoint", type=Path, required=True)
    match.add_argument("--first-steps", type=int, default=1, help="T of the reference engine")
    match.add_argument("--second-steps", type=int, default=64, help="T of the engine scored")
    match.add_argument(
        "--pairs", type=int, default=100, help="openings, each played with both colours"
    )
    match.add_argument("--openings", type=Path, default=Path("data/processed/games-v1/test"))
    match.add_argument("--seed", type=int, default=0)
    match.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    match.add_argument("--max-moves", type=int, default=150, help="full moves before a draw")
    match.add_argument("--concurrency", type=int, default=1, help="games played at once")
    match.add_argument("--timeout", type=float, help="seconds before the match is abandoned")
    match.add_argument("--output", type=Path, default=Path("runs/paired-compute-match"))
    trace_smoke = subparsers.add_parser(
        "trace-smoke", help="capture H0..Hn, policy/value, scratchpad, and attention artifacts"
    )
    trace_smoke.add_argument("--output", type=Path, default=Path("runs/trace-smoke"))
    trace_smoke.add_argument("--steps", type=int, default=8)
    trace_smoke.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    build_puzzles = subparsers.add_parser(
        "build-puzzles", help="create a deterministic frozen Lichess puzzle manifest"
    )
    build_puzzles.add_argument(
        "--source",
        type=Path,
        default=Path("data/raw/lichess/lichess_db_puzzle.csv.zst"),
    )
    build_puzzles.add_argument(
        "--output", type=Path, default=Path("data/eval/lichess-puzzles-50k.json")
    )
    build_puzzles.add_argument("--limit", type=int, default=50_000)
    build_corpus = subparsers.add_parser(
        "build-corpus", help="stream a compact leak-resistant Lichess evaluation corpus"
    )
    build_corpus.add_argument(
        "--source",
        type=Path,
        default=Path("data/raw/lichess/lichess_db_eval.jsonl.zst"),
    )
    build_corpus.add_argument(
        "--output", type=Path, default=Path("data/processed/debug-10k")
    )
    build_corpus.add_argument("--limit", type=int, default=10_000)
    build_corpus.add_argument(
        "--puzzle-manifest",
        type=Path,
        default=Path("data/eval/lichess-puzzles-50k.json"),
    )
    puzzle_corpus = subparsers.add_parser(
        "build-puzzle-corpus", help="encode every solver position of every puzzle line"
    )
    puzzle_corpus.add_argument(
        "--source", type=Path, default=Path("data/raw/lichess/lichess_db_puzzle.csv.zst")
    )
    puzzle_corpus.add_argument("--output", type=Path, default=Path("data/processed/puzzles-v1"))
    puzzle_corpus.add_argument(
        "--puzzle-manifest", type=Path, default=Path("data/eval/lichess-puzzles-50k.json")
    )
    puzzle_corpus.add_argument("--limit", type=int, help="maximum puzzles to read")
    puzzle_corpus.add_argument("--workers", type=int, default=4)
    game_corpus = subparsers.add_parser(
        "build-game-corpus", help="encode deep multi-candidate Lichess evaluations"
    )
    game_corpus.add_argument(
        "--source", type=Path, default=Path("data/raw/lichess/lichess_db_eval.jsonl.zst")
    )
    game_corpus.add_argument("--output", type=Path, default=Path("data/processed/games-v1"))
    game_corpus.add_argument(
        "--puzzle-manifest", type=Path, default=Path("data/eval/lichess-puzzles-50k.json")
    )
    game_corpus.add_argument("--limit", type=int, default=10_300_000)
    game_corpus.add_argument(
        "--skip-records",
        type=int,
        default=0,
        help="leave out the archive's first records and every position in them; "
        "give an earlier corpus's records_read to build a disjoint one",
    )
    game_corpus.add_argument(
        "--exclude-splits",
        type=Path,
        nargs="+",
        default=[],
        help="split directories with fens.npy whose positions are left out",
    )
    game_corpus.add_argument("--workers", type=int, default=4)
    chessbench = subparsers.add_parser(
        "import-chessbench",
        help="turn ChessBench action-value bags into a game corpus with every move labelled",
    )
    chessbench.add_argument(
        "--shards",
        default="",
        help="train shards to download, such as 0-2147; a position is complete only once "
        "every shard is in",
    )
    chessbench.add_argument(
        "--bags", type=Path, nargs="+", default=[], help="local action-value bags to import"
    )
    chessbench.add_argument(
        "--fraction",
        type=float,
        default=1.0,
        help="share of FENs kept, chosen by hash and the same in every shard",
    )
    chessbench.add_argument("--output", type=Path, default=Path("data/processed/chessbench"))
    chessbench.add_argument(
        "--work-dir",
        type=Path,
        default=Path("data/raw/chessbench"),
        help="where each shard is downloaded before it is read and deleted",
    )
    chessbench.add_argument(
        "--puzzle-manifest", type=Path, default=Path("data/eval/lichess-puzzles-50k.json")
    )
    chessbench.add_argument(
        "--exclude-splits",
        type=Path,
        nargs="*",
        default=[
            Path(f"data/processed/{corpus}/{split}")
            for corpus in ("games-v1", "games-v2", "puzzles-v1")
            for split in ("validation", "test")
        ],
        help="split directories with fens.npy whose positions are left out",
    )
    chessbench.add_argument(
        "--exclude-puzzles",
        type=Path,
        nargs="*",
        default=[Path("data/eval/external-benchmark/puzzles.csv")],
        help="puzzle CSVs whose every line position is left out",
    )
    chessbench.add_argument(
        "--exclude-positions",
        type=Path,
        nargs="*",
        default=[Path("data/eval/external-benchmark/positions.txt")],
        help="files of one FEN per line to leave out",
    )
    chessbench.add_argument(
        "--exclusion-keys",
        type=Path,
        help="excluded_keys.npy saved by an earlier import, used instead of the sets above",
    )
    chessbench.add_argument("--workers", type=int, default=4)
    chessbench.add_argument(
        "--partitions", type=int, default=64, help="FEN key ranges the join encodes one by one"
    )
    chessbench.add_argument(
        "--no-join",
        dest="join",
        action="store_false",
        help="only import shards; join later, once every shard is in",
    )
    chessbench.add_argument(
        "--delete-records",
        action="store_true",
        help="delete the imported records once the join has encoded them, to leave disk for "
        "the output; joining again then needs every shard downloaded again",
    )
    label_moves = subparsers.add_parser(
        "label-moves", help="score every legal move of game positions with Stockfish"
    )
    label_moves.add_argument("--split", type=Path, default=Path("data/processed/games-v1/train"))
    label_moves.add_argument(
        "--output", type=Path, default=Path("data/processed/games-v1/train-labels")
    )
    label_moves.add_argument(
        "--count", type=int, help="positions to label; every position not excluded by default"
    )
    label_moves.add_argument("--nodes", type=int, default=50_000)
    label_moves.add_argument("--engines", type=int, default=12)
    label_moves.add_argument("--stockfish", type=Path)
    label_moves.add_argument(
        "--exclude",
        type=Path,
        nargs="+",
        default=[],
        help="label directories over the same split whose positions are skipped",
    )
    label_moves.add_argument(
        "--reverse",
        action="store_true",
        help="work from the last chunk down, to share one output with a forward run",
    )
    merge_move_labels = subparsers.add_parser(
        "merge-move-labels", help="combine label directories over one split into one"
    )
    merge_move_labels.add_argument("--inputs", type=Path, nargs="+", required=True)
    merge_move_labels.add_argument("--output", type=Path, required=True)
    merge_move_labels.add_argument(
        "--allow-mixed-nodes",
        action="store_true",
        help="merge inputs searched with different node counts",
    )
    train_puzzles = subparsers.add_parser(
        "train-puzzles", help="train a square-readout recurrent core on puzzle lines"
    )
    _add_training_arguments(train_puzzles, corpus="data/processed/puzzles-v1", steps=30_000)
    train_puzzles.set_defaults(
        task="puzzles",
        labels=None,
        extra_train=None,
        temperature=4.0,
        move_value_weight=0.0,
        value_weight=0.0,
        lookahead_weight=0.0,
        reply_readout="squares",
        reply_weight=0.0,
        supervise_every_step=False,
        ema_decay=0.0,
        square_bias=False,
        scratch_boards=0,
        flat_scratch=False,
        refresh_board=False,
        local_attention=False,
        local_neighbourhood="king",
        move_readout="square",
        board_features=False,
        board_features_long_range=False,
        recompute_steps=False,
        puzzle_corpus=None,
        puzzle_share=0.0,
        puzzle_remaining="1,2",
        puzzle_steps_per_move=4,
        puzzle_max_tail=None,
        surprise_weights=None,
        surprise_share=0.5,
        child_value_share=0.0,
        core="recurrent",
        workspace_tokens=12,
        steps_per_ply=2,
        line_plies=2,
        readout_blocks=1,
        lines_min=1,
        lines_max=3,
        line_plies_min=1,
        line_plies_max=3,
        lines_max_final=None,
        line_plies_max_final=None,
        budget_ramp_from=0,
        budget_ramp_until=0,
        canonical_weight=1.0,
        line_policy_weight=0.25,
        reference_line_share=1.0,
        reference_line_until=None,
    )
    train_games = subparsers.add_parser(
        "train-games", help="train on game positions with graded win-chance targets (v7)"
    )
    _add_training_arguments(train_games, corpus="data/processed/games-v1", steps=120_000)
    train_games.add_argument(
        "--labels", type=Path, help="every-move labels directory; trains on that subset"
    )
    train_games.add_argument(
        "--extra-train",
        type=Path,
        nargs=2,
        action="append",
        metavar=("CORPUS", "LABELS"),
        help="another corpus and its labels whose labelled train positions join the pool",
    )
    train_games.add_argument("--temperature", type=float, default=4.0)
    train_games.add_argument("--move-value-weight", type=float, default=0.5)
    train_games.add_argument("--value-weight", type=float, default=0.25)
    train_games.add_argument(
        "--lookahead-weight",
        type=float,
        default=0.25,
        help="reply-square and future-board predictions; 0 also drops the lookahead heads",
    )
    train_games.add_argument(
        "--reply-readout",
        choices=("squares", "pair"),
        default="squares",
        help="pair scores the reply over all 64 x 64 origin and destination square pairs",
    )
    train_games.add_argument(
        "--reply-weight",
        type=float,
        default=0.0,
        help="joint reply-pair cross-entropy; needs --reply-readout pair",
    )
    train_games.add_argument(
        "--supervise-every-step",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="grade every gradient-tail step, later steps weighted more",
    )
    train_games.add_argument(
        "--ema-decay", type=float, default=0.999, help="weight averaging; 0 disables it"
    )
    train_games.add_argument(
        "--square-bias",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="learned 64 x 64 attention bias between square tokens",
    )
    train_games.add_argument(
        "--scratch-boards",
        type=int,
        default=0,
        help="64-token scratch boards added to the recurrent state, read by no head (v8)",
    )
    train_games.add_argument(
        "--flat-scratch",
        action="store_true",
        help="give the scratch tokens no square embedding or square bias (v8-flat control)",
    )
    train_games.add_argument(
        "--refresh-board",
        action="store_true",
        help="reset the square tokens to the board every step; only scratch tokens persist",
    )
    train_games.add_argument(
        "--local-attention",
        action="store_true",
        help="recurrent square tokens attend only to squares one king step away",
    )
    train_games.add_argument(
        "--local-neighbourhood",
        choices=("king", "king_knight", "move_graph", "move_lines"),
        default="king",
        help=(
            "squares local attention reaches; king_knight adds knight jumps, move_graph adds each"
            " position's attacks and pushes in both directions, move_lines lets sliders see"
            " through blockers with a learned bias for the pieces between"
        ),
    )
    train_games.add_argument(
        "--move-readout",
        choices=("square", "from_to"),
        default="square",
        help="from_to adds a score between each move's origin and destination square tokens",
    )
    train_games.add_argument(
        "--board-features",
        action="store_true",
        help="add each square's attackers, cheapest attacker, pins and check to its token",
    )
    train_games.add_argument(
        "--board-features-long-range",
        action="store_true",
        help="with --board-features, also sole defenders, checks, x-rays and material",
    )
    train_games.add_argument(
        "--recompute-steps",
        action="store_true",
        help="recompute each recurrent step in the backward pass to save activation memory",
    )
    train_games.add_argument(
        "--puzzle-corpus",
        type=Path,
        help="puzzle corpus for mixed batches and per-remaining-count puzzle validation (v10)",
    )
    train_games.add_argument(
        "--puzzle-share", type=float, default=0.0, help="share of batches drawn from puzzles"
    )
    train_games.add_argument(
        "--puzzle-remaining",
        default="1,2",
        help="solver-moves-remaining counts a puzzle batch draws one of, with equal chance",
    )
    train_games.add_argument(
        "--puzzle-steps-per-move",
        type=int,
        default=4,
        help="a puzzle batch runs this many steps per remaining move, graded at the end",
    )
    train_games.add_argument(
        "--puzzle-max-tail",
        type=int,
        help="longest puzzle gradient tail; a longer horizon starts without gradients",
    )
    train_games.add_argument(
        "--surprise-weights",
        type=Path,
        help="score-surprise output; batches then draw part of their rows by capped surprise",
    )
    train_games.add_argument(
        "--surprise-share",
        type=float,
        default=0.5,
        help="with --surprise-weights, share of each batch drawn by surprise; the rest uniform",
    )
    train_games.add_argument(
        "--child-value-share",
        type=float,
        default=0.0,
        help=(
            "share of each game batch's rows replaced by the position after one of the row's"
            " labelled moves, drawn uniformly, trained only on its value: 1 minus the move's win"
            " chance for the new side to move (item 24 in docs/plan.md); needs --labels"
        ),
    )
    _add_workspace_arguments(train_games)
    train_games.set_defaults(task="games", snapshot_every=5_000)
    corpus_curve = subparsers.add_parser(
        "evaluate-puzzle-corpus", help="accuracy by depth on a held-out puzzle corpus split"
    )
    corpus_curve.add_argument("--checkpoint", type=Path, required=True)
    corpus_curve.add_argument(
        "--split", type=Path, default=Path("data/processed/puzzles-v1/test")
    )
    corpus_curve.add_argument("--depths", default="0,1,2,4,8,16,32,64")
    corpus_curve.add_argument("--limit", type=int)
    corpus_curve.add_argument("--batch-size", type=int, default=512)
    corpus_curve.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    corpus_curve.add_argument("--output", type=Path, required=True)
    _add_gpu_sharing(corpus_curve)
    puzzle_depth = subparsers.add_parser(
        "evaluate-puzzle-depth",
        help="first-move puzzle accuracy by depth per moves-remaining group, paired with T8 (v10)",
    )
    puzzle_depth.add_argument("--checkpoint", type=Path, required=True)
    puzzle_depth.add_argument("--output", type=Path, required=True)
    puzzle_depth.add_argument("--split", type=Path, default=Path("data/processed/puzzles-v1/test"))
    puzzle_depth.add_argument("--depths", default="1,2,4,8,12,16,24,32")
    puzzle_depth.add_argument(
        "--per-group",
        type=int,
        help="positions per group; by default 3, 4 and 5+ are whole and 1 and 2 are sampled "
        "to the size of the largest of them",
    )
    puzzle_depth.add_argument("--seed", type=int, default=0)
    puzzle_depth.add_argument("--batch-size", type=int, default=512)
    puzzle_depth.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    _add_gpu_sharing(puzzle_depth)
    _add_move_selection(puzzle_depth)
    strength = subparsers.add_parser(
        "evaluate-strength", help="Stockfish-judged move quality and puzzle rating by depth"
    )
    strength.add_argument("--checkpoint", type=Path, required=True)
    strength.add_argument("--output", type=Path, required=True)
    strength.add_argument("--games", type=Path, default=Path("data/processed/games-v1/test"))
    strength.add_argument("--puzzles", type=Path, default=Path("data/processed/puzzles-v1/test"))
    strength.add_argument("--depths", default="0,1,2,4,8,16,32,64")
    strength.add_argument("--positions", type=int, default=2_000)
    strength.add_argument("--puzzle-count", type=int, default=1_000)
    strength.add_argument("--nodes", type=int, default=20_000)
    strength.add_argument("--engines", type=int, default=4)
    strength.add_argument("--stockfish", type=Path)
    strength.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    _add_move_selection(strength, children=True)
    _add_line_plies(strength)
    reply = subparsers.add_parser(
        "evaluate-reply",
        help="reply-head and policy agreement with Stockfish's reply to its best move",
    )
    reply.add_argument("--checkpoint", type=Path, required=True)
    reply.add_argument("--split", type=Path, default=Path("data/processed/games-v1/validation"))
    reply.add_argument("--positions", type=int, default=1_000)
    reply.add_argument("--nodes", type=int, default=200_000)
    reply.add_argument("--steps", type=int, default=8)
    reply.add_argument("--engines", type=int, default=4)
    reply.add_argument("--stockfish", type=Path)
    reply.add_argument("--cache-dir", type=Path, default=Path("runs/stockfish-cache/reply-lines"))
    reply.add_argument("--seed", type=int, default=0)
    reply.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    reply.add_argument(
        "--max-vram-gb", type=float, default=1.0, help="VRAM ceiling in GiB; 0 disables the cap"
    )
    reply.add_argument("--output", type=Path, help="also write the report to this JSON file")
    compare_strength = subparsers.add_parser(
        "compare-strength", help="paired move-quality difference between two strength runs"
    )
    compare_strength.add_argument("--first", type=Path, required=True)
    compare_strength.add_argument("--first-depth", type=int, required=True)
    compare_strength.add_argument("--second", type=Path, required=True)
    compare_strength.add_argument("--second-depth", type=int, required=True)
    compare = subparsers.add_parser(
        "compare-puzzle-runs", help="paired accuracy difference between two corpus evaluations"
    )
    compare.add_argument("--first", type=Path, required=True)
    compare.add_argument("--first-depth", type=int, required=True)
    compare.add_argument("--second", type=Path, required=True)
    compare.add_argument("--second-depth", type=int, required=True)
    prepare_errors = subparsers.add_parser(
        "prepare-game-errors",
        help="copy an imported ChessBench test corpus as the error set, marking training overlap",
    )
    prepare_errors.add_argument(
        "--source", type=Path, required=True, help="an import-chessbench output directory"
    )
    prepare_errors.add_argument("--output", type=Path, default=Path("data/eval/chessbench-test"))
    prepare_errors.add_argument(
        "--training",
        type=Path,
        nargs="*",
        default=[
            Path(f"data/processed/{corpus}/train")
            for corpus in ("games-v1", "games-v2", "puzzles-v1")
        ],
        help="training splits whose positions are left out of the evaluation",
    )
    game_errors = subparsers.add_parser(
        "evaluate-game-errors",
        help="win chance lost per move on held-out game positions with every move labelled",
    )
    game_errors.add_argument("--checkpoint", type=Path)
    game_errors.add_argument(
        "--external", choices=("chessbench9m",), help="score an external player instead"
    )
    game_errors.add_argument("--external-weights", type=Path)
    game_errors.add_argument("--output", type=Path, required=True)
    game_errors.add_argument("--error-set", type=Path, default=Path("data/eval/chessbench-test"))
    game_errors.add_argument("--depths", default="4,8,16")
    game_errors.add_argument(
        "--batch-size", type=int, default=256, help="positions, or the 9M's moves, per pass"
    )
    game_errors.add_argument("--limit", type=int, help="only the first positions, for a smoke run")
    game_errors.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    game_errors.add_argument(
        "--max-vram-gb", type=float, default=3.0, help="VRAM ceiling in GiB; 0 disables the cap"
    )
    compare_errors = subparsers.add_parser(
        "compare-game-errors", help="paired error-rate differences between two error runs"
    )
    compare_errors.add_argument("--first", type=Path, required=True)
    compare_errors.add_argument("--second", type=Path, required=True)
    compare_errors.add_argument(
        "--depth", type=int, help="depth of both runs; an external player's run has none"
    )
    compare_errors.add_argument(
        "--second-depth", type=int, help="the second run's depth when it differs"
    )
    benchmark = subparsers.add_parser(
        "export-external-benchmark",
        help="write held-out puzzles as a ChessBench-format CSV and the strength positions as FENs",
    )
    benchmark.add_argument("--output", type=Path, required=True)
    benchmark.add_argument("--split", type=Path, default=Path("data/processed/puzzles-v1/test"))
    benchmark.add_argument(
        "--source", type=Path, default=Path("data/raw/lichess/lichess_db_puzzle.csv.zst")
    )
    benchmark.add_argument("--games", type=Path, default=Path("data/processed/games-v1/test"))
    benchmark.add_argument("--puzzles", type=int, default=10_000)
    benchmark.add_argument("--seed", type=int, default=0)
    benchmark.add_argument("--positions", type=int, default=2_000)
    full_puzzles = subparsers.add_parser(
        "evaluate-full-puzzles",
        help="whole-puzzle accuracy on a ChessBench-format CSV, for a checkpoint or given moves",
    )
    full_puzzles.add_argument("--puzzles", type=Path, required=True)
    full_puzzles.add_argument("--output", type=Path, required=True)
    full_puzzles.add_argument("--checkpoint", type=Path)
    full_puzzles.add_argument("--moves", type=Path, help="JSON map of solver-position FEN to UCI")
    full_puzzles.add_argument("--depths", default="8,16")
    full_puzzles.add_argument("--batch-size", type=int, default=512)
    full_puzzles.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    full_puzzles.add_argument(
        "--max-vram-gb", type=float, default=1.0, help="VRAM ceiling in GiB; 0 disables the cap"
    )
    _add_move_selection(full_puzzles, children=True)
    _add_line_plies(full_puzzles)
    stopping = subparsers.add_parser(
        "evaluate-stopping",
        help="stopping rules for the recurrent steps on the strength positions and whole puzzles",
    )
    stopping.add_argument("--checkpoint", type=Path, required=True)
    stopping.add_argument("--output", type=Path, required=True)
    stopping.add_argument(
        "--puzzles", type=Path, default=Path("data/eval/external-benchmark/puzzles.csv")
    )
    stopping.add_argument("--games", type=Path, default=Path("data/processed/games-v1/test"))
    stopping.add_argument("--positions", type=int, default=2_000)
    stopping.add_argument("--nodes", type=int, default=20_000)
    stopping.add_argument("--engines", type=int, default=4)
    stopping.add_argument("--stockfish", type=Path)
    stopping.add_argument("--cache-dir", type=Path, default=Path("runs/stockfish-cache"))
    stopping.add_argument("--max-steps", type=int, default=32)
    stopping.add_argument("--batch-size", type=int, default=512)
    stopping.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    stopping.add_argument(
        "--max-vram-gb", type=float, default=1.0, help="VRAM ceiling in GiB; 0 disables the cap"
    )
    warm = subparsers.add_parser(
        "evaluate-warm-start",
        help="whole puzzles with each solver move starting from the previous move's thinking",
    )
    warm.add_argument("--checkpoint", type=Path, required=True)
    warm.add_argument("--output", type=Path, required=True)
    warm.add_argument(
        "--puzzles", type=Path, default=Path("data/eval/external-benchmark/puzzles.csv")
    )
    warm.add_argument("--depths", default="8,16")
    warm.add_argument("--alphas", default="0.25,0.5,1.0")
    warm.add_argument("--batch-size", type=int, default=512)
    warm.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    warm.add_argument(
        "--max-vram-gb", type=float, default=1.0, help="VRAM ceiling in GiB; 0 disables the cap"
    )
    surprise = subparsers.add_parser(
        "score-surprise",
        help="per-position policy KL and value error on labelled games, solution loss on puzzles",
    )
    surprise.add_argument("--checkpoint", type=Path, required=True)
    surprise.add_argument("--labels", type=Path, required=True)
    surprise.add_argument("--output", type=Path, required=True)
    surprise.add_argument("--corpus", type=Path, default=Path("data/processed/games-v1"))
    surprise.add_argument("--puzzle-corpus", type=Path, default=Path("data/processed/puzzles-v1"))
    surprise.add_argument("--depth", type=int, default=8)
    surprise.add_argument("--batch-size", type=int, default=512)
    surprise.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    surprise.add_argument(
        "--max-vram-gb", type=float, default=1.0, help="VRAM ceiling in GiB; 0 disables the cap"
    )
    external = subparsers.add_parser(
        "evaluate-external-strength",
        help="Stockfish-judged quality of given moves on the evaluate-strength positions",
    )
    external.add_argument("--moves", type=Path, required=True, help="JSON map of FEN to UCI")
    external.add_argument("--output", type=Path, required=True)
    external.add_argument("--games", type=Path, default=Path("data/processed/games-v1/test"))
    external.add_argument("--positions", type=int, default=2_000)
    external.add_argument("--nodes", type=int, default=20_000)
    external.add_argument("--engines", type=int, default=4)
    external.add_argument("--stockfish", type=Path)
    continuations = subparsers.add_parser(
        "annotate-continuations", help="add scored MultiPV continuation lines to a corpus"
    )
    continuations.add_argument("--source", type=Path, required=True)
    continuations.add_argument("--output", type=Path, required=True)
    continuations.add_argument("--stockfish", type=Path, required=True)
    continuations.add_argument("--nodes", type=int, default=5_000)
    continuations.add_argument("--multipv", type=int, default=8)
    continuations.add_argument("--plies", type=int, default=8)
    continuations.add_argument("--threads", type=int, default=4)
    continuations.add_argument("--hash-mb", type=int, default=512)
    continuations.add_argument("--limit", type=int)
    continuations.add_argument(
        "--field",
        choices=("continuations", "deep_continuations"),
        default="continuations",
        help="record field to write, allowing paired shallow/deep annotations",
    )
    train_baseline = subparsers.add_parser(
        "train-baseline", help="train the H0 non-recurrent supervised baseline"
    )
    train_baseline.add_argument(
        "--dataset", type=Path, default=Path("data/processed/debug-10k/train.jsonl.zst")
    )
    train_baseline.add_argument("--output", type=Path, default=Path("runs/baseline-smoke"))
    train_baseline.add_argument("--examples", type=int, default=256)
    train_baseline.add_argument("--epochs", type=int, default=2)
    train_baseline.add_argument("--batch-size", type=int, default=64)
    train_baseline.add_argument("--learning-rate", type=float, default=3e-4)
    train_baseline.add_argument("--reasoning-steps", type=int, default=0)
    train_baseline.add_argument("--prelude-blocks", type=int, default=3)
    train_baseline.add_argument("--recurrent-blocks", type=int, default=2)
    train_baseline.add_argument(
        "--use-update-gate", action=argparse.BooleanOptionalAction, default=False
    )
    train_baseline.add_argument("--move-value-weight", type=float, default=0.25)
    train_baseline.add_argument("--continuation-weight", type=float, default=0.25)
    train_baseline.add_argument("--policy-temperature", type=float, default=0.75)
    train_baseline.add_argument("--target-kind", choices=("deep", "shallow"), default="deep")
    train_baseline.add_argument("--seed", type=int, default=0)
    train_baseline.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    evaluate = subparsers.add_parser(
        "evaluate-checkpoint", help="measure a checkpoint on a compact held-out shard"
    )
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--dataset", type=Path, required=True)
    evaluate.add_argument("--reasoning-steps", type=int, default=0)
    evaluate.add_argument("--batch-size", type=int, default=128)
    evaluate.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    train_recurrent = subparsers.add_parser(
        "train-recurrent", help="train F(B,H) at broad variable recurrent lengths"
    )
    train_recurrent.add_argument("--initial-checkpoint", type=Path, required=True)
    train_recurrent.add_argument(
        "--dataset", type=Path, default=Path("data/processed/debug-10k/train.jsonl.zst")
    )
    train_recurrent.add_argument("--output", type=Path, default=Path("runs/recurrent-smoke"))
    train_recurrent.add_argument("--examples", type=int, default=512)
    train_recurrent.add_argument("--epochs", type=int, default=2)
    train_recurrent.add_argument("--batch-size", type=int, default=16)
    train_recurrent.add_argument("--effective-batch-size", type=int, default=128)
    train_recurrent.add_argument("--move-value-weight", type=float, default=0.25)
    train_recurrent.add_argument("--continuation-weight", type=float, default=0.25)
    train_recurrent.add_argument("--policy-temperature", type=float, default=0.75)
    train_recurrent.add_argument("--initial-max-steps", type=int, default=4)
    train_recurrent.add_argument("--final-max-steps", type=int, default=8)
    train_recurrent.add_argument("--minimum-steps", type=int, default=1)
    train_recurrent.add_argument("--proposal-auxiliary-weight", type=float, default=0.0)
    train_recurrent.add_argument("--disagreement-weight", type=float, default=0.0)
    train_recurrent.add_argument("--learning-rate", type=float, default=1e-4)
    train_recurrent.add_argument("--early-auxiliary-weight", type=float, default=0.1)
    train_recurrent.add_argument(
        "--progressive-weight",
        type=float,
        default=0.5,
        help="0..1 mix between ordinary and detached-prefix progressive losses",
    )
    train_recurrent.add_argument(
        "--max-gradient-steps",
        type=int,
        default=8,
        help="cap the backpropagated tail after a detached recurrent prefix",
    )
    train_recurrent.add_argument("--improvement-weight", type=float, default=0.1)
    train_recurrent.add_argument("--improvement-margin", type=float, default=0.01)
    train_recurrent.add_argument("--improvement-fraction", type=float, default=0.1)
    train_recurrent.add_argument(
        "--improvement-eligibility-epsilon", type=float, default=1e-3
    )
    train_recurrent.add_argument("--contraction-weight", type=float, default=0.0)
    train_recurrent.add_argument("--contraction-ratio", type=float, default=0.9)
    train_recurrent.add_argument(
        "--use-scratchpad-backup",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    train_recurrent.add_argument(
        "--use-future-workspaces",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="make compact action/resulting-board futures drive the root policy",
    )
    train_recurrent.add_argument("--future-workspace-weight", type=float, default=1.0)
    train_recurrent.add_argument("--future-board-weight", type=float, default=0.25)
    train_recurrent.add_argument("--future-value-weight", type=float, default=1.0)
    train_recurrent.add_argument("--max-residual-update", type=float, default=0.25)
    train_recurrent.add_argument(
        "--reset-recurrent-to-identity",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    train_recurrent.add_argument("--seed", type=int, default=0)
    train_recurrent.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    puzzles = subparsers.add_parser(
        "evaluate-puzzles", help="measure first-critical-move accuracy across recurrent depths"
    )
    puzzles.add_argument("--checkpoint", type=Path, required=True)
    puzzles.add_argument(
        "--manifest", type=Path, default=Path("data/eval/lichess-puzzles-50k.json")
    )
    puzzles.add_argument("--depths", default="0,1,2,4,8,16,32")
    puzzles.add_argument("--batch-size", type=int, default=128)
    puzzles.add_argument("--limit", type=int)
    puzzles.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    puzzles.add_argument("--output", type=Path, default=Path("runs/puzzle-curve/report.json"))
    quality = subparsers.add_parser(
        "evaluate-quality", help="measure per-position held-out quality across recurrent depths"
    )
    quality.add_argument("--checkpoint", type=Path, required=True)
    quality.add_argument("--dataset", type=Path, required=True)
    quality.add_argument("--depths", default="1,2,4,8,16")
    quality.add_argument("--value-weight", type=float, default=0.25)
    quality.add_argument("--move-value-weight", type=float, default=0.25)
    quality.add_argument("--continuation-weight", type=float, default=0.25)
    quality.add_argument("--policy-temperature", type=float, default=0.75)
    quality.add_argument("--batch-size", type=int, default=128)
    quality.add_argument("--limit", type=int)
    quality.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    quality.add_argument("--output", type=Path, default=Path("runs/quality-curve/report.json"))
    uci = subparsers.add_parser("uci", help="run the minimal UCI engine on standard input/output")
    uci.add_argument("--checkpoint", type=Path, help="play this checkpoint")
    uci.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    play = subparsers.add_parser(
        "play-stockfish", help="rate a checkpoint in Elo from paired games against Stockfish"
    )
    player = play.add_mutually_exclusive_group(required=True)
    player.add_argument("--checkpoint", type=Path)
    player.add_argument(
        "--external",
        choices=("chessbench9m",),
        help="play a released external engine by its own move rule instead of a checkpoint",
    )
    play.add_argument("--external-weights", type=Path, help="weights file of --external")
    play.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    play.add_argument("--steps", type=int, default=8)
    play.add_argument(
        "--elo", type=int, nargs="+", default=[1320], help="UCI_Elo levels; 0 is full strength"
    )
    play.add_argument("--pairs", type=int, default=50)
    play.add_argument("--movetime", type=float, default=0.1, help="Stockfish seconds per move")
    play.add_argument("--nodes", type=int, help="Stockfish node limit instead of a movetime")
    play.add_argument("--max-plies", type=int, default=300)
    play.add_argument("--openings", type=Path, default=Path("data/processed/games-v1/test"))
    play.add_argument("--seed", type=int, default=0)
    play.add_argument("--stockfish", type=Path)
    play.add_argument("--output", type=Path, default=Path("runs/stockfish-match"))
    play.add_argument(
        "--draw-guard",
        type=float,
        default=0.3,
        help="above this model value, skip moves that stalemate or repeat a third time",
    )
    play.add_argument(
        "--no-draw-guard", dest="draw_guard", action="store_const", const=None, help="play draws"
    )
    _add_move_selection(play)
    explain = subparsers.add_parser(
        "explain-move", help="explain one move of a checkpoint on CPU: depths, values, attention"
    )
    explain.add_argument("--checkpoint", type=Path, required=True)
    explain.add_argument("--fen", required=True)
    explain.add_argument("--move", required=True, help="the move to explain, in UCI")
    explain.add_argument("--compare", help="a second move to compare with, in UCI")
    explain.add_argument("--steps", type=int, default=8, help="depth for attention and occlusion")
    explain.add_argument("--depths", default="0,1,2,4,8,16")
    explain.add_argument("--json", action="store_true", help="print the full result as JSON")
    probe = subparsers.add_parser(
        "probe-steps",
        help="probe what the recurrent state stores at each step and test which parts cause gains",
    )
    probe.add_argument("--checkpoint", type=Path, required=True)
    probe.add_argument("--output", type=Path, required=True)
    probe.add_argument(
        "--positions", type=int, default=2_000, help="positions per sample (games, puzzles, labels)"
    )
    probe.add_argument("--depths", default="1,2,4,8,16")
    probe.add_argument("--early", type=int, default=4, help="early depth of the comparisons")
    probe.add_argument("--late", type=int, default=16, help="late depth of the comparisons")
    probe.add_argument("--games", type=Path, default=Path("data/processed/games-v1/validation"))
    probe.add_argument("--puzzles", type=Path, default=Path("data/processed/puzzles-v1/validation"))
    probe.add_argument("--labelled-games", type=Path, default=Path("data/processed/games-v1/train"))
    probe.add_argument(
        "--labels",
        type=Path,
        default=Path("data/processed/games-v1/train-labels"),
        help="every-move labels with Stockfish lines; skipped when the directory is missing",
    )
    probe.add_argument(
        "--strength",
        type=Path,
        help="a strength report directory whose positions and judged losses replace --games",
    )
    probe.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    probe.add_argument(
        "--max-vram-gb", type=float, default=1.0, help="VRAM ceiling in GiB; 0 disables the cap"
    )
    probe.add_argument(
        "--batch-size", type=int, help="positions per batch; by default from the cap"
    )
    probe.add_argument("--seed", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "doctor":
        return _doctor()
    if args.command == "eval-smoke":
        return _eval_smoke(args.output, args.device, args.seed)
    if args.command == "regret-smoke":
        return _regret_smoke(args.output, args.nodes)
    if args.command == "evaluate-regret":
        return _evaluate_regret_curve(args)
    if args.command == "match-smoke":
        return _match_smoke(args.output)
    if args.command == "evaluate-match":
        return _evaluate_match(args)
    if args.command == "trace-smoke":
        return _trace_smoke(args.output, args.steps, args.device)
    if args.command == "build-puzzles":
        return _build_puzzles(args.source, args.output, args.limit)
    if args.command == "build-corpus":
        return _build_corpus(args.source, args.output, args.limit, args.puzzle_manifest)
    if args.command == "build-puzzle-corpus":
        return _build_puzzle_corpus(args)
    if args.command == "build-game-corpus":
        return _build_game_corpus(args)
    if args.command == "import-chessbench":
        return _import_chessbench(args)
    if args.command == "label-moves":
        return _label_moves(args)
    if args.command == "merge-move-labels":
        return _merge_move_labels(args)
    if args.command in ("train-puzzles", "train-games"):
        return _train_puzzles(args)
    if args.command == "evaluate-puzzle-corpus":
        return _evaluate_puzzle_corpus(args)
    if args.command == "evaluate-puzzle-depth":
        return _evaluate_puzzle_depth(args)
    if args.command == "evaluate-strength":
        return _evaluate_strength(args)
    if args.command == "evaluate-reply":
        return _evaluate_reply(args)
    if args.command == "compare-strength":
        return _compare_strength(args)
    if args.command == "compare-puzzle-runs":
        return _compare_puzzle_runs(args)
    if args.command == "prepare-game-errors":
        return _prepare_game_errors(args)
    if args.command == "evaluate-game-errors":
        return _evaluate_game_errors(args)
    if args.command == "compare-game-errors":
        return _compare_game_errors(args)
    if args.command == "export-external-benchmark":
        return _export_external_benchmark(args)
    if args.command == "evaluate-full-puzzles":
        return _evaluate_full_puzzles(args)
    if args.command == "evaluate-stopping":
        return _evaluate_stopping(args)
    if args.command == "evaluate-warm-start":
        return _evaluate_warm_start(args)
    if args.command == "score-surprise":
        return _score_surprise(args)
    if args.command == "evaluate-external-strength":
        return _evaluate_external_strength(args)
    if args.command == "annotate-continuations":
        return _annotate_continuations(args)
    if args.command == "train-baseline":
        return _train_baseline(args)
    if args.command == "evaluate-checkpoint":
        return _evaluate_checkpoint(args)
    if args.command == "train-recurrent":
        return _train_recurrent(args)
    if args.command == "evaluate-puzzles":
        return _evaluate_puzzles(args)
    if args.command == "evaluate-quality":
        return _evaluate_quality(args)
    if args.command == "uci":
        from wombat.engine.uci import main as uci_main

        return uci_main(args.checkpoint, args.device)
    if args.command == "play-stockfish":
        return _play_stockfish(args)
    if args.command == "explain-move":
        return _explain_move(args)
    if args.command == "probe-steps":
        return _probe_steps(args)
    raise AssertionError(f"unhandled command: {args.command}")
