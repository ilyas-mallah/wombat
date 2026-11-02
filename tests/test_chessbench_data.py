from __future__ import annotations

from base64 import b64encode
from collections.abc import Iterator, Sequence
from hashlib import md5
import http.server
import json
from pathlib import Path
import random
import struct
import threading

import chess
import numpy as np
import pytest
import torch

from wombat import puzzle_training
from wombat.chess.encode import decode_position
from wombat.chess.moves import legal_action_mask, move_to_action, uci_action
from wombat.cli import build_parser, main
from wombat.data.chessbench import (
    KEY_SPACE,
    bag_records,
    decode_action_value,
    download,
    fen_key,
    fen_keys,
    import_chessbench,
    parse_shards,
    puzzle_line_fens,
    select_records,
)
from wombat.data.game_corpus import GameSplit, densify

# A position after 1. e4 e5 with its move counters, and the same position at other counters.
OPEN = "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"
OPEN_LATER = "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 4 6"
KINGS = "8/8/4k3/8/8/4K3/4P3/8 b - - 0 40"
EXCLUDED = "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"
PARTIAL = "8/5k2/8/8/8/8/2K5/8 w - - 0 50"


def _record(fen: str, move: str, win: float) -> bytes:
    def text(value: str) -> bytes:
        data, length = value.encode(), len(value.encode())
        prefix = bytearray()
        while True:
            prefix.append((length & 0x7F) | (0x80 if length > 0x7F else 0))
            length >>= 7
            if not length:
                return bytes(prefix) + data

    return text(fen) + text(move) + struct.pack(">d", win)


def _write_bag(path: Path, records: Sequence[bytes]) -> Path:
    """Records followed by each one's end offset, as ChessBench's ``BagWriter`` leaves them."""

    ends, total = [], 0
    for record in records:
        total += len(record)
        ends.append(total)
    path.write_bytes(b"".join(records) + b"".join(struct.pack("<q", end) for end in ends))
    return path


def _scores(fen: str, seed: int) -> dict[str, float]:
    """A distinct made-up win chance for every legal move."""

    moves = sorted(move.uci() for move in chess.Board(fen).legal_moves)
    wins = random.Random(seed).sample(range(1, 1000), len(moves))
    return {move: win / 1000 for move, win in zip(moves, wins, strict=True)}


def _records(scores: dict[str, dict[str, float]]) -> list[bytes]:
    return [_record(fen, move, win) for fen, moves in scores.items() for move, win in moves.items()]


def test_bag_records_decode_to_fen_move_and_win(tmp_path: Path) -> None:
    long_fen = "x" * 200
    records = [_record(OPEN, "g1f3", 0.53), _record(long_fen, "e7e8q", 1.0)]
    bag = _write_bag(tmp_path / "tiny.bag", records)
    assert list(bag_records(bag)) == records
    assert decode_action_value(records[0]) == (OPEN, "g1f3", 0.53)
    assert decode_action_value(records[1]) == (long_fen, "e7e8q", 1.0)
    assert list(bag_records(_write_bag(tmp_path / "empty.bag", []))) == []
    with pytest.raises(ValueError, match="malformed"):
        decode_action_value(records[0] + b"\x00")


def test_selection_keeps_a_share_of_fens_with_all_their_records(tmp_path: Path) -> None:
    fens = [chess.Board().fen(), OPEN, KINGS, EXCLUDED, PARTIAL, "x" * 150]
    records = [_record(fen, move, 0.5) for fen in fens for move in ("a2a3", "b2b3")]
    everything = select_records(records)
    assert everything.records == len(records)
    assert np.all(np.diff(everything.keys.astype(np.float64)) >= 0)
    assert sorted(everything.fens.tolist()) == sorted(fen.encode() for fen in fens)
    threshold = KEY_SPACE // 2
    half = select_records(records, threshold)
    chosen = {fen for fen in fens if fen_key(fen.encode()) < threshold}
    assert {bytes(fen).decode() for fen in half.fens} == chosen
    assert len(half.keys) == 2 * len(chosen)
    assert set(half.keys.tolist()) == {fen_key(fen.encode()) for fen in chosen}


def test_uci_action_matches_move_to_action() -> None:
    randomizer = random.Random(3)
    for _ in range(20):
        board = chess.Board()
        for _ in range(randomizer.randrange(0, 60)):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(randomizer.choice(moves))
        for move in board.legal_moves:
            assert uci_action(move.uci(), board.turn) == move_to_action(board, move)


def test_puzzle_lines_list_every_position(tmp_path: Path) -> None:
    path = tmp_path / "puzzles.csv"
    path.write_text(f"PuzzleId,FEN,Moves\nabc,{chess.Board().fen()},e2e4 e7e5\n", encoding="utf-8")
    board = chess.Board()
    expected = [board.fen()]
    for move in ("e2e4", "e7e5"):
        board.push_uci(move)
        expected.append(board.fen())
    assert list(puzzle_line_fens(path)) == expected


@pytest.fixture(scope="module")
def scores() -> dict[str, dict[str, float]]:
    fens = (chess.Board().fen(), OPEN, KINGS, EXCLUDED, PARTIAL)
    table = {fen: _scores(fen, seed) for seed, fen in enumerate(fens)}
    # Whichever counters come first is kept, so both carry the same scores.
    table[OPEN_LATER] = table[OPEN]
    del table[PARTIAL]["c2c3"]
    return table


def _import(output: Path, bags: Sequence[Path], **options: object) -> dict[str, object] | None:
    arguments: dict[str, object] = {
        "work_dir": output / "work",
        "fraction": 1.0,
        "excluded": fen_keys([EXCLUDED]),
        "exclusion": {"fens": 1},
        "partitions": 3,
        "low_priority": False,
        "minimum_free_gb": 0,
        **options,
    }
    return import_chessbench(
        [(bag.stem, str(bag)) for bag in bags],
        output,
        **arguments,  # type: ignore[arg-type]
    )


@pytest.fixture(scope="module")
def imported(
    scores: dict[str, dict[str, float]], tmp_path_factory: pytest.TempPathFactory
) -> tuple[Path, dict[str, object]]:
    """Every position's moves split between two bags, as the train shards scatter them."""

    root = tmp_path_factory.mktemp("chessbench")
    halves: list[list[bytes]] = [[], []]
    for fen, moves in scores.items():
        for number, (move, win) in enumerate(moves.items()):
            halves[number % 2].append(_record(fen, move, win))
    for half in halves:
        random.Random(0).shuffle(half)
    first = _write_bag(root / "first.bag", halves[0])
    second = _write_bag(root / "second.bag", halves[1])
    output = root / "corpus"
    with pytest.raises(ValueError, match="no complete positions"):
        _import(output, [first])
    manifest = _import(output, [first, second])
    assert manifest is not None
    return output, manifest


def test_join_keeps_complete_positions_once_every_bag_is_in(
    imported: tuple[Path, dict[str, object]], scores: dict[str, dict[str, float]]
) -> None:
    output, manifest = imported
    assert manifest["records"] == sum(len(moves) for moves in scores.values())
    assert manifest["fens"] == len(scores)
    assert manifest["rejected"] == {"excluded": 1, "missing_moves": 1}
    # The later counters repeat a position kept from its first occurrence.
    assert manifest["duplicates"] == 1
    assert manifest["splits"]["train"]["positions"] == 3  # type: ignore[index]
    assert [entry["name"] for entry in manifest["shards"]] == ["first", "second"]  # type: ignore[attr-defined]
    stored = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert stored["splits"] == manifest["splits"]
    labels = json.loads((output / "train-labels" / "manifest.json").read_text(encoding="utf-8"))
    assert labels["split"]["positions"] == 3
    assert labels["moves"] == manifest["splits"]["train"]["moves"]  # type: ignore[index]
    assert not (output / "join").exists()


def test_labels_round_trip_through_game_split(
    imported: tuple[Path, dict[str, object]], scores: dict[str, dict[str, float]]
) -> None:
    output, _ = imported
    split = GameSplit(output / "train", labels=output / "train-labels")
    assert len(split) == 3
    assert np.array_equal(split.trainable(), np.arange(3))
    found = set()
    for row in range(len(split)):
        board = decode_position(split.pieces[row], split.state[row])
        fen = next(fen for fen in scores if chess.Board(fen).board_fen() == board.board_fen())
        found.add(fen)
        assert board.halfmove_clock == 0
        wins = {
            move_to_action(board, chess.Move.from_uci(move)): win
            for move, win in scores[fen].items()
        }
        best = max(wins, key=wins.__getitem__)
        assert split.action[row] == best
        start, stop = split.label_offsets[row], split.label_offsets[row + 1]
        actions = split.label_actions[start:stop].tolist()
        assert actions == sorted(wins, key=lambda action: -wins[action])
        assert np.allclose(split.label_win[start:stop], [wins[a] for a in actions], atol=1e-3)
        candidates = split.candidate_actions[
            split.candidate_offsets[row] : split.candidate_offsets[row + 1]
        ]
        assert candidates.tolist() == actions[:5]
        assert np.array_equal(split.legal_mask(np.array([row]))[0], legal_action_mask(board))
        targets = split.targets(np.array([row]))
        assert np.array_equal(targets["move_value_mask"][0], legal_action_mask(board))
        values = targets["move_value_target"][0, actions]
        assert np.allclose(values, [2 * wins[a] - 1 for a in actions], atol=2e-3)
        assert targets["policy_target"][0].argmax() == best
    assert found == {chess.Board().fen(), OPEN, KINGS}
    indices = np.arange(3)
    built = densify(split.sparse_batch(indices), split.temperature, torch.device("cpu"))
    for key, value in split.batch(indices).items():
        if value.is_floating_point():
            assert torch.allclose(built[key], value, atol=1e-6, equal_nan=True), key
        else:
            assert torch.equal(built[key], value), key


def test_train_games_trains_with_chessbench_as_extra_positions(
    imported: tuple[Path, dict[str, object]],
    game_corpus: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(puzzle_training, "lower_priority", lambda: None)
    monkeypatch.setattr("wombat.eval.puzzle_corpus_eval.lower_priority", lambda: None)
    output, _ = imported
    run = tmp_path / "run"
    arguments = ["train-games", "--corpus", str(game_corpus)]
    arguments += ["--extra-train", str(output), str(output / "train-labels")]
    arguments += ["--output", str(run), "--device", "cpu", "--steps", "2", "--batch-size", "2"]
    arguments += ["--warmup-steps", "1", "--max-depth", "2", "--max-gradient-steps", "1"]
    arguments += ["--eval-every", "2", "--eval-depths", "0,1", "--d-model", "24", "--heads", "6"]
    arguments += ["--mlp-width", "48", "--latent-tokens", "4", "--recurrent-blocks", "1"]
    assert main(arguments) == 0
    config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    games = len(GameSplit(game_corpus / "train"))
    assert config["train_positions"] == games + 3
    row = json.loads((run / "metrics.jsonl").read_text().splitlines()[-1])
    assert np.isfinite(row["train_loss"])


def test_imported_shards_are_skipped_and_fractions_must_agree(
    scores: dict[str, dict[str, float]], tmp_path: Path
) -> None:
    bag = _write_bag(tmp_path / "only.bag", _records(scores))
    output = tmp_path / "corpus"
    first = _import(output, [bag])
    assert first is not None
    marker = output / "records" / "only.json"
    stamp = marker.stat().st_mtime_ns
    again = _import(output, [bag])
    assert marker.stat().st_mtime_ns == stamp
    assert again is not None and again["splits"] == first["splits"]
    other = _write_bag(tmp_path / "other.bag", _records(scores))
    with pytest.raises(ValueError, match="another fraction"):
        _import(output, [other], fraction=0.5)


def test_deleting_records_keeps_the_corpus(
    scores: dict[str, dict[str, float]], tmp_path: Path
) -> None:
    bag = _write_bag(tmp_path / "only.bag", _records(scores))
    kept = _import(tmp_path / "kept", [bag])
    deleted = _import(tmp_path / "deleted", [bag], delete_records=True)
    assert kept is not None and deleted is not None
    assert deleted["splits"] == kept["splits"]
    assert not (tmp_path / "deleted" / "records").exists()
    assert not (tmp_path / "deleted" / "fens").exists()
    assert sorted(path.name for path in (tmp_path / "kept").iterdir()) == [
        "excluded_keys.npy",
        "fens",
        "manifest.json",
        "records",
        "train",
        "train-labels",
        "work",
    ]


@pytest.fixture
def server(tmp_path: Path) -> Iterator[tuple[str, bytes]]:
    payload = bytes(random.Random(1).randrange(256) for _ in range(100_000))
    digest = b64encode(md5(payload).digest()).decode()

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_HEAD(self) -> None:
            self._send(head=True)

        def do_GET(self) -> None:
            self._send(head=False)

        def _send(self, head: bool) -> None:
            ranged = self.headers.get("Range")
            start = int(ranged.split("=")[1].split("-")[0]) if ranged else 0
            body = payload if head else payload[start:]
            self.send_response(206 if start else 200)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("x-goog-hash", "crc32c=AAAAAA==")
            self.send_header("x-goog-hash", f"md5={digest}")
            self.end_headers()
            if not head:
                self.wfile.write(body)

        def log_message(self, *args: object) -> None:
            pass

    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}/shard.bag", payload
    finally:
        httpd.shutdown()


def test_download_resumes_a_partial_file_and_restarts_a_corrupt_one(
    server: tuple[str, bytes], tmp_path: Path
) -> None:
    url, payload = server
    path = tmp_path / "shard.bag"
    path.with_name("shard.bag.part").write_bytes(payload[:30_000])
    first = download(url, path)
    assert path.read_bytes() == payload
    assert first["bytes"] == len(payload)
    path.unlink()
    path.with_name("shard.bag.part").write_bytes(b"\x00" * 30_000)
    assert download(url, path, attempts=2) == first
    assert path.read_bytes() == payload
    assert not path.with_name("shard.bag.part").exists()


def test_shard_lists_read_ranges_and_single_shards() -> None:
    assert parse_shards("0-3,7, 2") == [0, 1, 2, 3, 7]
    assert parse_shards("") == []
    with pytest.raises(ValueError):
        parse_shards("2140-2148")


def test_import_chessbench_parser_defaults() -> None:
    args = build_parser().parse_args(["import-chessbench"])
    assert (args.shards, args.bags, args.fraction, args.join) == ("", [], 1.0, True)
    assert (args.delete_records, args.exclusion_keys, args.partitions) == (False, None, 64)
    assert Path("data/processed/games-v2/test") in args.exclude_splits
    assert len(args.exclude_splits) == 6
    assert args.output == Path("data/processed/chessbench")
