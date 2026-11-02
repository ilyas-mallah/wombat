# Third-party software, data and tools

Wombat's own code is licensed under the Apache License 2.0 (`LICENSE`, `NOTICE`). This file lists everything else
the code imports, runs or downloads, with the license and the page it was read from. Every page was read on
2026-11-02. This is a record of what the sources state, not legal advice.

## Python packages

None is copied into this repository. They are installed by `pip` from `pyproject.toml`.

| Package | Role | License | Source |
| --- | --- | --- | --- |
| python-chess (`chess`, pulled in by the `python-chess` 1.999 package) | Board, legal moves, PGN, UCI engine control. Imported throughout the package | GPL-3.0-or-later | https://github.com/niklasf/python-chess (README: "licensed under the GPL 3 (or any later version at your option)"), https://pypi.org/pypi/chess/json |
| PyTorch | Model, training | BSD-3-Clause and other permissive licenses (PyPI SPDX expression: Apache-2.0 AND Apache-2.0 WITH LLVM-exception AND BSD-2-Clause AND BSD-3-Clause AND BSL-1.0 AND ISC AND MIT AND Zlib) | https://pypi.org/pypi/torch/json |
| NumPy | Arrays | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 | https://pypi.org/pypi/numpy/json |
| PyYAML | Config files | MIT | https://pypi.org/pypi/PyYAML/json |
| psutil | Memory and process limits | BSD-3-Clause | https://pypi.org/pypi/psutil/json |
| tqdm | Progress bars | MPL-2.0 AND MIT | https://pypi.org/pypi/tqdm/json |
| zstandard | Reads the Lichess `.zst` files | BSD-3-Clause | https://pypi.org/pypi/zstandard/json |
| pytest (dev only) | Tests | MIT | https://pypi.org/pypi/pytest/json |

Not confirmed: the `cu130` PyTorch wheels from `download.pytorch.org` bundle NVIDIA CUDA libraries that carry
NVIDIA's own license terms. I did not read those terms.

## External programs

Wombat starts these as separate processes (UCI over standard input and output, or a subprocess). It does not link
them or include their code. Each is found through `configs/local.yaml` or the `PATH`.

| Program | Role | License | Source |
| --- | --- | --- | --- |
| Stockfish 19 | Move labels, regret scoring, the opponent in the games ladder | GPL-3.0 | https://github.com/official-stockfish/Stockfish (README: "free and distributed under the GNU General Public License version 3"), release: https://stockfishchess.org/blog/2026/stockfish-19/ |
| Cute Chess CLI | Paired engine matches | GPL-3.0-or-later (its `projects/lib/components` and `projects/gui/components` folders are MIT) | https://github.com/cutechess/cutechess (README: "released under the GPLv3+ license") |
| Ordo (optional) | Rating computation from game results | GPL-3.0 | https://github.com/michiguel/Ordo (README: "The source code is available under the GPL.") |

The labels and scores that Stockfish writes into Wombat's training data and result files are program output. I found
no statement in the Stockfish README that claims them. I did not read the Stockfish network (NNUE) license terms.

## Data

Nothing here is in the repository. The commands download it into `data/`.

| Data | Used for | License | Source |
| --- | --- | --- | --- |
| Lichess puzzle database | Puzzle training, the puzzle benchmark | CC0 | https://database.lichess.org/ ("Database exports are released under the Creative Commons CC0 license") |
| Lichess evaluation database | Positions for the game corpus, labelled again by Stockfish | CC0 | https://database.lichess.org/ (same statement) |
| ChessBench action-value test data (`action_value_data.bag`) and `puzzles.csv` | Held-out game positions for the error-tail screen, the puzzle comparison | CC0 for the portions from Lichess, CC BY 4.0 for the rest | https://github.com/google-deepmind/searchless_chess (README, License section) |
| ChessBench 9M weights | The comparison model. The weights are converted to a PyTorch file by the user and are not distributed here | CC BY 4.0 | https://github.com/google-deepmind/searchless_chess (README: "The model weights are licensed under Creative Commons Attribution 4.0 (CC-BY)") |

`wombat import-chessbench` can read ChessBench's training shards (same license as the test data). The released
checkpoints were not trained on them.

## Ported code

`src/wombat/external/chessbench.py` is a PyTorch port of the ChessBench 9M model and tokenizer from
google-deepmind/searchless_chess, Apache License 2.0, Copyright DeepMind Technologies Limited
(https://github.com/google-deepmind/searchless_chess, file `src/tokenizer.py` header read 2026-11-02). It keeps the
attribution in the module docstring and in `NOTICE`. `src/wombat/data/chessbench.py` reads the same `.bag` container
format with its own code. The ChessBench paper is Ruoss et al., "Amortized Planning with Large-Scale Transformers:
A Case Study on Chess", NeurIPS 2024.

## Conflicts and open points with the Apache-2.0 license

- **python-chess is GPL-3.0-or-later and Wombat imports it.** The GNU GPL FAQ says that when a program links to a GPL
  library, "the terms of the GPL apply to the entire combination", and that "the software modules that link with the
  library may be under various GPL compatible licenses, but the work as a whole must be licensed under the GPL"
  (https://www.gnu.org/licenses/gpl-faq.html, entry `IfLibraryIsGPL`). The FSF lists Apache 2.0 as "compatible with
  version 3 of the GNU GPL" and not with GPL version 2 (https://www.gnu.org/licenses/license-list.html, entry
  `apache2`). python-chess allows version 3 or later, so the pair is compatible. In practice, Wombat's source stays
  Apache-2.0 and is published without python-chess. Anyone who ships Wombat together with python-chess, for example
  in a container image, a bundled app or a Hugging Face Space, ships a combined work that has to meet the GPL-3.0 terms
  as a whole. Removing the dependency would take a replacement for the board, move generation and PGN code.
- **Stockfish, Cute Chess and Ordo are GPL programs run as separate processes.** No code from them is in this
  repository, so the Apache-2.0 license does not conflict. A package that bundles their binaries has to follow the GPL
  for those binaries.
- **The ported model file carries Apache-2.0 attribution.** Redistributing it requires `LICENSE` and `NOTICE`, which
  this repository includes.
- **ChessBench weights and data (CC BY 4.0) need attribution** wherever they are redistributed. This repository
  redistributes neither.
- **Not confirmed:** the NVIDIA CUDA library terms in the `cu130` wheels, the Stockfish network license, and the
  exact CC BY and CC0 split inside ChessBench's data (the README does not say which records are which).
