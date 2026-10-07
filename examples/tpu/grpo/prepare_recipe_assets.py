#!/usr/bin/env python3
# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Pre-flight preparation script for Trellis PyTorch RL Recipes on TPU.

Responsibilities:
1. Patches Qwen3-*-Base `generation_config.json` with multi-EOS token IDs `[151645, 151643]`
   so vLLM stops cleanly on both `<|im_end|>` and `<|endoftext|>`.
2. Downloads and preprocesses datasets into `verl` parquet format:
   - Recipe 1: `openai/gsm8k` -> `<data_dir>/gsm8k/{train,test}.parquet`
   - Recipe 2A: `BytedTsinghua-SIA/DAPO-Math-17k` -> `<data_dir>/dapo_math/dapo-math-17k.parquet`
                `BytedTsinghua-SIA/AIME-2024` -> `<data_dir>/aime_2024/aime-2024.parquet`
   - Recipe 2B: `PRIME-RL/Eurus-2-RL-Data` (coding subset) -> `<data_dir>/eurus_2_rl/{train,test}.parquet`
"""

import argparse
import json
from pathlib import Path
from typing import Sequence

QWEN3_MULTI_EOS_TOKEN_IDS = [151645, 151643]
QWEN3_BOS_PAD_TOKEN_ID = 151643


def patch_qwen3_generation_config(model_dir: str | Path) -> Path:
    """Ensure `generation_config.json` in `model_dir` contains multi-EOS `[151645, 151643]`."""
    model_path = Path(model_dir)
    if not model_path.exists():
        raise FileNotFoundError(f"Model directory does not exist: {model_path}")

    gen_cfg_path = model_path / "generation_config.json"
    if gen_cfg_path.exists():
        with open(gen_cfg_path, encoding="utf-8") as f:
            cfg = json.load(f)
    else:
        cfg = {
            "bos_token_id": QWEN3_BOS_PAD_TOKEN_ID,
            "pad_token_id": QWEN3_BOS_PAD_TOKEN_ID,
        }

    existing_eos = cfg.get("eos_token_id")
    if isinstance(existing_eos, int):
        eos_list = [existing_eos]
    elif isinstance(existing_eos, list):
        eos_list = list(existing_eos)
    else:
        eos_list = []

    for token_id in QWEN3_MULTI_EOS_TOKEN_IDS:
        if token_id not in eos_list:
            eos_list.append(token_id)

    cfg["eos_token_id"] = eos_list
    cfg.setdefault("bos_token_id", QWEN3_BOS_PAD_TOKEN_ID)
    cfg.setdefault("pad_token_id", QWEN3_BOS_PAD_TOKEN_ID)

    with open(gen_cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, sort_keys=True)
        f.write("\n")

    return gen_cfg_path


def prepare_dapo_and_aime(data_root: Path) -> tuple[Path, Path]:
    """Download `BytedTsinghua-SIA/DAPO-Math-17k` and `BytedTsinghua-SIA/AIME-2024` to parquet."""
    import datasets

    dapo_dir = data_root / "dapo_math"
    aime_dir = data_root / "aime_2024"
    dapo_dir.mkdir(parents=True, exist_ok=True)
    aime_dir.mkdir(parents=True, exist_ok=True)

    dapo_out = dapo_dir / "dapo-math-17k.parquet"
    if not dapo_out.exists():
        ds_dapo = datasets.load_dataset("BytedTsinghua-SIA/DAPO-Math-17k", "default")["train"]
        ds_dapo.to_parquet(str(dapo_out))

    aime_out = aime_dir / "aime-2024.parquet"
    if not aime_out.exists():
        ds_aime = datasets.load_dataset("BytedTsinghua-SIA/AIME-2024", "default")["train"]
        ds_aime.to_parquet(str(aime_out))

    return dapo_out, aime_out


def prepare_prime_code(data_root: Path) -> tuple[Path, Path]:
    """Download `PRIME-RL/Eurus-2-RL-Data` and filter to single-turn coding (`prime_code`)."""
    import datasets

    out_dir = data_root / "eurus_2_rl"
    out_dir.mkdir(parents=True, exist_ok=True)
    train_out = out_dir / "train.parquet"
    test_out = out_dir / "test.parquet"

    if not train_out.exists() or not test_out.exists():
        ds = datasets.load_dataset("PRIME-RL/Eurus-2-RL-Data")
        for split_name, target_path in [("train", train_out), ("validation", test_out)]:
            split_ds = ds[split_name]
            coding_ds = split_ds.filter(
                lambda ex: ex.get("data_source") == "prime_code" or ex.get("ability") == "coding"
            )
            if len(coding_ds) == 0:
                coding_ds = split_ds
            coding_ds.to_parquet(str(target_path))

    return train_out, test_out


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare datasets and model configs for Trellis RL recipes.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/data/jialei/data"),
        help="Root directory where parquet datasets are saved.",
    )
    parser.add_argument(
        "--patch-model-dirs",
        nargs="*",
        default=[],
        help="Model checkpoint directories whose generation_config.json should be patched with multi-EOS IDs.",
    )
    parser.add_argument(
        "--recipes",
        nargs="*",
        choices=["dapo_math", "prime_code"],
        default=[],
        help="Datasets to download/export (for gsm8k, use examples/data_preprocess/gsm8k.py).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    for model_dir in args.patch_model_dirs:
        out_path = patch_qwen3_generation_config(model_dir)
        print(f"[prepare_recipe_assets] Patched multi-EOS generation_config: {out_path}")

    if "dapo_math" in args.recipes:
        dapo_out, aime_out = prepare_dapo_and_aime(args.data_root)
        print(f"[prepare_recipe_assets] Prepared DAPO train={dapo_out}, AIME val={aime_out}")

    if "prime_code" in args.recipes:
        train_out, test_out = prepare_prime_code(args.data_root)
        print(f"[prepare_recipe_assets] Prepared PRIME coding train={train_out}, val={test_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
