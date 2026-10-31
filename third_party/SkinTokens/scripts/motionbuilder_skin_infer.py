#!/usr/bin/env python3
"""Kimodo bridge: SkinTokens skin inference from UniRig skeleton (numpy, no bpy_server)."""

from __future__ import annotations

import argparse
import os
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

os.environ.setdefault("XFORMERS_IGNORE_FLASH_VERSION_CHECK", "1")
os.environ.setdefault("TRANSFORMERS_ATTN_IMPLEMENTATION", "sdpa")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model.spec import ModelInput  # noqa: E402
from src.rig_package.info.asset import Asset  # noqa: E402
from src.server.spec import get_model  # noqa: E402
from src.tokenizer.parse import get_tokenizer  # noqa: E402
from src.data.transform import Transform  # noqa: E402
from src.tokenizer.spec import TokenizeInput  # noqa: E402

DEFAULT_CKPT = "experiments/articulation_xl_quantization_256_token_4/grpo_1400.ckpt"


def _load_input(path: Path) -> dict:
    data = np.load(path, allow_pickle=True)
    return {key: data[key][()] for key in data.files}


def _normalize_parents(parents_obj) -> np.ndarray:
    out: list[int] = []
    for p in np.asarray(parents_obj, dtype=object).ravel():
        if p is None:
            out.append(-1)
        else:
            out.append(int(p))
    return np.array(out, dtype=np.int64)


def _build_asset(payload: dict) -> Asset:
    vertices = np.asarray(payload["vertices"], dtype=np.float32)
    faces = np.asarray(payload["faces"], dtype=np.int64)
    joints = np.asarray(payload["joints"], dtype=np.float32)
    parents = _normalize_parents(payload["parents"])
    names_raw = payload.get("names")
    if names_raw is None:
        names = [f"bone_{i}" for i in range(joints.shape[0])]
    else:
        names = [str(n) for n in list(names_raw)]

    j_count = joints.shape[0]
    matrix_local = np.tile(np.eye(4, dtype=np.float32), (j_count, 1, 1))
    matrix_local[:, :3, 3] = joints

    asset = Asset(
        vertices=vertices,
        faces=faces,
        joint_names=names,
        parents=parents,
        matrix_local=matrix_local,
        cls="articulation",
    )
    asset.build_normals()
    return asset


def _as_tensor(value) -> Tensor:
    if isinstance(value, Tensor):
        return value
    return torch.as_tensor(value)


def _collate_single(model, model_input: ModelInput) -> dict:
    processed_list = model.process_fn([model_input])
    for i, item in enumerate(processed_list):
        non = processed_list[i].get("non", {})
        non["model_input"] = deepcopy(model_input)
        processed_list[i]["non"] = non

    item = processed_list[0]
    batch: dict = {
        "vertices": item["vertices"].unsqueeze(0),
        "normals": item["normals"].unsqueeze(0),
        "cls": [item["non"]["cls"]],
        "model_input": [item["non"]["model_input"]],
    }
    if "skeleton_tokens" in item:
        batch["skeleton_tokens"] = _as_tensor(item["skeleton_tokens"]).unsqueeze(0)
        batch["skeleton_mask"] = _as_tensor(item["skeleton_mask"]).unsqueeze(0)
    return batch


def run_inference(
    input_path: Path,
    output_path: Path,
    *,
    model_ckpt: str,
    top_k: int = 5,
    top_p: float = 0.95,
    temperature: float = 1.0,
    repetition_penalty: float = 2.0,
    num_beams: int = 10,
) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("SkinTokens skin inference requires CUDA (>=14GB VRAM recommended).")

    payload = _load_input(input_path)
    ckpt_path = Path(model_ckpt)
    if not ckpt_path.is_file():
        ckpt_path = ROOT / model_ckpt
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"SkinTokens checkpoint not found: {model_ckpt}")

    device = "cuda"
    model = get_model(str(ckpt_path), hf_path=None, device=device)
    model.eval()

    assert model.tokenizer_config is not None
    tokenizer = get_tokenizer(**model.tokenizer_config)
    transform = Transform.parse(**model.transform_config["predict_transform"])

    asset = _build_asset(payload)
    transform.apply(asset=asset)

    token_input = TokenizeInput(
        joints=asset.joints,
        parents=asset.parents,
        cls=asset.cls,
        joint_names=asset.joint_names,
    )
    tokens = tokenizer.tokenize(input=token_input)
    model_input = ModelInput(asset=asset, tokens=tokens)

    batch = _collate_single(model, model_input)
    batch["generate_kwargs"] = {
        "max_length": 2048,
        "top_k": int(top_k),
        "top_p": float(top_p),
        "temperature": float(temperature),
        "repetition_penalty": float(repetition_penalty),
        "num_return_sequences": 1,
        "num_beams": int(num_beams),
        "do_sample": True,
    }

    batch = {
        key: (value.to(device) if isinstance(value, Tensor) else value)
        for key, value in batch.items()
    }

    if "skeleton_mask" not in batch or "skeleton_tokens" not in batch:
        raise RuntimeError("Failed to tokenize UniRig skeleton for SkinTokens.")

    mask = batch["skeleton_mask"][0] == 1
    skeleton_tokens = batch["skeleton_tokens"][0][mask].detach().cpu().numpy()

    with torch.inference_mode():
        preds = model.predict_step(
            batch,
            skeleton_tokens=[skeleton_tokens],
            make_asset=True,
        )["results"]

    result_asset = preds[0].asset
    if result_asset is None or result_asset.skin is None:
        raise RuntimeError("SkinTokens did not produce skin weights.")

    skin = np.asarray(result_asset.skin, dtype=np.float32)
    if skin.ndim != 2:
        raise ValueError(f"Unexpected skin shape: {skin.shape}")

    n_joints = int(np.asarray(payload["joints"]).shape[0])
    if skin.shape[0] == n_joints and skin.shape[1] != n_joints:
        skin = skin.T

    out_vertices = np.asarray(payload["vertices"], dtype=np.float32)
    if skin.shape[0] != out_vertices.shape[0]:
        raise ValueError(
            f"Skin vertex count {skin.shape[0]} != mesh {out_vertices.shape[0]}"
        )
    if skin.shape[1] != n_joints:
        raise ValueError(
            f"Skin joint count {skin.shape[1]} != skeleton {n_joints}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        skin=skin,
        vertices=out_vertices,
    )
    print(f"Wrote skin weights: {output_path} skin={skin.shape}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Kimodo SkinTokens skin inference")
    parser.add_argument("--input", required=True, help="kimodo_skin_input.npz path")
    parser.add_argument("--output", required=True, help="kimodo_skin_output.npz path")
    parser.add_argument("--model_ckpt", default=DEFAULT_CKPT)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--repetition_penalty", type=float, default=2.0)
    parser.add_argument("--num_beams", type=int, default=10)
    args = parser.parse_args()

    run_inference(
        Path(args.input),
        Path(args.output),
        model_ckpt=args.model_ckpt,
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        num_beams=args.num_beams,
    )


if __name__ == "__main__":
    main()
