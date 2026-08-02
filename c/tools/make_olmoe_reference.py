#!/usr/bin/env python3
"""Create a greedy OLMoE continuation with a separate Transformers implementation.

This tool does not prove that Transformers is a mathematically independent
oracle. It records enough source, library and generation provenance to make one
reference continuation reproducible and auditable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import tempfile
from pathlib import Path
from typing import Any

MAX_TOTAL_TOKENS = 2048
MAX_NEW_TOKENS = 512


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _sample_file(path: Path, sample_size: int = 65536) -> dict[str, Any]:
    size = path.stat().st_size
    positions = {0, max(0, size // 2 - sample_size // 2), max(0, size - sample_size)}
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for position in sorted(positions):
            stream.seek(position)
            chunk = stream.read(sample_size)
            digest.update(position.to_bytes(8, "little"))
            digest.update(len(chunk).to_bytes(8, "little"))
            digest.update(chunk)
    return {"name": path.name, "size": size, "sampled_sha256": digest.hexdigest()}


def fingerprint_local_model(model: Path) -> str:
    model = model.expanduser().resolve()
    if not model.is_dir():
        raise ValueError(f"local model directory not found: {model}")
    entries: list[dict[str, Any]] = []
    for name in ("config.json", "tokenizer.json", "model.safetensors.index.json"):
        path = model / name
        if path.is_file():
            entries.append({"name": name, "size": path.stat().st_size, "sha256": sha256_file(path)})
    shards = sorted(model.glob("*.safetensors"))
    if not shards:
        raise ValueError(f"no safetensors shards found in local model: {model}")
    entries.extend(_sample_file(path) for path in shards)
    payload = json.dumps(
        {"schema_version": 1, "files": entries},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_once_json(path: Path, value: dict[str, Any]) -> None:
    data = (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            target_fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            if path.is_file() and not path.is_symlink() and path.read_bytes() == data:
                return
            raise FileExistsError(f"refusing to overwrite write-once reference: {path}")
        try:
            with os.fdopen(target_fd, "wb") as target:
                target.write(data)
                target.flush()
                os.fsync(target.fileno())
        except BaseException:
            try:
                path.unlink()
            except OSError:
                pass
            raise
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a provenance-bound greedy OLMoE token reference."
    )
    parser.add_argument("--model", required=True, help="Hugging Face repo ID or local model directory")
    parser.add_argument("--revision", help="Hub revision; resolved to an immutable commit")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda", "mps"), default="auto",
        help="model placement request; auto uses Accelerate device_map=auto",
    )
    parser.add_argument(
        "--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto",
    )
    parser.add_argument(
        "--plain-prompt", action="store_true",
        help="tokenize directly instead of using the tokenizer chat template",
    )
    return parser.parse_args(argv)


def resolve_source(model: str, requested_revision: str | None, model_info_fn) -> dict[str, Any]:
    path = Path(model).expanduser()
    if path.exists():
        if requested_revision is not None:
            raise ValueError("--revision is not valid for a local model directory")
        resolved = path.resolve()
        return {
            "source_kind": "local",
            "load_target": str(resolved),
            "resolved_revision": None,
            "source_fingerprint": fingerprint_local_model(resolved),
        }
    info = model_info_fn(model, revision=requested_revision)
    sha = str(getattr(info, "sha", "") or "")
    if len(sha) != 40 or any(character not in "0123456789abcdef" for character in sha.lower()):
        raise ValueError("Hugging Face did not return a valid immutable 40-character commit")
    return {
        "source_kind": "huggingface",
        "load_target": model,
        "resolved_revision": sha.lower(),
        "source_fingerprint": None,
    }


def resolve_dtype(name: str, torch_module):
    if name == "auto":
        return "auto"
    return {
        "float32": torch_module.float32,
        "float16": torch_module.float16,
        "bfloat16": torch_module.bfloat16,
    }[name]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if isinstance(args.tokens, bool) or not 1 <= args.tokens <= MAX_NEW_TOKENS:
        raise SystemExit(f"--tokens must be between 1 and {MAX_NEW_TOKENS}")
    if not args.prompt.strip() or len(args.prompt) > 32768:
        raise SystemExit("--prompt must be non-empty and at most 32768 characters")

    try:
        import torch
        import transformers
        from huggingface_hub import model_info
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            f"Missing dependency: {exc}. Install with: "
            "pip install torch transformers accelerate safetensors huggingface_hub"
        ) from exc

    try:
        source = resolve_source(args.model, args.revision, model_info)
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from error
    common: dict[str, object] = {"trust_remote_code": False}
    if source["resolved_revision"] is not None:
        common["revision"] = source["resolved_revision"]
    tokenizer = AutoTokenizer.from_pretrained(source["load_target"], **common)
    model_kwargs: dict[str, object] = {
        **common,
        "torch_dtype": resolve_dtype(args.dtype, torch),
        "low_cpu_mem_usage": True,
    }
    if args.device == "auto":
        model_kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(source["load_target"], **model_kwargs)
    if args.device != "auto":
        model = model.to(args.device)
    model.eval()

    if not args.plain_prompt and getattr(tokenizer, "chat_template", None):
        input_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": args.prompt}],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        template_mode = "tokenizer_chat_template"
    else:
        input_ids = tokenizer(
            args.prompt, return_tensors="pt", add_special_tokens=True
        )["input_ids"]
        template_mode = "plain_prompt"
    input_device = model.get_input_embeddings().weight.device
    if input_device.type == "meta":
        raise SystemExit("input embedding device is unresolved (meta); choose an explicit device")
    input_ids = input_ids.to(input_device)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        raise SystemExit("tokenizer defines neither pad_token_id nor eos_token_id")
    generation = {
        "do_sample": False,
        "num_beams": 1,
        "use_cache": True,
        "max_new_tokens": args.tokens,
        "pad_token_id": int(pad_token_id),
    }
    with torch.inference_mode():
        generated = model.generate(input_ids=input_ids, **generation)

    prompt_ids = [int(value) for value in input_ids[0].detach().cpu().tolist()]
    full_ids = [int(value) for value in generated[0].detach().cpu().tolist()]
    if full_ids[:len(prompt_ids)] != prompt_ids or len(full_ids) <= len(prompt_ids):
        raise SystemExit("Transformers did not produce a valid prompt+continuation sequence")
    actual_new_tokens = len(full_ids) - len(prompt_ids)
    if len(full_ids) > MAX_TOTAL_TOKENS:
        raise SystemExit(f"reference exceeds {MAX_TOTAL_TOKENS} total tokens")
    if actual_new_tokens > MAX_NEW_TOKENS:
        raise SystemExit(f"reference exceeds {MAX_NEW_TOKENS} continuation tokens")

    payload = {
        "schema_version": 2,
        "generator": "transformers-greedy",
        "model": args.model,
        "source_kind": source["source_kind"],
        "requested_revision": args.revision,
        "resolved_revision": source["resolved_revision"],
        "source_fingerprint": source["source_fingerprint"],
        "versions": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "transformers": str(transformers.__version__),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "device_request": args.device,
        "dtype_request": args.dtype,
        "template_mode": template_mode,
        "prompt_sha256": sha256_text(args.prompt),
        "requested_new_tokens": args.tokens,
        "actual_new_tokens": actual_new_tokens,
        "generation": {
            "do_sample": False,
            "num_beams": 1,
            "use_cache": True,
            "max_new_tokens": args.tokens,
            "pad_token_id": int(pad_token_id),
        },
        "prompt_ids": prompt_ids,
        "full_ids": full_ids,
    }
    try:
        write_once_json(Path(args.output), payload)
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print(json.dumps({
        "output": str(Path(args.output).expanduser().resolve()),
        "source_kind": source["source_kind"],
        "resolved_revision": source["resolved_revision"],
        "source_fingerprint": source["source_fingerprint"],
        "prompt_tokens": len(prompt_ids),
        "continuation_tokens": actual_new_tokens,
        "template_mode": template_mode,
    }, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
