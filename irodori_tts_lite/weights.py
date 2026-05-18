"""Resolve checkpoint paths, optionally downloading from Hugging Face."""
from __future__ import annotations

import os
from pathlib import Path

DEFAULT_REPO = "kizuna-intelligence/Irodori-TTS-Lite-int4"
DEFAULT_DIT_FILE = "dit_int4.safetensors"
DEFAULT_DACVAE_FILE = "dacvae_int4.safetensors"


def _hf_download(repo_id: str, filename: str) -> str:
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id=repo_id, filename=filename)


def resolve_checkpoint(
    arg: str | None,
    *,
    default_filename: str = DEFAULT_DIT_FILE,
    default_repo: str = DEFAULT_REPO,
) -> str:
    """Return a local path to the checkpoint.

    Resolution order:
      - `hf://<repo>/<filename>` URI  → `hf_hub_download(repo, filename)`
      - None / empty                  → `hf_hub_download(default_repo, default_filename)`
      - Otherwise a local path        → returned as-is (must exist).
    """
    if not arg:
        return _hf_download(default_repo, default_filename)

    if arg.startswith("hf://"):
        spec = arg[len("hf://"):]
        repo_id, _, filename = spec.rpartition("/")
        if not repo_id or not filename:
            raise ValueError(
                f"Bad hf:// URI {arg!r} — expected hf://<org>/<repo>/<filename>"
            )
        return _hf_download(repo_id, filename)

    path = Path(os.path.expanduser(arg))
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    return str(path)
