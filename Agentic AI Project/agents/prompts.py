"""Prompt loading.

Prompts live in ``agents/prompts/*.md``, not as f-strings in node code. Three
reasons, all of which bite in practice:

* **Diffable.** A prompt change shows as a prose diff in review, which is
  where a subtle instruction change should be argued about.
* **Tunable without a redeploy.** Prompt engineering is iterative, and
  rebuilding a container to reword a sentence discourages the iteration.
* **Auditable.** When a trade goes wrong, "which prompt produced this?" is
  answerable from git history.

``_shared.md`` prefixes every role, so ground rules are stated once rather
than drifting between seven copies.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

__all__ = ["PromptNotFoundError", "available_prompts", "load_prompt"]

_PROMPT_DIR = Path(__file__).parent / "prompts"
_SHARED = "_shared"


class PromptNotFoundError(FileNotFoundError):
    """No prompt file exists for the requested role."""


@lru_cache(maxsize=32)
def load_prompt(role: str) -> str:
    """Return the system prompt for ``role``, prefixed with the shared rules.

    Cached: a prompt is read once per process. Restart to pick up an edit, or
    call ``load_prompt.cache_clear()`` while iterating.
    """
    path = _PROMPT_DIR / f"{role}.md"
    if not path.is_file():
        raise PromptNotFoundError(
            f"no prompt for role {role!r} at {path}. Available: {', '.join(available_prompts())}"
        )

    shared_path = _PROMPT_DIR / f"{_SHARED}.md"
    shared = shared_path.read_text(encoding="utf-8").strip() if shared_path.is_file() else ""
    body = path.read_text(encoding="utf-8").strip()
    return f"{shared}\n\n---\n\n{body}" if shared else body


def available_prompts() -> list[str]:
    return sorted(p.stem for p in _PROMPT_DIR.glob("*.md") if not p.stem.startswith("_"))
