#!/usr/bin/env python3
"""Translate the keys listed by `ja_sync.py todo` into Japanese with the Gemini API.

Usage:
  GEMINI_API_KEY=... python3 scripts/translate_ja_gemini.py TODO.json RESULT.json

Writes RESULT.json in the shape `ja_sync.py apply` expects:
  { "<ja.json path>": { "<dotted.key>": "日本語訳" } }

Environment:
  GEMINI_API_KEY   required
  GEMINI_MODEL     model id (default: gemini-flash-latest)
  BATCH_SIZE       keys per request (default: 60)
  --dry-run        print the first prompt instead of calling the API
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
PLACEHOLDER_RE = re.compile(r"\{\{.*?\}\}|\{[^{}]+\}")
MAX_EXAMPLES_PER_KEY = 6

INSTRUCTIONS = """\
You are translating the UI of a game-server management panel (a web control panel for
hosting game servers) from English into natural Japanese.

Rules:
- Keep every placeholder exactly as written in English, e.g. {count}, {name}, {{ settings.app.name }}.
  Do not translate, rename, add or remove placeholders.
- Keep HTML tags and Markdown intact.
- Keys ending in .zero/.one/.two/.few/.many/.other are plural forms. Japanese normally uses
  the same text for all of them.
- Match the terminology and tone of the reference translations below (existing translations
  of nearby keys). Use the same words for the same concepts.
- Items with "old_en" and "current_ja" had their English text changed. Update current_ja so it
  matches the new "en"; keep it as is if it already fits.
- Return one result for every item, using the item's "file" and "key" unchanged.
"""

RESPONSE_SCHEMA = {
    "type": "ARRAY",
    "items": {
        "type": "OBJECT",
        "properties": {
            "file": {"type": "STRING"},
            "key": {"type": "STRING"},
            "ja": {"type": "STRING"},
        },
        "required": ["file", "key", "ja"],
    },
}


def flatten(data: dict, prefix: str = "") -> dict[str, object]:
    flat: dict[str, object] = {}
    for key, value in data.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(flatten(value, path))
        else:
            flat[path] = value
    return flat


def load_flat(path: str) -> dict[str, object]:
    return flatten(json.loads((REPO_ROOT / path).read_text(encoding="utf-8")))


def reference_translations(keys: list[str], en: dict, ja: dict) -> dict[str, dict[str, object]]:
    """Existing translations of sibling keys (same parent prefix), for terminology."""
    refs: dict[str, dict[str, object]] = {}
    for key in keys:
        parent = key.rsplit(".", 1)[0]
        count = 0
        for other, ja_value in ja.items():
            if other in keys or other in refs or not other.startswith(parent + "."):
                continue
            if other in en:
                refs[other] = {"en": en[other], "ja": ja_value}
                count += 1
                if count >= MAX_EXAMPLES_PER_KEY:
                    break
    return refs


def build_prompt(file: str, batch: list[tuple[str, dict]], en: dict, ja: dict) -> str:
    items = []
    for key, entry in batch:
        item = {"file": file, "key": key, "en": entry["en"]}
        if "old_en" in entry:
            item["old_en"] = entry["old_en"]
            item["current_ja"] = entry.get("ja")
        items.append(item)
    refs = reference_translations([k for k, _ in batch], en, ja)
    return (
        INSTRUCTIONS
        + "\nReference translations (key: en → ja):\n"
        + json.dumps(refs, ensure_ascii=False, indent=1)
        + "\n\nItems to translate:\n"
        + json.dumps(items, ensure_ascii=False, indent=1)
    )


def call_gemini(prompt: str, api_key: str, model: str) -> list[dict]:
    body = json.dumps(
        {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseSchema": RESPONSE_SCHEMA,
                "temperature": 0.2,
            },
        }
    ).encode()
    request = urllib.request.Request(
        API_URL.format(model=model),
        data=body,
        headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
        method="POST",
    )
    delay = 10
    for attempt in range(1, 6):
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                data = json.load(response)
            text = "".join(p.get("text", "") for p in data["candidates"][0]["content"]["parts"])
            return json.loads(text)
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:500]
            if e.code not in (429, 500, 502, 503, 504) or attempt == 5:
                raise SystemExit(f"Gemini API error {e.code}: {detail}")
            print(f"  HTTP {e.code}, retrying in {delay}s", file=sys.stderr)
        except (KeyError, IndexError, json.JSONDecodeError, urllib.error.URLError, TimeoutError) as e:
            if attempt == 5:
                raise SystemExit(f"Gemini API returned an unusable response: {e!r}")
            print(f"  {e!r}, retrying in {delay}s", file=sys.stderr)
        time.sleep(delay)
        delay *= 2
    raise AssertionError("unreachable")


def placeholders(text: object) -> list[str]:
    return sorted(set(PLACEHOLDER_RE.findall(text))) if isinstance(text, str) else []


def main(argv: list[str]) -> int:
    dry_run = "--dry-run" in argv
    args = [a for a in argv if a != "--dry-run"]
    if len(args) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    todo_path, result_path = args

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key and not dry_run:
        print("error: GEMINI_API_KEY is not set", file=sys.stderr)
        return 1
    model = os.environ.get("GEMINI_MODEL") or "gemini-flash-latest"
    batch_size = int(os.environ.get("BATCH_SIZE") or 60)

    todo: dict[str, dict[str, dict]] = json.loads(Path(todo_path).read_text(encoding="utf-8"))
    result: dict[str, dict[str, str]] = {}
    skipped = 0

    for file, entries in todo.items():
        en = load_flat(str(Path(file).with_name("en.json")))
        ja = load_flat(file)
        pending = list(entries.items())
        for start in range(0, len(pending), batch_size):
            batch = pending[start : start + batch_size]
            prompt = build_prompt(file, batch, en, ja)
            if dry_run:
                print(prompt)
                return 0
            print(f"{file}: translating {start + 1}-{start + len(batch)} of {len(pending)} ({model})")
            wanted = dict(batch)
            for item in call_gemini(prompt, api_key, model):
                key, value = item.get("key"), item.get("ja")
                if item.get("file") != file or key not in wanted or not isinstance(value, str):
                    continue
                if placeholders(value) != placeholders(wanted[key]["en"]):
                    print(f"  skip {key}: placeholder mismatch ({value!r})", file=sys.stderr)
                    skipped += 1
                    continue
                result.setdefault(file, {})[key] = value

    Path(result_path).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    done = sum(len(v) for v in result.values())
    total = sum(len(v) for v in todo.values())
    print(f"translated {done}/{total} key(s), {skipped} skipped for placeholder mismatch")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
