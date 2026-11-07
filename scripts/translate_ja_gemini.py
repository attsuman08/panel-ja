#!/usr/bin/env python3
"""Translate the keys listed by `ja_sync.py todo` into Japanese with the Gemini API.

Usage:
  GEMINI_API_KEY=... python3 scripts/translate_ja_gemini.py TODO.json RESULT.json

Writes RESULT.json in the shape `ja_sync.py apply` expects:
  { "<ja.json path>": { "<dotted.key>": "日本語訳" } }

Environment:
  GEMINI_API_KEY   required
  GEMINI_MODEL     comma-separated model ids, tried in order; on overload (429/5xx/timeout)
                   the next one is used (default: gemini-3.6-flash,gemini-flash-latest,gemini-3.5-flash-lite)
  BATCH_SIZE       keys per request (default: 30)
  REQUEST_TIMEOUT  seconds to wait for one response (default: 120)
  --dry-run        print the first prompt instead of calling the API
"""

from __future__ import annotations

import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
PLACEHOLDER_RE = re.compile(r"\{\{.*?\}\}|\{[^{}]+\}")
MAX_EXAMPLES_PER_KEY = 6
DEFAULT_MODELS = "gemini-3.8-flash,gemini-3.6-flash,gemini-3.5-flash-lite"
MAX_ATTEMPTS = 6
ATTEMPTS_PER_MODEL = 1
HEARTBEAT_SECONDS = 30
START = time.monotonic()


def log(message: str) -> None:
    elapsed = int(time.monotonic() - START)
    print(f"[{time.strftime('%H:%M:%S')} +{elapsed // 60}m{elapsed % 60:02d}s] {message}", flush=True)


class BatchFailed(Exception):
    pass

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


def post_with_heartbeat(request: urllib.request.Request, timeout: int) -> dict:
    """POST and return the JSON body, logging every HEARTBEAT_SECONDS while waiting."""
    done = threading.Event()
    started = time.monotonic()

    def heartbeat() -> None:
        while not done.wait(HEARTBEAT_SECONDS):
            log(f"    still waiting for Gemini ({int(time.monotonic() - started)}s / timeout {timeout}s)")

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    finally:
        done.set()


class ModelPool:
    """Rotates through models: a model that keeps failing is swapped for the next one,
    a model the API rejects (400/404) is dropped, and the last model that worked is
    reused for the next batch."""

    def __init__(self, models: list[str]):
        self.models = models
        self.index = 0
        self.failures = 0

    @property
    def current(self) -> str:
        if not self.models:
            raise BatchFailed("no usable model left")
        return self.models[self.index % len(self.models)]

    def failed(self) -> None:
        self.failures += 1
        if self.failures >= ATTEMPTS_PER_MODEL and len(self.models) > 1:
            old = self.current
            self.index = (self.index + 1) % len(self.models)
            self.failures = 0
            log(f"  switching model: {old} -> {self.current}")

    def drop(self) -> None:
        old = self.current
        self.models.remove(old)
        self.failures = 0
        if self.models:
            self.index %= len(self.models)
            log(f"  dropping model {old}; next: {self.current}")

    def succeeded(self) -> None:
        self.failures = 0


def call_gemini(prompt: str, api_key: str, pool: ModelPool, timeout: int) -> list[dict]:
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
    delay = 10
    for attempt in range(1, MAX_ATTEMPTS + 1):
        model = pool.current
        request = urllib.request.Request(
            API_URL.format(model=model),
            data=body,
            headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
            method="POST",
        )
        log(f"  request attempt {attempt}/{MAX_ATTEMPTS} model={model} ({len(body) // 1024} KB)")
        started = time.monotonic()
        try:
            data = post_with_heartbeat(request, timeout)
            took = time.monotonic() - started
            candidate = data["candidates"][0]
            usage = data.get("usageMetadata", {})
            log(
                f"  response in {took:.1f}s: finishReason={candidate.get('finishReason')}"
                f" tokens(prompt={usage.get('promptTokenCount')}, output={usage.get('candidatesTokenCount')},"
                f" thinking={usage.get('thoughtsTokenCount', 0)})"
            )
            text = "".join(p.get("text", "") for p in candidate["content"]["parts"])
            items = json.loads(text)
            pool.succeeded()
            return items
        except urllib.error.HTTPError as e:
            detail = " ".join(e.read().decode(errors="replace").split())[:300]
            log(f"  HTTP {e.code} after {time.monotonic() - started:.1f}s: {detail}")
            if e.code in (400, 404):
                # Unknown model id, or a model that rejects this request: try the next one now.
                pool.drop()
                continue
            if e.code not in (429, 500, 502, 503, 504):
                raise BatchFailed(f"HTTP {e.code} (not retryable)")
            pool.failed()
        except (KeyError, IndexError, json.JSONDecodeError, urllib.error.URLError, TimeoutError, OSError) as e:
            log(f"  {e!r} after {time.monotonic() - started:.1f}s")
            pool.failed()
        if attempt == MAX_ATTEMPTS:
            break
        log(f"  retrying in {delay}s")
        time.sleep(delay)
        delay = min(delay * 2, 60)
    raise BatchFailed(f"gave up after {MAX_ATTEMPTS} attempts")


def placeholders(text: object) -> list[str]:
    return sorted(set(PLACEHOLDER_RE.findall(text))) if isinstance(text, str) else []


def write_result(path: str, result: dict) -> None:
    Path(path).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


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
    models = [m.strip() for m in (os.environ.get("GEMINI_MODEL") or DEFAULT_MODELS).split(",") if m.strip()]
    pool = ModelPool(models)
    batch_size = int(os.environ.get("BATCH_SIZE") or 30)
    timeout = int(os.environ.get("REQUEST_TIMEOUT") or 120)

    todo: dict[str, dict[str, dict]] = json.loads(Path(todo_path).read_text(encoding="utf-8"))
    result: dict[str, dict[str, str]] = {}
    skipped = 0
    failed_batches = 0
    total = sum(len(v) for v in todo.values())
    batches = sum((len(v) + batch_size - 1) // batch_size for v in todo.values())
    log(f"{total} key(s) in {batches} request(s), models={",".join(models)}, batch_size={batch_size}, timeout={timeout}s")
    batch_no = 0

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
            batch_no += 1
            log(f"[{batch_no}/{batches}] {file}: keys {start + 1}-{start + len(batch)} of {len(pending)}")
            wanted = dict(batch)
            try:
                items = call_gemini(prompt, api_key, pool, timeout)
            except BatchFailed as e:
                print(f"::warning::batch {batch_no}/{batches} failed: {e}; these keys stay untranslated", flush=True)
                failed_batches += 1
                continue
            got = 0
            for item in items:
                key, value = item.get("key"), item.get("ja")
                if item.get("file") != file or key not in wanted or not isinstance(value, str):
                    continue
                if placeholders(value) != placeholders(wanted[key]["en"]):
                    log(f"  skip {key}: placeholder mismatch ({value!r})")
                    skipped += 1
                    continue
                result.setdefault(file, {})[key] = value
                got += 1
            log(f"  accepted {got}/{len(batch)}")
            # Save after every batch so a cancelled/timed-out step keeps what it has.
            write_result(result_path, result)

    write_result(result_path, result)
    done = sum(len(v) for v in result.values())
    log(
        f"translated {done}/{total} key(s); {skipped} skipped for placeholder mismatch;"
        f" {failed_batches}/{batches} request(s) failed"
    )
    # Partial results are still worth applying; fail only when nothing came back.
    return 1 if failed_batches == batches else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
