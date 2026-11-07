#!/usr/bin/env python3
"""Helpers for keeping ja.json files in sync with en.json after upstream merges.

Subcommands:
  resolve <ja.json>...   Resolve a git merge conflict on ja.json with a per-key
                         three-way merge (stage 1 = base, 2 = ours, 3 = theirs).
                         Our translation wins when both sides changed a key.
  normalize [ja.json]... Reorder keys to match en.json and drop keys that no
                         longer exist in en.json.
  todo --old-ref REF     Write the keys that need (re)translation as JSON:
                         keys missing from ja.json, plus keys whose English
                         text changed since REF.
  apply RESULT.json      Merge translations ({"<ja.json path>": {"<dotted.key>": "訳"}})
                         into the ja.json files, then normalize them.
  check [ja.json]...     Fail if keys are missing or placeholders differ.

With no ja.json arguments, every known translation file is processed.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
JA_FILES = [
    "frontend/public/translations/ja.json",
    "shared/mails/variables/ja.json",
]
PLACEHOLDER_RE = re.compile(r"\{\{.*?\}\}|\{[^{}]+\}")
MISSING = object()


def en_path_for(ja_path: str) -> str:
    return str(Path(ja_path).with_name("en.json"))


def git_show(spec: str) -> dict | None:
    result = subprocess.run(
        ["git", "show", spec], cwd=REPO_ROOT, capture_output=True, text=True
    )
    if result.returncode != 0:
        return None
    return json.loads(result.stdout)


def load(path: str) -> dict:
    return json.loads((REPO_ROOT / path).read_text(encoding="utf-8"))


def dump(path: str, data: dict) -> None:
    text = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    (REPO_ROOT / path).write_text(text, encoding="utf-8")


def flatten(data: dict, prefix: str = "") -> dict[str, object]:
    flat: dict[str, object] = {}
    for key, value in data.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(flatten(value, path))
        else:
            flat[path] = value
    return flat


def build_like_en(en: dict, pick, prefix: str = "") -> dict:
    """Rebuild a dict with en's structure/order, taking leaf values from pick()."""
    out: dict = {}
    for key, en_value in en.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(en_value, dict):
            child = build_like_en(en_value, pick, path)
            if child:
                out[key] = child
        else:
            value = pick(path)
            if value is not MISSING:
                out[key] = value
    return out


def cmd_resolve(files: list[str]) -> int:
    for ja in files:
        base = flatten(git_show(f":1:{ja}") or {})
        ours = flatten(git_show(f":2:{ja}") or {})
        theirs = flatten(git_show(f":3:{ja}") or {})
        en = load(en_path_for(ja))

        def pick(key: str):
            o = ours.get(key, MISSING)
            t = theirs.get(key, MISSING)
            b = base.get(key, MISSING)
            if o is MISSING:
                return t
            if t is MISSING:
                return o
            # Only theirs changed it (e.g. Crowdin updated a string we never touched).
            if o == b and t != b:
                return t
            return o

        dump(ja, build_like_en(en, pick))
        subprocess.run(["git", "add", ja], cwd=REPO_ROOT, check=True)
        print(f"resolved {ja}")
    return 0


def cmd_normalize(files: list[str]) -> int:
    for ja in files:
        flat = flatten(load(ja))
        en = load(en_path_for(ja))
        dump(ja, build_like_en(en, lambda k: flat.get(k, MISSING)))
    return 0


def cmd_todo(old_ref: str, output: str) -> int:
    todo: dict[str, dict] = {}
    for ja in JA_FILES:
        en = flatten(load(en_path_for(ja)))
        old_en = flatten(git_show(f"{old_ref}:{en_path_for(ja)}") or {})
        cur = flatten(load(ja))
        items = {}
        for key, en_value in en.items():
            if key not in cur:
                items[key] = {"en": en_value}
            elif key in old_en and old_en[key] != en_value:
                items[key] = {"en": en_value, "old_en": old_en[key], "ja": cur[key]}
        if items:
            todo[ja] = items
    Path(output).write_text(json.dumps(todo, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    count = sum(len(v) for v in todo.values())
    print(count)
    return 0


def cmd_apply(result_path: str) -> int:
    result = json.loads(Path(result_path).read_text(encoding="utf-8"))
    for ja, items in result.items():
        if ja not in JA_FILES:
            print(f"skipping unknown file {ja}", file=sys.stderr)
            continue
        flat = flatten(load(ja))
        en = flatten(load(en_path_for(ja)))
        for key, value in items.items():
            if key in en and isinstance(value, str):
                flat[key] = value
            else:
                print(f"skipping unknown key {ja}:{key}", file=sys.stderr)
        dump(ja, build_like_en(load(en_path_for(ja)), lambda k: flat.get(k, MISSING)))
    return 0


def cmd_check(files: list[str]) -> int:
    ok = True
    for ja in files:
        en = flatten(load(en_path_for(ja)))
        cur = flatten(load(ja))
        for key, en_value in en.items():
            if key not in cur:
                print(f"MISSING\t{ja}\t{key}")
                ok = False
                continue
            if isinstance(en_value, str) and isinstance(cur[key], str):
                want = sorted(set(PLACEHOLDER_RE.findall(en_value)))
                got = sorted(set(PLACEHOLDER_RE.findall(cur[key])))
                if want != got:
                    print(f"PLACEHOLDER\t{ja}\t{key}\ten={want} ja={got}")
                    ok = False
    return 0 if ok else 1


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("resolve", "normalize", "check"):
        p = sub.add_parser(name)
        p.add_argument("files", nargs="*")
    p = sub.add_parser("apply")
    p.add_argument("result")
    p = sub.add_parser("todo")
    p.add_argument("--old-ref", required=True)
    p.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    if args.cmd == "todo":
        return cmd_todo(args.old_ref, args.output)
    if args.cmd == "apply":
        return cmd_apply(args.result)
    if args.cmd == "resolve" and not args.files:
        parser.error("resolve needs the conflicted ja.json path(s)")
    files = args.files or JA_FILES
    return {"resolve": cmd_resolve, "normalize": cmd_normalize, "check": cmd_check}[args.cmd](files)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
