#!/usr/bin/env python3
"""Compare en.json and ja.json: missing keys, extra keys, and key order.

Usage:
  python3 scripts/find_missing_ja_keys.py
  python3 scripts/find_missing_ja_keys.py --en path/to/en.json --ja path/to/ja.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EN = REPO_ROOT / "frontend" / "public" / "translations" / "en.json"
DEFAULT_JA = REPO_ROOT / "frontend" / "public" / "translations" / "ja.json"

MAX_ORDER_KEYS = 12


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a JSON object at the root")
    return data


def flatten(data: dict, prefix: str = "") -> dict[str, object]:
    flat: dict[str, object] = {}
    for key, value in data.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(flatten(value, path))
        else:
            flat[path] = value
    return flat


def join_path(prefix: str, key: str) -> str:
    return f"{prefix}.{key}" if prefix else key


def format_key_list(keys: list[str]) -> str:
    if len(keys) <= MAX_ORDER_KEYS:
        shown = keys
        suffix = ""
    else:
        shown = keys[:MAX_ORDER_KEYS]
        suffix = f", ... (+{len(keys) - MAX_ORDER_KEYS})"
    return "[" + ", ".join(shown) + suffix + "]"


def first_order_divergence(en_keys: list[str], ja_keys: list[str]) -> str:
    for index, (en_key, ja_key) in enumerate(zip(en_keys, ja_keys)):
        if en_key != ja_key:
            return f"index {index}: expected {en_key!r}, found {ja_key!r}"
    if len(en_keys) != len(ja_keys):
        return f"shared key count differs: en={len(en_keys)} ja={len(ja_keys)}"
    return "order differs"


def collect_order_mismatches(
    en: dict,
    ja: dict,
    prefix: str = "",
) -> list[tuple[str, list[str], list[str]]]:
    mismatches: list[tuple[str, list[str], list[str]]] = []

    en_common = [key for key in en if key in ja]
    ja_common = [key for key in ja if key in en]
    if en_common != ja_common:
        mismatches.append((prefix or "<root>", en_common, ja_common))

    for key in en:
        if key not in ja:
            continue
        en_value = en[key]
        ja_value = ja[key]
        if isinstance(en_value, dict) and isinstance(ja_value, dict):
            mismatches.extend(collect_order_mismatches(en_value, ja_value, join_path(prefix, key)))

    return mismatches


def print_section(title: str) -> None:
    print(f"=== {title} ===")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Compare en.json and ja.json for missing keys, extra keys, and key order.",
    )
    parser.add_argument("--en", type=Path, default=DEFAULT_EN, help="path to en.json")
    parser.add_argument("--ja", type=Path, default=DEFAULT_JA, help="path to ja.json")
    parser.add_argument(
        "--keys-only",
        action="store_true",
        help="print only keys, without English/Japanese values",
    )
    parser.add_argument(
        "--missing-only",
        action="store_true",
        help="report only keys missing from ja.json",
    )
    parser.add_argument(
        "--order-only",
        action="store_true",
        help="report only key-order mismatches",
    )
    args = parser.parse_args(argv)

    if args.missing_only and args.order_only:
        print("error: --missing-only and --order-only cannot be used together", file=sys.stderr)
        return 2

    if not args.en.exists():
        print(f"error: en.json not found: {args.en}", file=sys.stderr)
        return 1
    if not args.ja.exists():
        print(f"error: ja.json not found: {args.ja}", file=sys.stderr)
        return 1

    en_data = load_json(args.en)
    ja_data = load_json(args.ja)
    en_flat = flatten(en_data)
    ja_flat = flatten(ja_data)

    missing = [(key, en_flat[key]) for key in en_flat if key not in ja_flat]
    extra = [(key, ja_flat[key]) for key in ja_flat if key not in en_flat]
    order_mismatches = [] if args.missing_only else collect_order_mismatches(en_data, ja_data)

    show_missing = not args.order_only
    show_extra = not args.missing_only and not args.order_only
    show_order = not args.missing_only

    printed = False

    if show_missing:
        print_section(f"Missing from ja.json ({len(missing)})")
        if missing:
            for key, value in missing:
                if args.keys_only:
                    print(key)
                else:
                    print(f"{key}\t{json.dumps(value, ensure_ascii=False)}")
        else:
            print("(none)")
        printed = True

    if show_extra:
        if printed:
            print()
        print_section(f"Extra in ja.json ({len(extra)})")
        if extra:
            for key, value in extra:
                if args.keys_only:
                    print(key)
                else:
                    print(f"{key}\t{json.dumps(value, ensure_ascii=False)}")
        else:
            print("(none)")
        printed = True

    if show_order:
        if printed:
            print()
        print_section(f"Key order mismatches ({len(order_mismatches)})")
        if order_mismatches:
            for path, en_keys, ja_keys in order_mismatches:
                print(path)
                print(f"  en: {format_key_list(en_keys)}")
                print(f"  ja: {format_key_list(ja_keys)}")
                print(f"  {first_order_divergence(en_keys, ja_keys)}")
        else:
            print("(none)")

    print(
        f"\n{len(missing)} missing, {len(extra)} extra, {len(order_mismatches)} order mismatch(es)",
        file=sys.stderr,
    )
    return 1 if missing or extra or order_mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
