"""Chuyển events.csv của Retailrocket sang file cặp user-item của khung BIPN."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

EVENTS = {"view": "view", "addtocart": "cart", "transaction": "buy"}
COLUMNS = ("timestamp", "visitorid", "event", "itemid")


def convert(events_path, out_dir):
    events_path = Path(events_path)
    out_dir = Path(out_dir)
    if not events_path.is_file():
        raise FileNotFoundError(f"Thiếu {events_path}")
    buys = _read_buys(events_path)
    if not buys:
        raise ValueError("Không có sự kiện transaction.")
    split = {visitor: _split_buys(rows) for visitor, rows in buys.items()}
    views, carts = _read_auxiliary(events_path, split)
    user_map, item_map = _remap(split, views, carts)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_count(out_dir / "count.txt", len(user_map), len(item_map))
    _write_behavior(out_dir / "view.txt", views, user_map, item_map)
    _write_behavior(out_dir / "cart.txt", carts, user_map, item_map)
    _write_buys(out_dir / "buy.txt", split, "train", user_map, item_map)
    _write_buys(out_dir / "validation.txt", split, "valid", user_map, item_map)
    _write_buys(out_dir / "test.txt", split, "test", user_map, item_map)
    return {"users": len(user_map), "items": len(item_map)}


def _read_buys(path):
    buys = defaultdict(list)
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        _check_header(reader.fieldnames, path)
        for line_no, row in enumerate(reader, 2):
            event = row["event"].strip()
            if event not in EVENTS:
                raise ValueError(f"{path}:{line_no} có event không dùng được: {event}")
            if event != "transaction":
                continue
            visitor, item, time = _cells(row, path, line_no)
            buys[visitor].append((time, item))
    return buys


def _read_auxiliary(path, split):
    views = defaultdict(set)
    carts = defaultdict(set)
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for line_no, row in enumerate(reader, 2):
            event = row["event"].strip()
            if event not in EVENTS:
                raise ValueError(f"{path}:{line_no} có event không dùng được: {event}")
            visitor, item, time = _cells(row, path, line_no)
            if visitor not in split or time >= split[visitor]["cutoff"]:
                continue
            if event == "view":
                views[visitor].add(item)
            elif event == "addtocart":
                carts[visitor].add(item)
    return views, carts


def _split_buys(rows):
    ordered = []
    seen = set()
    for time, item in sorted(rows, key=lambda pair: pair[0]):
        if item in seen:
            continue
        seen.add(item)
        ordered.append((time, item))
    if len(ordered) >= 3:
        train, valid, test = ordered[:-2], [ordered[-2]], [ordered[-1]]
    elif len(ordered) == 2:
        train, valid, test = [ordered[0]], [], [ordered[1]]
    else:
        train, valid, test = [], [], [ordered[0]]
    held = valid + test
    return {"train": train, "valid": valid, "test": test, "cutoff": held[0][0]}


def _remap(split, views, carts):
    visitors = sorted(split)
    items = set()
    for visitor in visitors:
        for part in ("train", "valid", "test"):
            items.update(item for _, item in split[visitor][part])
        items.update(views[visitor])
        items.update(carts[visitor])
    user_map = {old: new for new, old in enumerate(visitors, 1)}
    item_map = {old: new for new, old in enumerate(sorted(items), 1)}
    return user_map, item_map


def _check_header(fieldnames, path):
    names = set(fieldnames or [])
    missing = [name for name in COLUMNS if name not in names]
    if missing:
        raise ValueError(f"{path} thiếu cột {', '.join(missing)}")


def _cells(row, path, line_no):
    try:
        visitor = int(row["visitorid"])
        item = int(row["itemid"])
        time = int(row["timestamp"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path}:{line_no} có ô không phải số nguyên") from exc
    if visitor < 0 or item < 0:
        raise ValueError(f"{path}:{line_no} có id âm")
    return visitor, item, time


def _write_count(path, n_users, n_items):
    path.write_text(json.dumps({"user": n_users, "item": n_items}), encoding="utf-8")


def _write_behavior(path, grouped, user_map, item_map):
    lines = []
    for visitor, items in grouped.items():
        user = user_map[visitor]
        for item in sorted(items):
            lines.append(f"{user} {item_map[item]}")
    _write_lines(path, lines)


def _write_buys(path, split, part, user_map, item_map):
    lines = []
    for visitor in sorted(split):
        user = user_map[visitor]
        for _, item in split[visitor][part]:
            lines.append(f"{user} {item_map[item]}")
    _write_lines(path, lines)


def _write_lines(path, lines):
    text = "\n".join(lines)
    if text:
        text += "\n"
    path.write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Chuyển Retailrocket sang định dạng BIPN")
    parser.add_argument("--events", required=True, help="Đường dẫn events.csv")
    parser.add_argument("--out", required=True, help="Thư mục đầu ra")
    args = parser.parse_args()
    summary = convert(args.events, args.out)
    print(f"users {summary['users']} items {summary['items']} -> {args.out}")


if __name__ == "__main__":
    main()
