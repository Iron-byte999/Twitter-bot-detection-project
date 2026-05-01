from __future__ import annotations

import argparse
import csv
import json
import math
import random
import tarfile
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

from utils import data_path


DEFAULT_OUTPUT = data_path("botwiki_verified_2019_balanced.csv")
DEFAULT_VERIFIED = data_path("verified-2019.tar.gz")
DEFAULT_BOTWIKI = data_path("botwiki-2019.tar.gz")
RANDOM_STATE = 42

FIELDNAMES = [
    "User ID",
    "Username",
    "Tweet",
    "Retweet Count",
    "Mention Count",
    "Follower Count",
    "Verified",
    "Bot Label",
    "Location",
    "Created At",
    "Hashtags",
]

MATCH_FEATURES = [
    "statuses_count_log",
    "friends_count_log",
    "followers_count_log",
    "account_age_days",
]


def parse_twitter_date(value: str) -> datetime | None:
    """Parse a Twitter API date string into a UTC datetime."""
    if not value:
        return None
    try:
        parsed = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def load_labels(archive: tarfile.TarFile, member_name: str) -> dict[str, int]:
    """Load user IDs and binary labels from a TSV member inside an archive."""
    labels = {}
    member = archive.extractfile(member_name)
    if member is None:
        raise FileNotFoundError(f"Missing {member_name}")

    for raw_line in member.read().decode("utf-8").splitlines():
        if not raw_line.strip():
            continue
        user_id, label = raw_line.split("\t")
        labels[user_id] = 1 if label == "bot" else 0
    return labels


def load_records(archive_path: Path, tsv_name: str, json_name: str) -> list[dict[str, Any]]:
    """Load raw account records and neutralize obvious source-identifying fields."""
    with tarfile.open(archive_path, "r:gz") as archive:
        labels = load_labels(archive, tsv_name)
        json_member = archive.extractfile(json_name)
        if json_member is None:
            raise FileNotFoundError(f"Missing {json_name}")
        raw_records = json.load(json_member)

    records = []
    seen_user_ids = set()
    for record in raw_records:
        user = record.get("user", {})
        user_id = str(user.get("id_str") or user.get("id") or "")
        if not user_id or user_id in seen_user_ids or user_id not in labels:
            continue

        statuses_count = int(user.get("statuses_count") or 0)
        friends_count = int(user.get("friends_count") or 0)
        followers_count = int(user.get("followers_count") or 0)
        created_at = user.get("created_at") or record.get("created_at") or ""
        created_date = parse_twitter_date(created_at)
        collected_date = parse_twitter_date(record.get("created_at") or "")
        reference_date = collected_date or datetime(2019, 6, 5, tzinfo=timezone.utc)
        account_age_days = (
            max((reference_date - created_date).days, 0) if created_date else 0
        )

        records.append(
            {
                "User ID": user_id,
                "Username": "account",
                "Tweet": "profile",
                "Retweet Count": statuses_count,
                "Mention Count": friends_count,
                "Follower Count": followers_count,
                "Verified": False,
                "Bot Label": labels[user_id],
                "Location": "",
                "Created At": "Wed Jun 05 00:00:00 +0000 2019",
                "Hashtags": "none",
                "statuses_count_log": math.log1p(statuses_count),
                "friends_count_log": math.log1p(friends_count),
                "followers_count_log": math.log1p(followers_count),
                "account_age_days": account_age_days,
            }
        )
        seen_user_ids.add(user_id)
    return records


def scale_features(records: list[dict[str, Any]]) -> dict[str, tuple[float, float]]:
    """Compute mean and standard deviation for nearest-neighbor matching features."""
    stats = {}
    for feature in MATCH_FEATURES:
        values = [float(record[feature]) for record in records]
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        stats[feature] = (mean, math.sqrt(variance) or 1.0)
    return stats


def distance(left: dict[str, Any], right: dict[str, Any], stats: dict[str, tuple[float, float]]) -> float:
    """Compute standardized squared distance between two account records."""
    total = 0.0
    for feature in MATCH_FEATURES:
        mean, std = stats[feature]
        left_value = (float(left[feature]) - mean) / std
        right_value = (float(right[feature]) - mean) / std
        total += (left_value - right_value) ** 2
    return total


def match_humans_to_bots(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select one nearest human match for each bot and return a balanced dataset."""
    rng = random.Random(RANDOM_STATE)
    bots = [record for record in records if record["Bot Label"] == 1]
    available_humans = [record for record in records if record["Bot Label"] == 0]
    stats = scale_features(records)

    rng.shuffle(bots)
    matched_humans = []
    for bot in bots:
        best_index = min(
            range(len(available_humans)),
            key=lambda index: distance(bot, available_humans[index], stats),
        )
        matched_humans.append(available_humans.pop(best_index))

    balanced_records = bots + matched_humans
    rng.shuffle(balanced_records)
    return balanced_records


def to_csv_row(record: dict[str, Any]) -> dict[str, Any]:
    """Drop internal matching fields before writing a CSV row."""
    return {field: record[field] for field in FIELDNAMES}


def build_dataset(verified_archive: Path, botwiki_archive: Path, output_path: Path) -> None:
    """Build the bias-reduced balanced CSV from the raw 2019 archives."""
    records = []
    records.extend(
        load_records(
            verified_archive,
            "verified-2019.tsv",
            "verified-2019_tweets.json",
        )
    )
    records.extend(
        load_records(
            botwiki_archive,
            "botwiki-2019.tsv",
            "botwiki-2019_tweets.json",
        )
    )

    balanced_records = match_humans_to_bots(records)
    rows = [to_csv_row(record) for record in balanced_records]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    bot_count = sum(row["Bot Label"] == 1 for row in rows)
    human_count = len(rows) - bot_count
    print(f"Saved {len(rows)} rows to {output_path}")
    print(f"Humans: {human_count}")
    print(f"Bots: {bot_count}")
    print("Text, hashtags, username, location, created date, and verified flags were neutralized.")
    print("Humans were nearest-neighbor matched to bots on account count features.")


def parse_args() -> argparse.Namespace:
    """Parse bias-reduced preprocessing command-line options."""
    parser = argparse.ArgumentParser(
        description="Create a bias-reduced 2019 bot/human CSV for train_final.py."
    )
    parser.add_argument("--verified-archive", type=Path, default=DEFAULT_VERIFIED)
    parser.add_argument("--botwiki-archive", type=Path, default=DEFAULT_BOTWIKI)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    """Run bias-reduced preprocessing using CLI arguments."""
    args = parse_args()
    build_dataset(args.verified_archive, args.botwiki_archive, args.output)


if __name__ == "__main__":
    main()
