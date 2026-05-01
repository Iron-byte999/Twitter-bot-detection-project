from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Any

from preprocess_final import (
    DEFAULT_BOTWIKI,
    DEFAULT_VERIFIED,
    FIELDNAMES,
    RANDOM_STATE,
    distance,
    load_records,
    scale_features,
)
from utils import data_path


DEFAULT_OUTPUT = data_path("botwiki_verified_2019_strict.csv")


def match_pairs(records: list[dict[str, Any]]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Pair each bot with the nearest available human account."""
    rng = random.Random(RANDOM_STATE)
    bots = [record for record in records if record["Bot Label"] == 1]
    available_humans = [record for record in records if record["Bot Label"] == 0]
    stats = scale_features(records)

    rng.shuffle(bots)
    pairs = []
    for bot in bots:
        best_index = min(
            range(len(available_humans)),
            key=lambda index: distance(bot, available_humans[index], stats),
        )
        pairs.append((bot, available_humans.pop(best_index)))
    return pairs


def neutralized_pair_rows(
    bot: dict[str, Any],
    human: dict[str, Any],
) -> list[dict[str, Any]]:
    """Create two rows with different labels but identical model inputs."""
    retweet_count = round((bot["Retweet Count"] + human["Retweet Count"]) / 2)
    mention_count = round((bot["Mention Count"] + human["Mention Count"]) / 2)
    follower_count = round((bot["Follower Count"] + human["Follower Count"]) / 2)

    shared_values = {
        "Username": "account",
        "Tweet": "profile",
        "Retweet Count": retweet_count,
        "Mention Count": mention_count,
        "Follower Count": follower_count,
        "Verified": False,
        "Location": "",
        "Created At": "Wed Jun 05 00:00:00 +0000 2019",
        "Hashtags": "none",
    }

    return [
        {
            "User ID": f"bot_{bot['User ID']}",
            **shared_values,
            "Bot Label": 1,
        },
        {
            "User ID": f"human_{human['User ID']}",
            **shared_values,
            "Bot Label": 0,
        },
    ]


def build_dataset(verified_archive: Path, botwiki_archive: Path, output_path: Path) -> None:
    """Build the strict leakage-check CSV from matched bot/human pairs."""
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

    rows = []
    for bot, human in match_pairs(records):
        rows.extend(neutralized_pair_rows(bot, human))

    random.Random(RANDOM_STATE).shuffle(rows)
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
    print("Strict mode: every bot row has a paired human row with identical model inputs.")


def parse_args() -> argparse.Namespace:
    """Parse strict leakage-check preprocessing command-line options."""
    parser = argparse.ArgumentParser(
        description="Create a strict leakage-check CSV for train_final.py."
    )
    parser.add_argument("--verified-archive", type=Path, default=DEFAULT_VERIFIED)
    parser.add_argument("--botwiki-archive", type=Path, default=DEFAULT_BOTWIKI)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    """Run strict leakage-check preprocessing using CLI arguments."""
    args = parse_args()
    build_dataset(args.verified_archive, args.botwiki_archive, args.output)


if __name__ == "__main__":
    main()
