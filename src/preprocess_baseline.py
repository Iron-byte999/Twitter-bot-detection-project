from __future__ import annotations

import argparse
import csv
import json
import random
import re
import tarfile
from pathlib import Path
from typing import Any

from utils import data_path


DEFAULT_OUTPUT = data_path("botwiki_verified_2019.csv")
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


def load_records(
    archive_path: Path,
    tsv_name: str,
    json_name: str,
    keep_verified: bool,
) -> list[dict[str, Any]]:
    """Convert raw Twitter account records into train-compatible rows."""
    with tarfile.open(archive_path, "r:gz") as archive:
        labels = load_labels(archive, tsv_name)
        json_member = archive.extractfile(json_name)
        if json_member is None:
            raise FileNotFoundError(f"Missing {json_name}")
        raw_records = json.load(json_member)

    rows = []
    seen_user_ids = set()
    for record in raw_records:
        user = record.get("user", {})
        user_id = str(user.get("id_str") or user.get("id") or "")
        if not user_id or user_id in seen_user_ids or user_id not in labels:
            continue

        description = user.get("description") or ""
        hashtags = " ".join(re.findall(r"#\w+", description))
        rows.append(
            {
                "User ID": user_id,
                "Username": user.get("screen_name") or "",
                "Tweet": description,
                "Retweet Count": int(user.get("statuses_count") or 0),
                "Mention Count": int(user.get("friends_count") or 0),
                "Follower Count": int(user.get("followers_count") or 0),
                "Verified": bool(user.get("verified")) if keep_verified else False,
                "Bot Label": labels[user_id],
                "Location": user.get("location") or "",
                "Created At": user.get("created_at") or record.get("created_at") or "",
                "Hashtags": hashtags,
            }
        )
        seen_user_ids.add(user_id)
    return rows


def build_dataset(
    verified_archive: Path,
    botwiki_archive: Path,
    output_path: Path,
    keep_verified: bool,
) -> None:
    """Build the main combined 2019 CSV from verified-human and Botwiki-bot archives."""
    rows = []
    rows.extend(
        load_records(
            verified_archive,
            "verified-2019.tsv",
            "verified-2019_tweets.json",
            keep_verified,
        )
    )
    rows.extend(
        load_records(
            botwiki_archive,
            "botwiki-2019.tsv",
            "botwiki-2019_tweets.json",
            keep_verified,
        )
    )
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
    if not keep_verified:
        print("Verified was neutralized to avoid source-label leakage.")


def parse_args() -> argparse.Namespace:
    """Parse preprocessing command-line options."""
    parser = argparse.ArgumentParser(
        description="Convert Botwiki/verified 2019 archives into train-compatible CSV."
    )
    parser.add_argument("--verified-archive", type=Path, default=DEFAULT_VERIFIED)
    parser.add_argument("--botwiki-archive", type=Path, default=DEFAULT_BOTWIKI)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--keep-verified",
        action="store_true",
        help="Keep raw Twitter verified flags. Usually avoided because this dataset makes that feature leak the label.",
    )
    return parser.parse_args()


def main() -> None:
    """Run preprocessing using CLI arguments."""
    args = parse_args()
    build_dataset(
        args.verified_archive,
        args.botwiki_archive,
        args.output,
        args.keep_verified,
    )


if __name__ == "__main__":
    main()
