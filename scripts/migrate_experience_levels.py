"""Relabel saved interviews from year ranges ("3-5") to named levels ("Intermediate").

Usage (from the project folder):
    .venv/bin/python scripts/migrate_experience_levels.py           # dry run: show counts only
    .venv/bin/python scripts/migrate_experience_levels.py --apply   # update the documents
    .venv/bin/python scripts/migrate_experience_levels.py --undo    # restore the year ranges

The original value is kept in `legacy_experience_level`, so --undo is exact even though two
ranges map to the same level.
"""

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv
from pymongo import MongoClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]

LEVEL_FOR_RANGE = {
    "1-3": "Beginner",
    "3-5": "Intermediate",
    "5-7": "Proficient",
    "7-9": "Advanced",
    "9-12": "Advanced",
    "12+": "Expert",
}


def get_collection():
    load_dotenv(PROJECT_ROOT / ".env")
    missing = [
        name
        for name in ("MONGODB_URI", "MONGODB_DATABASE", "MONGODB_COLLECTION")
        if not os.getenv(name)
    ]
    if missing:
        sys.exit(f"Missing in .env: {', '.join(missing)}")
    client = MongoClient(os.environ["MONGODB_URI"], serverSelectionTimeoutMS=10000)
    return client[os.environ["MONGODB_DATABASE"]][os.environ["MONGODB_COLLECTION"]]


def level_counts(collection) -> Counter:
    return Counter(
        {
            row["_id"]: row["count"]
            for row in collection.aggregate(
                [{"$group": {"_id": "$experience_level", "count": {"$sum": 1}}}]
            )
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--apply", action="store_true", help="update the saved documents")
    mode.add_argument("--undo", action="store_true", help="restore the original year ranges")
    args = parser.parse_args()

    collection = get_collection()
    print("Before:", dict(level_counts(collection)))

    if args.undo:
        result = collection.update_many(
            {"legacy_experience_level": {"$exists": True}},
            [
                {"$set": {"experience_level": "$legacy_experience_level"}},
                {"$unset": "legacy_experience_level"},
            ],
        )
        print(f"Restored {result.modified_count} documents.")
    else:
        for old, new in LEVEL_FOR_RANGE.items():
            query = {"experience_level": old}
            if not args.apply:
                print(f"  would relabel {collection.count_documents(query)} x {old!r} -> {new!r}")
                continue
            result = collection.update_many(
                query,
                {"$set": {"experience_level": new, "legacy_experience_level": old}},
            )
            print(f"  relabeled {result.modified_count} x {old!r} -> {new!r}")
        if not args.apply:
            print("Dry run only. Re-run with --apply to update.")
            return

    print("After:", dict(level_counts(collection)))


if __name__ == "__main__":
    main()
