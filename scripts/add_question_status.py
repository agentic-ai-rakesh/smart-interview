"""Give every saved question a permanent `question_id` and the default "Unchecked" status.

Usage (from the project folder):
    .venv/bin/python scripts/add_question_status.py           # dry run: show counts only
    .venv/bin/python scripts/add_question_status.py --apply   # update the documents

Only missing fields are added: an existing question_id or status (e.g. "Checked") is kept,
so running it again is safe.
"""

import argparse
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from migrate_experience_levels import get_collection  # noqa: E402

DEFAULT_STATUS = "Unchecked"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--apply", action="store_true", help="update the saved documents")
    args = parser.parse_args()

    collection = get_collection()
    documents = questions = missing_id = missing_status = 0

    for document in collection.find({}, {"questions": 1}):
        changed = False
        updated_questions = []
        for question in document.get("questions", []):
            questions += 1
            question = dict(question)
            if not question.get("question_id"):
                question["question_id"] = uuid.uuid4().hex
                missing_id += 1
                changed = True
            if question.get("status") not in ("Unchecked", "Checked"):
                question["status"] = DEFAULT_STATUS
                missing_status += 1
                changed = True
            updated_questions.append(question)

        if changed:
            documents += 1
            if args.apply:
                collection.update_one(
                    {"_id": document["_id"]}, {"$set": {"questions": updated_questions}}
                )

    verb = "Updated" if args.apply else "Would update"
    print(
        f"{verb} {documents} saved interviews: {questions} questions checked, "
        f"{missing_id} given a question_id, {missing_status} set to {DEFAULT_STATUS!r}."
    )
    if not args.apply:
        print("Dry run only. Re-run with --apply to update.")


if __name__ == "__main__":
    main()
