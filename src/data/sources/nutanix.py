from typing import List
from datasets import load_dataset

from ..schema import RawRecord

SOURCE_NAME = "nutanix_code_suggestions"

def fetch_nutanix_code_suggestions(
    data_file: str = "code_suggestions.csv",
    split: str = "train",
) -> List[RawRecord]:
   
    ds = load_dataset("Nutanix/codereview-dataset", data_files=data_file, split=split)

    out: List[RawRecord] = []
    dropped = 0
    for ex in ds:
        try:
            code = (ex["existing_code_snippet"] or "").strip()
            comment = (ex["content"] or "").strip()
        except KeyError as e:
            raise ValueError(f"Dataset missing expected column: {e}")

        if not code or not comment:
            dropped += 1
            continue

        out.append(RawRecord(source=SOURCE_NAME, code=code, comment=comment))

    if dropped:
        print(f"[nutanix] dropped {dropped} empty records")

    return out
