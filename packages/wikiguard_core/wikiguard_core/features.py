"""build_features(): the one place features are computed, for training and for serving.

Sharing this function is what prevents training/serving skew. It takes only an EditEvent, which
can't hold a tag or a label, so nothing from after the edit can reach a model.
"""

import re
import unicodedata
from dataclasses import asdict, dataclass

from wikiguard_core.badwords import BADWORDS
from wikiguard_core.schemas import EditEvent

# Bump when any feature changes. A model refuses to load against a different version.
FEATURE_VERSION = 1

# Invisible characters that hide a word from a model and from a human reader.
INVISIBLE = dict.fromkeys(
    [0x00AD, *range(0x200B, 0x2010), *range(0x202A, 0x202F), *range(0x2060, 0x2065), 0xFEFF]
)
WORD = re.compile(r"\w+")
EDIT_COUNT_MISSING = -1
EDIT_COUNT_BUCKET_TOPS = (0, 10, 100, 1000)


@dataclass(frozen=True)
class EditFeatures:
    feature_version: int
    added_text: str
    removed_text: str
    comment: str
    is_anonymous: bool
    user_edit_count_bucket: int
    bytes_delta: int
    is_minor: bool
    chars_added: int
    uppercase_ratio: float
    badword_count: int
    has_external_link: bool
    hour_of_day: int
    page_length: int

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def normalize_text(text: str) -> str:
    """Undo the tricks that disguise a word: full-width letters, ligatures, and hidden characters.

    Applied here so a disguised word and a plain one look the same to every model.
    """
    return unicodedata.normalize("NFKC", text).translate(INVISIBLE)


def edit_count_bucket(count: int | None) -> int:
    """Bucket 0 is a first edit and the last bucket is over 1000. Missing has its own value."""
    if count is None:
        return EDIT_COUNT_MISSING
    for bucket, top in enumerate(EDIT_COUNT_BUCKET_TOPS):
        if count <= top:
            return bucket
    return len(EDIT_COUNT_BUCKET_TOPS)


def count_badwords(text: str) -> int:
    # Word lookups in a set, so the time stays linear however hostile the text is.
    return sum(1 for word in WORD.findall(text.lower()) if word in BADWORDS)


def uppercase_share(text: str) -> float:
    letters = [char for char in text if char.isalpha()]
    return sum(char.isupper() for char in letters) / len(letters) if letters else 0.0


def build_features(edit: EditEvent) -> EditFeatures:
    added = normalize_text(edit.added_text)
    new_length = edit.new_length or 0
    return EditFeatures(
        feature_version=FEATURE_VERSION,
        added_text=added,
        removed_text=normalize_text(edit.removed_text),
        # A hidden comment and an empty one look the same to the model, so hiding can't leak.
        comment=normalize_text(edit.comment or ""),
        is_anonymous=edit.is_anonymous,
        user_edit_count_bucket=edit_count_bucket(edit.user_edit_count),
        bytes_delta=new_length - (edit.old_length or 0),
        is_minor=edit.is_minor,
        chars_added=len(added),
        uppercase_ratio=uppercase_share(added),
        badword_count=count_badwords(added),
        has_external_link="http://" in added.lower() or "https://" in added.lower(),
        hour_of_day=edit.timestamp.hour,
        page_length=new_length,
    )
