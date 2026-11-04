"""A small list of words common in vandalism, for the badword_count feature.

This is one weak signal among many, so it stays short on purpose. It holds common profanity and
schoolyard vandalism words, and leaves out slurs so the public repo doesn't carry them. The
transformers learn the rest from the text itself.
"""

BADWORDS = frozenset(
    {
        "arse",
        "ass",
        "asshole",
        "bastard",
        "bitch",
        "bollocks",
        "boob",
        "boobs",
        "bullshit",
        "butt",
        "crap",
        "damn",
        "dick",
        "dumb",
        "fart",
        "fuck",
        "fucked",
        "fucker",
        "fucking",
        "hahaha",
        "idiot",
        "loser",
        "lol",
        "moron",
        "penis",
        "piss",
        "poop",
        "pussy",
        "sex",
        "sexy",
        "shit",
        "shitty",
        "slut",
        "stupid",
        "sucks",
        "tits",
        "vagina",
        "wanker",
        "whore",
    }
)
