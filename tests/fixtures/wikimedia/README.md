# Recorded Wikimedia responses

Real responses from Wikimedia's EventStreams and the MediaWiki Action API, captured on 2026-11-04 and scrubbed for this public repo. Tests read these files, so they never have to call Wikimedia.

`manifest.json` lists every file and how it was treated. `scripts/capture_fixtures.py` recorded the responses, and `scripts/scrub_fixtures.py` produced these copies from them. `tests/test_fixtures_clean.py` fails if a username, IP address, or email address gets in.

## What was changed

These apply to every file:

- Usernames are placeholders that keep their format: `ExampleUser12`, or `~2026-00016-01` for a temporary account. IP usernames use reserved documentation addresses such as `192.0.2.5`.
- User IDs are remapped.
- Emails are replaced with `user@example.org`.
- Summaries that could name another editor, such as revert messages, have their words replaced even in kept-text files.

Files marked `neutral` also have words replaced with `xxxx`, digits replaced with `0`, titles replaced with `Example article N`, and revision and page IDs remapped. The HTML, field names, value types, and text lengths are real, so parsers see the true structure. These files cover reverted and hidden edits, every stream event, and anything with text about living people that shouldn't be repeated here. They show format only, so don't treat them as examples of vandalism.

An ID stays consistent across neutral files, so the same edit can be followed from one response to another. Stream URLs all end in the same placeholder, so they don't match their event's `title`.

Files marked `ordinary` keep the real title, wikitext, and revision ID. Each is a settled edit that wasn't reverted or flagged, and a person read its text before it was kept.

## Attribution

Text in the `ordinary` files comes from English Wikipedia contributors and is available under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). The `source` link for each one in `manifest.json` gives the original edit and its page history.
