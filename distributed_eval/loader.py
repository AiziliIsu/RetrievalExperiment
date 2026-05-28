from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Dict, Generator, Iterable, List, Optional, Tuple


def _sanitize_json_text(text: str) -> str:
    # Remove ASCII control chars that commonly break JSON parsing.
    return "".join(ch for ch in text if ch in ("\t", "\n", "\r") or ord(ch) >= 32)


def _safe_json_loads(raw_obj: str, source: Path) -> Optional[Dict]:
    try:
        return json.loads(raw_obj)
    except json.JSONDecodeError:
        try:
            return json.loads(_sanitize_json_text(raw_obj))
        except json.JSONDecodeError as exc:
            print(f"WARNING: Skipping invalid JSON object in {source}: {exc}")
            return None


def _iter_json_objects(path: Path) -> Generator[Dict, None, None]:
    """
    Stream JSON objects from:
    - single JSON object
    - JSON array
    - JSONL
    - concatenated / multiline JSON objects
    """
    decoder_buffer = []
    in_string = False
    escape = False
    depth = 0
    started = False

    with path.open("r", encoding="utf-8") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break

            for ch in chunk:
                if started:
                    decoder_buffer.append(ch)

                if in_string:
                    if escape:
                        escape = False
                    elif ch == "\\":
                        escape = True
                    elif ch == '"':
                        in_string = False
                    continue

                if ch == '"':
                    in_string = True
                    continue

                if ch == "{":
                    if not started:
                        started = True
                        decoder_buffer = ["{"]
                    depth += 1
                    continue

                if ch == "}" and started:
                    depth -= 1
                    if depth == 0:
                        raw_obj = "".join(decoder_buffer).strip()
                        decoder_buffer = []
                        started = False
                        if raw_obj:
                            parsed = _safe_json_loads(raw_obj, path)
                            if parsed is not None:
                                yield parsed


def _should_include_file(path: Path, exclude_contains: List[str], exclude_endswith_tokens: List[str]) -> bool:
    stem = path.stem.lower()
    if any(token.lower() in stem for token in exclude_contains):
        return False
    for token in exclude_endswith_tokens:
        tok = token.lower()
        if re.search(rf"(^|[_\-\s]){re.escape(tok)}$", stem):
            return False
    return True


def list_corpus_files(
    path: str,
    recursive: bool = False,
    exclude_contains: Optional[List[str]] = None,
    exclude_endswith_tokens: Optional[List[str]] = None,
) -> Dict[str, List[Dict[str, str]]]:
    """Returns included and skipped corpus files with skip reasons."""
    p = Path(path)
    exclude_contains = exclude_contains or []
    exclude_endswith_tokens = exclude_endswith_tokens or []

    included: List[Dict[str, str]] = []
    skipped: List[Dict[str, str]] = []

    def _append_included(fpath: Path) -> None:
        included.append({"path": str(fpath)})

    def _append_skipped(fpath: Path, reason: str) -> None:
        skipped.append({"path": str(fpath), "reason": reason})

    if p.is_file():
        if p.suffix.lower() not in {".json", ".jsonl"}:
            _append_skipped(p, "unsupported_extension")
        elif not _should_include_file(p, exclude_contains, exclude_endswith_tokens):
            _append_skipped(p, "filtered_out")
        else:
            _append_included(p)
        return {"included": included, "skipped": skipped}

    if not p.is_dir():
        return {"included": included, "skipped": skipped}

    iterator = p.rglob("*") if recursive else p.glob("*")
    for fpath in sorted(x for x in iterator if x.is_file()):
        suffix = fpath.suffix.lower()
        if suffix not in {".json", ".jsonl"}:
            continue
        if not _should_include_file(fpath, exclude_contains, exclude_endswith_tokens):
            _append_skipped(fpath, "filtered_out")
            continue
        _append_included(fpath)

    return {"included": included, "skipped": skipped}


def iter_corpus_records_with_source(
    path: str,
    recursive: bool = False,
    exclude_contains: Optional[List[str]] = None,
    exclude_endswith_tokens: Optional[List[str]] = None,
) -> Generator[Tuple[Dict, str], None, None]:
    """Yields (record, source_file_path) for each corpus record."""
    listed = list_corpus_files(
        path,
        recursive=recursive,
        exclude_contains=exclude_contains,
        exclude_endswith_tokens=exclude_endswith_tokens,
    )
    for entry in listed["included"]:
        fpath = Path(entry["path"])
        for row in _iter_json_objects(fpath):
            yield row, str(fpath)


def iter_corpus_records(
    path: str,
    recursive: bool = False,
    exclude_contains: Optional[List[str]] = None,
    exclude_endswith_tokens: Optional[List[str]] = None,
) -> Generator[Dict, None, None]:
    for row, _source in iter_corpus_records_with_source(
        path,
        recursive=recursive,
        exclude_contains=exclude_contains,
        exclude_endswith_tokens=exclude_endswith_tokens,
    ):
        yield row


def _normalize_lang(value: str) -> str:
    if not value:
        return ""
    return value.strip().lower()


def _infer_language_from_filename(path: Path) -> str:
    """Infer language from filenames containing questions_en/questions_ru/questions_kg."""
    stem = path.stem.lower()
    patterns = {
        "en": r"(?:^|[^a-z0-9])questions_en(?:[^a-z0-9]|$)",
        "ru": r"(?:^|[^a-z0-9])questions_ru(?:[^a-z0-9]|$)",
        "kg": r"(?:^|[^a-z0-9])questions_kg(?:[^a-z0-9]|$)",
    }
    for lang, regex in patterns.items():
        if re.search(regex, stem):
            return lang
    return ""


def _extract_question_text(row: Dict, lang: str) -> str:
    """Extract question text from common schema variants."""
    keys = ["question"]
    if lang:
        keys.append(f"question_{lang}")
    keys.extend(["question_en", "question_kg", "question_ru"])

    seen = set()
    for key in keys:
        if key in seen:
            continue
        seen.add(key)
        value = row.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text

    for key, value in row.items():
        if not str(key).startswith("question_"):
            continue
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text

    return ""


def discover_question_files(path: str) -> List[Dict[str, str]]:
    """
    Returns question file entries with inferred language.
    Supports either a single file path or a directory path.
    """
    p = Path(path)
    out: List[Dict[str, str]] = []

    if p.is_file():
        lang = _infer_language_from_filename(p)
        if lang:
            out.append({"path": str(p), "language": lang})
        return out

    if not p.is_dir():
        return out

    for fpath in sorted(x for x in p.rglob("*") if x.is_file()):
        lang = _infer_language_from_filename(fpath)
        if not lang:
            continue
        out.append({"path": str(fpath), "language": lang})
    return out


def load_questions(path: str, language: str) -> List[Dict]:
    lang = _normalize_lang(language)
    p = Path(path)

    if p.suffix.lower() == ".csv":
        rows = []
        with p.open("r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                q_lang = _normalize_lang(row.get("language", ""))
                if q_lang and q_lang != lang:
                    continue
                rows.append(
                    {
                        "id_question": row.get("id_question") or row.get("question_id"),
                        "id": str(row.get("id", "")).strip(),
                        "question": _extract_question_text(row, lang),
                        "language": q_lang or lang,
                    }
                )
        return rows

    # JSON / JSONL / array / multiline objects
    questions = []
    for row in _iter_json_objects(p):
        q_lang = _normalize_lang(str(row.get("language", "")))
        if q_lang and q_lang != lang:
            continue
        if not q_lang:
            fallback_lang = _normalize_lang(str(row.get("lang", "")))
            q_lang = fallback_lang or lang
            if q_lang != lang:
                continue
        questions.append(
            {
                "id_question": row.get("id_question") or row.get("question_id"),
                "id": str(row.get("id", "")).strip(),
                "question": _extract_question_text(row, lang),
                "language": q_lang,
            }
        )
    return questions


def iter_jsonl(path: Path) -> Iterable[Dict]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)
