import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

def _cmp_relation(actual: int, target: int, relation: str) -> bool:
    if relation == "exactly":
        return actual == target
    if relation == "at least":
        return actual >= target
    if relation == "at most":
        return actual <= target
    raise ValueError(f"Unknown relation: {relation!r}")


_INVISIBLE_RE = re.compile(r"[\u200b\u200c\u200d\u2060\ufeff]")


def _strip_invisible(s: str) -> str:
    return _INVISIBLE_RE.sub("", s or "")


def _clean_text(text: str) -> str:
    t = _strip_invisible(text or "")
    return t.replace("\r\n", "\n").replace("\r", "\n")


def _norm_exact(s: str) -> str:
    return _strip_invisible(s or "").strip()


_QUOTE_TRANSLATION = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201A": "'",
        "\u201B": "'",
        "\u2032": "'",
        "\u00B4": "'",
        "\u0060": "'",
        "\u201C": '"',
        "\u201D": '"',
        "\u201E": '"',
        "\u201F": '"',
        "\u00AB": '"',
        "\u00BB": '"',
        "\u2033": '"',
        "\uFF02": '"',
    }
)


def _normalize_quotes(s: str) -> str:
    return (s or "").translate(_QUOTE_TRANSLATION)


def _norm_exact_quote(s: str) -> str:
    return _normalize_quotes(_norm_exact(s))


def _explode_punct_tokens(tokens: Optional[List[str]]) -> List[str]:
    if not tokens:
        return []
    out: List[str] = []
    for tok in tokens:
        if tok is None:
            continue
        tok = str(tok).strip()
        if not tok:
            continue
        if len(tok) == 1:
            out.append(tok)
        else:
            for ch in tok:
                if ch.strip():
                    out.append(ch)

    seen = set()
    dedup: List[str] = []
    for x in out:
        if x not in seen:
            seen.add(x)
            dedup.append(x)
    return dedup


_DEFAULT_MARKER_RE = re.compile(r"^(?:\*{3,}|={3,}|-{3,}|_{3,})$")


def _is_divider_or_marker_line(line: str, cons: Optional[Dict[str, Any]] = None) -> bool:
    if line is None:
        return False
    ln = line.strip()
    if not ln:
        return False

    cons = cons or {}
    pdiv_regex = cons.get("paragraph_divider_regex", None)

    if pdiv_regex:
        try:
            r = re.compile(re.escape(str(pdiv_regex)))
            return r.fullmatch(ln) is not None
        except re.error:
            return _DEFAULT_MARKER_RE.fullmatch(ln) is not None

    return _DEFAULT_MARKER_RE.fullmatch(ln) is not None


def _first_nonempty_line(text: str) -> Optional[str]:
    t = _clean_text(text)
    for ln in t.splitlines():
        if ln.strip():
            return ln
    return None


def _last_nonempty_line(text: str) -> Optional[str]:
    t = _clean_text(text)
    for ln in reversed(t.splitlines()):
        if ln.strip():
            return ln
    return None


def _last_content_line(text: str, *, cons: Optional[Dict[str, Any]] = None) -> Optional[str]:
    t = _clean_text(text)
    for ln in reversed(t.splitlines()):
        if not ln.strip():
            continue
        if _is_divider_or_marker_line(ln, cons):
            continue
        return ln
    return None


def _lines_nonempty(text: str) -> List[str]:
    t = _clean_text(text)
    return [ln for ln in t.splitlines() if ln.strip()]


_TITLE_LINE_RE = re.compile(r"^\s*(#+\s+|title\s*[:\-]|title\s+is\s*:)", flags=re.IGNORECASE)


def _is_title_line(line: str, cons: Dict[str, Any]) -> bool:
    if line is None:
        return False
    ln = line.strip()
    ts = cons.get("title_style", None)
    if ts is not None:
        return ln.startswith(str(ts))
    return _TITLE_LINE_RE.match(line) is not None


_WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)*")


_SENT_BOUNDARY_CAND_RE = re.compile(r"[.!?]")


_ABBREV_TAIL_RE = re.compile(
    r"(?:"
    r"(?:\b[A-Z]\.){2,}"
    r"|(?:\b(?:e\.g|i\.e)\.)"
    r"|(?:\b(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|Mt|vs|etc|No)\.)"
    r")$",
    flags=re.IGNORECASE,
)


def _looks_like_sentence_start(ch: str) -> bool:
    return bool(re.match(r'[A-Z0-9"\(\[\']', ch))


def _split_sentences_heuristic(text: str) -> List[str]:
    t = _clean_text(text).strip()
    if not t:
        return []

    sentences: List[str] = []
    start = 0
    n = len(t)

    for m in _SENT_BOUNDARY_CAND_RE.finditer(t):
        end_punct = m.end()
        if end_punct >= n:
            continue

        j = end_punct
        if j < n and not t[j].isspace():
            continue

        while j < n and t[j].isspace():
            j += 1
        if j >= n:
            break

        if not _looks_like_sentence_start(t[j]):
            continue

        prefix = t[start:end_punct].rstrip()
        tail = prefix[-20:] if len(prefix) > 20 else prefix
        if _ABBREV_TAIL_RE.search(tail):
            continue

        chunk = t[start:end_punct].strip()
        if chunk:
            sentences.append(chunk)
        start = j

    tail_chunk = t[start:].strip()
    if tail_chunk:
        sentences.append(tail_chunk)

    return sentences if sentences else [t]


_EMOJI_CHAR_RE = re.compile(
    r"^["
    r"\U0001F300-\U0001F5FF"
    r"\U0001F600-\U0001F64F"
    r"\U0001F680-\U0001F6FF"
    r"\U0001F700-\U0001F77F"
    r"\U0001F780-\U0001F7FF"
    r"\U0001F800-\U0001F8FF"
    r"\U0001F900-\U0001F9FF"
    r"\U0001FA00-\U0001FAFF"
    r"\u2600-\u26FF"
    r"\u2700-\u27BF"
    r"]$",
    flags=re.UNICODE,
)


@dataclass
class TextStats:
    total_words: int
    total_chars: int
    num_sentences: int
    sentence_word_counts: List[int]
    num_paragraphs: int
    paragraphs: List[str]
    emoji_count: int


def compute_text_stats(text: str, cons: Optional[Dict[str, Any]] = None) -> TextStats:
    cons = cons or {}
    raw = _clean_text(text)
    total_chars = len(raw)

    words = _WORD_RE.findall(raw)
    total_words = len(words)

    stripped = raw.strip()
    sentences = _split_sentences_heuristic(stripped) if stripped else []
    sentence_word_counts = [len(_WORD_RE.findall(s)) for s in sentences]
    num_sentences = len(sentences)

    paragraphs: List[str] = []
    if stripped:
        pdiv_required = cons.get("paragraph_divider_required") is True
        pdiv_regex = cons.get("paragraph_divider_regex", None)

        if pdiv_required:
            div_re = re.compile(re.escape(str(pdiv_regex))) if pdiv_regex else _DEFAULT_MARKER_RE
            buf: List[str] = []
            for ln in stripped.splitlines():
                ln_s = ln.strip()
                if ln_s and div_re.fullmatch(ln_s):
                    chunk = "\n".join(buf).strip()
                    if chunk:
                        paragraphs.append(chunk)
                    buf = []
                else:
                    buf.append(ln)
            tail = "\n".join(buf).strip()
            if tail:
                paragraphs.append(tail)
        else:
            paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", stripped) if p.strip()]

    title_expected = (cons.get("title_required") is True) or (cons.get("title_style") is not None)
    if title_expected and paragraphs:
        first_para = paragraphs[0]
        first_lines = [ln for ln in first_para.splitlines() if ln.strip()]
        if len(first_lines) == 1 and _is_title_line(first_lines[0], cons):
            paragraphs = paragraphs[1:]

    num_paragraphs = len(paragraphs)

    emoji_count = sum(1 for ch in raw if _EMOJI_CHAR_RE.match(ch) is not None)

    return TextStats(
        total_words=total_words,
        total_chars=total_chars,
        num_sentences=num_sentences,
        sentence_word_counts=sentence_word_counts,
        num_paragraphs=num_paragraphs,
        paragraphs=paragraphs,
        emoji_count=emoji_count,
    )


def _word_count(text: str) -> int:
    return len(_WORD_RE.findall(_clean_text(text)))


def eval_length_constraints_strict(
    text: str,
    cons: Dict[str, Any],
    *,
    return_details: bool = True,
) -> Tuple[bool, Dict[str, Any]]:
    stats = compute_text_stats(text, cons)
    failed: List[str] = []

    num_words_target = cons.get("num_words", None)
    words_rel = cons.get("words_relation", None)
    num_words_actual: Optional[int] = None
    if num_words_target is not None and words_rel is not None:
        num_words_actual = _word_count(text)
        if not _cmp_relation(num_words_actual, int(num_words_target), str(words_rel)):
            failed.append("num_words")

    def check_num(field: str, rel_field: str, actual: int):
        target = cons.get(field, None)
        rel = cons.get(rel_field, None)
        if target is None or rel is None:
            return
        if not _cmp_relation(actual, int(target), str(rel)):
            failed.append(field)

    check_num("num_chars", "chars_relation", stats.total_chars)
    check_num("num_sentences", "sentences_relation", stats.num_sentences)
    check_num("num_paragraphs", "paragraphs_relation", stats.num_paragraphs)

    max_sw = cons.get("max_sentence_words", None)
    if max_sw is not None:
        if any(cnt > int(max_sw) for cnt in stats.sentence_word_counts):
            failed.append("max_sentence_words")

    min_sw = cons.get("min_sentence_words", None)
    if min_sw is not None:
        if any(cnt < int(min_sw) for cnt in stats.sentence_word_counts):
            failed.append("min_sentence_words")

    ok = (len(failed) == 0)
    info = {
        "ok": ok,
        "failed": failed[0] if failed else None,
        "failed_all": failed,
        "total_words": stats.total_words,
        "total_chars": stats.total_chars,
        "num_sentences": stats.num_sentences,
        "sentence_word_counts": stats.sentence_word_counts,
        "num_paragraphs": stats.num_paragraphs,
        "paragraphs_preview": stats.paragraphs[:5] if stats.paragraphs else None,
        "num_words_actual": num_words_actual,
        "num_words_target": int(num_words_target) if num_words_target is not None else None,
        "words_relation": str(words_rel) if words_rel is not None else None,
        "word_regex": _WORD_RE.pattern,
        "sent_split_method": "heuristic_abbrev_guard",
    }
    return (ok, info) if return_details else (ok, {})


def _contains_word_or_phrase(text_lower: str, w: str) -> bool:
    w = w.strip().lower()
    if not w:
        return False
    if re.search(r"\s", w):
        return w in text_lower
    if re.fullmatch(r"\w+", w):
        return re.search(rf"\b{re.escape(w)}\b", text_lower) is not None
    return w in text_lower


def eval_style_constraints_strict(
    text: str,
    cons: Dict[str, Any],
    *,
    return_details: bool = True,
) -> Tuple[bool, Dict[str, Any]]:
    norm = _clean_text(text)
    stats = compute_text_stats(norm)
    failed: List[str] = []

    text_lower = norm.lower()

    forbidden_p = _explode_punct_tokens(cons.get("forbidden_punctuations"))
    required_p = _explode_punct_tokens(cons.get("required_punctuations"))

    for p in forbidden_p:
        if p in norm:
            failed.append("forbidden_punctuations")
            break

    for p in required_p:
        if p not in norm:
            failed.append("required_punctuations")
            break

    forbidden_words = cons.get("forbidden_words") or []
    for w in forbidden_words:
        if w is None:
            continue
        if _contains_word_or_phrase(text_lower, str(w)):
            failed.append("forbidden_words")
            break

    required_keywords = cons.get("required_keywords") or []
    for w in required_keywords:
        if w is None:
            continue
        if not _contains_word_or_phrase(text_lower, str(w)):
            failed.append("required_keywords")
            break

    if cons.get("no_digits", None) is True:
        if re.search(r"[0-9]", norm):
            failed.append("no_digits")

    ec = cons.get("emoji_count", None)
    er = cons.get("emoji_relation", None)
    if ec is not None and er is not None:
        if not _cmp_relation(stats.emoji_count, int(ec), str(er)):
            failed.append("emoji_count")

    ok = (len(failed) == 0)
    info = {
        "ok": ok,
        "failed": failed[0] if failed else None,
        "failed_all": failed,
        "forbidden_punctuations": forbidden_p,
        "required_punctuations": required_p,
        "emoji_count": stats.emoji_count,
    }
    return (ok, info) if return_details else (ok, {})


def _has_markdown_table(text: str) -> bool:
    lines = _clean_text(text).splitlines()
    sep_re = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$")
    for i in range(len(lines) - 1):
        a = lines[i].strip()
        b = lines[i + 1].strip()
        if "|" in a and "-" in b and "|" in b:
            if sep_re.match(b):
                return True
    return False


def _has_paragraph_divider(text: str, cons: Dict[str, Any]) -> bool:
    lines = _clean_text(text).splitlines()
    pdiv_regex = cons.get("paragraph_divider_regex", None)

    if pdiv_regex:
        try:
            r = re.compile(re.escape(str(pdiv_regex)))
            return any(r.fullmatch(ln.strip()) for ln in lines if ln.strip())
        except re.error:
            return any(_DEFAULT_MARKER_RE.fullmatch(ln.strip()) for ln in lines if ln.strip())

    return any(_DEFAULT_MARKER_RE.fullmatch(ln.strip()) for ln in lines if ln.strip())


def _iter_separator_lines_between_content(text: str, cons: Dict[str, Any]) -> List[str]:
    lines = _clean_text(text).splitlines()
    pdiv_regex = cons.get("paragraph_divider_regex", None)
    r: Optional[re.Pattern] = None
    if pdiv_regex:
        try:
            r = re.compile(re.escape(str(pdiv_regex)))
        except re.error:
            r = None

    def is_candidate(s: str) -> bool:
        if not s:
            return False
        if r is not None and r.fullmatch(s) is not None:
            return True
        return _DEFAULT_MARKER_RE.fullmatch(s) is not None

    seps: List[str] = []
    for i, ln in enumerate(lines):
        s = ln.strip()
        if not s or not is_candidate(s):
            continue

        j = i - 1
        while j >= 0 and not lines[j].strip():
            j -= 1
        k = i + 1
        while k < len(lines) and not lines[k].strip():
            k += 1

        if j < 0 or k >= len(lines):
            continue

        prev_s = lines[j].strip()
        next_s = lines[k].strip()

        if not prev_s or not next_s:
            continue
        if is_candidate(prev_s) or is_candidate(next_s):
            continue

        seps.append(s)

    return seps


def _paragraph_divider_regex_ok(text: str, cons: Dict[str, Any]) -> Tuple[bool, List[str]]:
    pdiv_regex = cons.get("paragraph_divider_regex", None)
    if not pdiv_regex:
        return True, []

    if cons.get("paragraph_divider_required") is not True:
        return True, []

    try:
        r = re.compile(re.escape(str(pdiv_regex)))
    except re.error:
        return False, ["<invalid paragraph_divider_regex>"]

    bad: List[str] = []
    seps = _iter_separator_lines_between_content(text, cons)
    for s in seps:
        if r.fullmatch(s) is None:
            bad.append(s)

    return (len(bad) == 0), bad


def eval_structure_constraints_strict(
    text: str,
    cons: Dict[str, Any],
    *,
    return_details: bool = True,
) -> Tuple[bool, Dict[str, Any]]:
    failed: List[str] = []
    norm = _clean_text(text)

    title_required = cons.get("title_required", None)
    no_title = cons.get("no_title", None)
    title_style = cons.get("title_style", None)
    title_position = cons.get("title_position", None)

    first_ln = _first_nonempty_line(norm)
    first_ln_stripped = first_ln.strip() if first_ln else ""

    if title_required is True:
        if first_ln is None:
            failed.append("title_required")
        else:
            if title_style is not None and not first_ln_stripped.startswith(str(title_style)):
                failed.append("title_style")

    if no_title is True:
        for ln in _lines_nonempty(norm):
            if _TITLE_LINE_RE.match(ln):
                failed.append("no_title")
                break

    if title_position is not None and title_required is True:
        pos = str(title_position).strip().lower()
        if pos in {"top", "start", "beginning", "first"}:
            pass
        elif pos in {"bottom", "end", "last"}:
            last_ln = _last_nonempty_line(norm)
            if last_ln is None or not _is_title_line(last_ln, cons):
                failed.append("title_position")

    table_required = cons.get("table_required", None)
    no_table = cons.get("no_table", None)

    has_table = _has_markdown_table(norm)
    if table_required is True and not has_table:
        failed.append("table_required")
    if no_table is True and has_table:
        failed.append("no_table")

    pdiv_required = cons.get("paragraph_divider_required", None)
    has_div = _has_paragraph_divider(norm, cons)

    if pdiv_required is True and not has_div:
        failed.append("paragraph_divider_required")
    if pdiv_required is False and has_div:
        failed.append("paragraph_divider_required")

    pdiv_regex_ok, pdiv_bad_lines = _paragraph_divider_regex_ok(norm, cons)
    if cons.get("paragraph_divider_regex", None) is not None and not pdiv_regex_ok:
        failed.append("paragraph_divider_regex")

    ok = (len(failed) == 0)
    info = {
        "ok": ok,
        "failed": failed[0] if failed else None,
        "failed_all": failed,
        "first_nonempty_line": first_ln_stripped,
        "has_table": has_table,
        "has_paragraph_divider": has_div,
        "paragraph_divider_regex_ok": pdiv_regex_ok,
        "paragraph_divider_regex_bad_lines": pdiv_bad_lines[:10] if pdiv_bad_lines else None,
        "paragraph_divider_separators_detected": _iter_separator_lines_between_content(norm, cons)[:10],
    }
    return (ok, info) if return_details else (ok, {})


def eval_format_constraints_strict(
    text: str,
    cons: Dict[str, Any],
    *,
    return_details: bool = True,
) -> Tuple[bool, Dict[str, Any]]:
    failed: List[str] = []
    out = _clean_text(text)

    exp_first = cons.get("first_line", None)
    found_first = _first_nonempty_line(out)
    if exp_first is not None:
        if found_first is None or _norm_exact_quote(found_first) != _norm_exact_quote(str(exp_first)):
            failed.append("first_line")

    exp_last = cons.get("last_line", None)
    found_last = _last_nonempty_line(out)
    if exp_last is not None:
        if found_last is None:
            failed.append("last_line")
        else:
            exp_last_n = _norm_exact_quote(str(exp_last))
            found_last_n = _norm_exact_quote(found_last)
            if not found_last_n.endswith(exp_last_n):
                failed.append("last_line")

    sw = cons.get("starts_with", None)
    if sw is not None:
        if not out.lstrip().startswith(str(sw)):
            failed.append("starts_with")

    ew = cons.get("ends_with", None)
    if ew is not None:
        if not out.rstrip().endswith(str(ew)):
            failed.append("ends_with")

    ok = (len(failed) == 0)
    info = {
        "ok": ok,
        "failed": failed[0] if failed else None,
        "failed_all": failed,
        "found_first_line": _norm_exact(found_first) if found_first is not None else None,
        "found_last_line": _norm_exact(found_last) if found_last is not None else None,
        "expected_first_line": _norm_exact(str(exp_first)) if exp_first is not None else None,
        "expected_last_line": _norm_exact(str(exp_last)) if exp_last is not None else None,
        "found_first_line_qnorm": _norm_exact_quote(found_first) if found_first is not None else None,
        "found_last_line_qnorm": _norm_exact_quote(found_last) if found_last is not None else None,
        "expected_first_line_qnorm": _norm_exact_quote(str(exp_first)) if exp_first is not None else None,
        "expected_last_line_qnorm": _norm_exact_quote(str(exp_last)) if exp_last is not None else None,
        "last_content_line": _norm_exact(_last_content_line(out, cons=cons) or "") or None,
    }
    return (ok, info) if return_details else (ok, {})


def evaluate_all_constraints_strict(
    text: str,
    cons: Dict[str, Any],
    *,
    return_details: bool = True,
) -> Tuple[bool, Dict[str, Any]]:
    ok_len, info_len = eval_length_constraints_strict(text, cons, return_details=True)
    ok_style, info_style = eval_style_constraints_strict(text, cons, return_details=True)
    ok_struct, info_struct = eval_structure_constraints_strict(text, cons, return_details=True)
    ok_fmt, info_fmt = eval_format_constraints_strict(text, cons, return_details=True)

    ok = ok_len and ok_style and ok_struct and ok_fmt
    details = {
        "ok": ok,
        "length": info_len,
        "style": info_style,
        "structure": info_struct,
        "format": info_fmt,
    }
    return (ok, details) if return_details else (ok, {})


from dataclasses import dataclass
from collections import defaultdict, Counter
from typing import Iterable

def _active_fields_length(cons: Dict[str, Any]) -> List[str]:
    active: List[str] = []
    if cons.get("num_words") is not None and cons.get("words_relation") is not None:
        active.append("num_words")
    if cons.get("num_chars") is not None and cons.get("chars_relation") is not None:
        active.append("num_chars")
    if cons.get("num_sentences") is not None and cons.get("sentences_relation") is not None:
        active.append("num_sentences")
    if cons.get("num_paragraphs") is not None and cons.get("paragraphs_relation") is not None:
        active.append("num_paragraphs")
    if cons.get("max_sentence_words") is not None:
        active.append("max_sentence_words")
    if cons.get("min_sentence_words") is not None:
        active.append("min_sentence_words")
    return active


def _active_fields_style(cons: Dict[str, Any]) -> List[str]:
    active: List[str] = []
    if _explode_punct_tokens(cons.get("forbidden_punctuations")):
        active.append("forbidden_punctuations")
    if _explode_punct_tokens(cons.get("required_punctuations")):
        active.append("required_punctuations")
    if cons.get("forbidden_words"):
        active.append("forbidden_words")
    if cons.get("required_keywords"):
        active.append("required_keywords")
    if cons.get("no_digits", None) is True:
        active.append("no_digits")
    if cons.get("emoji_count") is not None and cons.get("emoji_relation") is not None:
        active.append("emoji_count")
    return active


def _active_fields_structure(cons: Dict[str, Any]) -> List[str]:
    active: List[str] = []

    title_required = cons.get("title_required", None)
    no_title = cons.get("no_title", None)
    title_style = cons.get("title_style", None)
    title_position = cons.get("title_position", None)

    if title_required is True:
        active.append("title_required")
        if title_style is not None:
            active.append("title_style")

    if no_title is True:
        active.append("no_title")

    if title_position is not None and title_required is True:
        active.append("title_position")

    if cons.get("table_required", None) is True:
        active.append("table_required")
    if cons.get("no_table", None) is True:
        active.append("no_table")

    if cons.get("paragraph_divider_required", None) is True or cons.get("paragraph_divider_required", None) is False:
        active.append("paragraph_divider_required")

    if cons.get("paragraph_divider_regex", None) is not None and cons.get("paragraph_divider_required") is True:
        active.append("paragraph_divider_regex")

    return active


def _active_fields_format(cons: Dict[str, Any]) -> List[str]:
    active: List[str] = []
    if cons.get("first_line", None) is not None:
        active.append("first_line")
    if cons.get("last_line", None) is not None:
        active.append("last_line")
    if cons.get("starts_with", None) is not None:
        active.append("starts_with")
    if cons.get("ends_with", None) is not None:
        active.append("ends_with")
    return active


def _active_fields_all(cons: Dict[str, Any]) -> Dict[str, List[str]]:
    return {
        "length": _active_fields_length(cons),
        "style": _active_fields_style(cons),
        "structure": _active_fields_structure(cons),
        "format": _active_fields_format(cons),
    }


def _failed_set(details: Dict[str, Any], section: str) -> set:
    sec = details.get(section, {}) or {}
    fa = sec.get("failed_all") or []
    return set(fa)


@dataclass
class BenchReport:
    prompt_total: int
    prompt_correct: int
    prompt_accuracy: float

    constraint_total: int
    constraint_correct: int
    constraint_accuracy: float

    section_prompt_total: Dict[str, int]
    section_prompt_correct: Dict[str, int]
    section_prompt_accuracy: Dict[str, float]

    field_total: Dict[str, int]
    field_correct: Dict[str, int]
    field_accuracy: Dict[str, float]

    failure_counter: Dict[str, int]
    failure_counter_by_section: Dict[str, Dict[str, int]]


def evaluate_benchmark_report(
    records: Iterable[Dict[str, Any]],
    *,
    text_key: str = "text",
    cons_key: str = "cons",
    id_key: str = "id",
    return_per_record: bool = False,
) -> Tuple[Dict[str, Any], Optional[List[Dict[str, Any]]]]:
    prompt_total = 0
    prompt_correct = 0

    constraint_total = 0
    constraint_correct = 0

    section_prompt_total = defaultdict(int)
    section_prompt_correct = defaultdict(int)

    field_total = defaultdict(int)
    field_correct = defaultdict(int)

    failure_counter = Counter()
    failure_counter_by_section = defaultdict(Counter)

    per_record_out: List[Dict[str, Any]] = []

    for rec in records:
        if not isinstance(rec, dict):
            continue
        if text_key not in rec or cons_key not in rec:
            continue

        sid = rec.get(id_key, None)
        text = rec[text_key]
        cons = rec[cons_key]

        ok, details = evaluate_all_constraints_strict(text, cons, return_details=True)

        prompt_total += 1
        if ok:
            prompt_correct += 1

        active = _active_fields_all(cons)

        for sec in ["length", "style", "structure", "format"]:
            if not active[sec]:
                continue
            section_prompt_total[sec] += 1
            if details.get(sec, {}).get("ok") is True:
                section_prompt_correct[sec] += 1

        failed_len = _failed_set(details, "length")
        failed_style = _failed_set(details, "style")
        failed_struct = _failed_set(details, "structure")
        failed_fmt = _failed_set(details, "format")

        sec2failed = {
            "length": failed_len,
            "style": failed_style,
            "structure": failed_struct,
            "format": failed_fmt,
        }

        for sec in ["length", "style", "structure", "format"]:
            for f in sec2failed[sec]:
                failure_counter[f] += 1
                failure_counter_by_section[sec][f] += 1

        for sec in ["length", "style", "structure", "format"]:
            for f in active[sec]:
                constraint_total += 1
                field_total[f] += 1
                if f not in sec2failed[sec]:
                    constraint_correct += 1
                    field_correct[f] += 1

        if return_per_record:
            per_record_out.append(
                {
                    "id": sid,
                    "ok": ok,
                    "failed_all": {
                        "length": sorted(list(failed_len)),
                        "style": sorted(list(failed_style)),
                        "structure": sorted(list(failed_struct)),
                        "format": sorted(list(failed_fmt)),
                    },
                    "active_fields": active,
                }
            )

    prompt_accuracy = (prompt_correct / prompt_total) if prompt_total else 0.0
    constraint_accuracy = (constraint_correct / constraint_total) if constraint_total else 0.0

    section_prompt_accuracy = {}
    for sec in ["length", "style", "structure", "format"]:
        t = section_prompt_total.get(sec, 0)
        c = section_prompt_correct.get(sec, 0)
        section_prompt_accuracy[sec] = (c / t) if t else 0.0

    field_accuracy = {}
    for f in sorted(field_total.keys()):
        t = field_total[f]
        c = field_correct.get(f, 0)
        field_accuracy[f] = (c / t) if t else 0.0

    report = BenchReport(
        prompt_total=prompt_total,
        prompt_correct=prompt_correct,
        prompt_accuracy=prompt_accuracy,
        constraint_total=constraint_total,
        constraint_correct=constraint_correct,
        constraint_accuracy=constraint_accuracy,
        section_prompt_total=dict(section_prompt_total),
        section_prompt_correct=dict(section_prompt_correct),
        section_prompt_accuracy=section_prompt_accuracy,
        field_total=dict(field_total),
        field_correct=dict(field_correct),
        field_accuracy=field_accuracy,
        failure_counter=dict(failure_counter),
        failure_counter_by_section={k: dict(v) for k, v in failure_counter_by_section.items()},
    )

    report_dict = dataclass_to_report_dict(report)
    return report_dict, (per_record_out if return_per_record else None)


def dataclass_to_report_dict(r: BenchReport) -> Dict[str, Any]:
    return {
        "prompt_accuracy": r.prompt_accuracy,
        "prompt_total": r.prompt_total,
        "prompt_correct": r.prompt_correct,

        "constraint_accuracy": r.constraint_accuracy,
        "constraint_total": r.constraint_total,
        "constraint_correct": r.constraint_correct,

        "section_prompt_accuracy": r.section_prompt_accuracy,
        "section_prompt_total": r.section_prompt_total,
        "section_prompt_correct": r.section_prompt_correct,

        "constraint_accuracy_by_field": r.field_accuracy,
        "constraint_total_by_field": r.field_total,
        "constraint_correct_by_field": r.field_correct,

        "failures_top": dict(Counter(r.failure_counter).most_common(30)),
        "failures_by_section_top": {
            sec: dict(Counter(cnts).most_common(30))
            for sec, cnts in r.failure_counter_by_section.items()
        },
    }


def print_benchmark_report(report: Dict[str, Any]) -> None:
    print("=" * 80)
    print("Benchmark Report")
    print("- prompt_accuracy      :", f"{report.get('prompt_accuracy', 0.0):.4f}",
          f"({report.get('prompt_correct', 0)}/{report.get('prompt_total', 0)})")
    print("- constraint_accuracy  :", f"{report.get('constraint_accuracy', 0.0):.4f}",
          f"({report.get('constraint_correct', 0)}/{report.get('constraint_total', 0)})")

    print("\n[Section prompt accuracies] (only when section has active constraints)")
    for sec in ["length", "style", "structure", "format"]:
        acc = report.get("section_prompt_accuracy", {}).get(sec, 0.0)
        c = report.get("section_prompt_correct", {}).get(sec, 0)
        t = report.get("section_prompt_total", {}).get(sec, 0)
        print(f"- {sec:9s}: {acc:.4f} ({c}/{t})")

    print("\n[Constraint accuracies by field] (active-only)")
    fa = report.get("constraint_accuracy_by_field", {}) or {}
    ft = report.get("constraint_total_by_field", {}) or {}
    fc = report.get("constraint_correct_by_field", {}) or {}
    for f in sorted(fa.keys()):
        print(f"- {f:24s}: {fa[f]:.4f} ({fc.get(f,0)}/{ft.get(f,0)})")

    print("\n[Top failures overall]")
    for k, v in (report.get('failures_top', {}) or {}).items():
        print(f"- {k:24s}: {v}")

    print("\n[Top failures by section]")
    fbs = report.get("failures_by_section_top", {}) or {}
    for sec in ["length", "style", "structure", "format"]:
        print(f"\n  * {sec}")
        for k, v in (fbs.get(sec, {}) or {}).items():
            print(f"    - {k:22s}: {v}")
    print("=" * 80)
