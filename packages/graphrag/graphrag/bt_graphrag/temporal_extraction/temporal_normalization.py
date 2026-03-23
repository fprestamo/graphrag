# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Temporal expression normalization and document dating.

Scans text units for temporal expressions and normalizes them to ISO 8601
intervals. Assigns t_valid and t_tx to every document and text unit.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any

logger = logging.getLogger(__name__)

# Common date patterns for extraction
_ISO_DATE_RE = re.compile(
    r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b"
)
_MONTH_YEAR_RE = re.compile(
    r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2},?\s+)?(\d{4})\b",
    re.IGNORECASE,
)
_QUARTER_RE = re.compile(
    r"\b[Qq]([1-4])\s+(\d{4})\b"
)
_YEAR_RANGE_RE = re.compile(
    r"\b(\d{4})\s*[-–—]\s*(\d{4}|present|now|current)\b",
    re.IGNORECASE,
)
_YEAR_RE = re.compile(
    r"\b((?:19|20)\d{2})\b"
)
_RELATIVE_RE = re.compile(
    r"\b(last|past|previous|next)\s+(year|month|week|day|quarter)\b",
    re.IGNORECASE,
)
_AS_OF_RE = re.compile(
    r"\b[Aa]s\s+of\s+(.+?)(?:[,.]|\s+the\b|\s+and\b|$)"
)
_FROM_TO_RE = re.compile(
    r"\b[Ff]rom\s+(\d{4})\s+(?:to|through|until)\s+(\d{4}|present|now)\b",
    re.IGNORECASE,
)
_UNTIL_RE = re.compile(
    r"\b[Uu]ntil\s+(.+?)(?:[,.]|$)"
)

MONTH_MAP = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}


def parse_date_from_string(text: str, reference_date: datetime | None = None) -> datetime | None:
    """Best-effort date parsing from a natural language string.

    Uses regex-based extraction. Falls back to dateutil if available.
    """
    if reference_date is None:
        reference_date = datetime.now(timezone.utc)

    # Try ISO date
    m = _ISO_DATE_RE.search(text)
    if m:
        try:
            return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)), tzinfo=timezone.utc)
        except ValueError:
            pass

    # Try "Month Day, Year" or "Month Year"
    m = _MONTH_YEAR_RE.search(text)
    if m:
        month = MONTH_MAP.get(m.group(1).lower(), 1)
        day_str = m.group(2)
        day = int(day_str.strip().rstrip(",")) if day_str else 1
        year = int(m.group(3))
        try:
            return datetime(year, month, day, tzinfo=timezone.utc)
        except ValueError:
            return datetime(year, month, 1, tzinfo=timezone.utc)

    # Try quarter notation: Q3 2023
    m = _QUARTER_RE.search(text)
    if m:
        quarter = int(m.group(1))
        year = int(m.group(2))
        month = (quarter - 1) * 3 + 1
        return datetime(year, month, 1, tzinfo=timezone.utc)

    # Try plain year
    m = _YEAR_RE.search(text)
    if m:
        return datetime(int(m.group(1)), 1, 1, tzinfo=timezone.utc)

    # Try dateutil as fallback
    try:
        from dateutil import parser as dateutil_parser
        return dateutil_parser.parse(text, default=reference_date).replace(tzinfo=timezone.utc)
    except Exception:
        pass

    return None


def extract_temporal_anchors(
    text: str,
    reference_date: datetime,
) -> list[dict[str, Any]]:
    """Extract all temporal expressions from a text unit.

    Returns a list of temporal anchors, each with:
    - start: datetime (start of the interval)
    - end: datetime | None (end of the interval, None = open-ended)
    - expression: str (the original text matched)
    - type: str (absolute, relative, range, quarter)
    """
    anchors: list[dict[str, Any]] = []

    # Year ranges: "from 2010 to 2015", "2010-2015"
    for m in _YEAR_RANGE_RE.finditer(text):
        start_year = int(m.group(1))
        end_str = m.group(2).lower()
        start = datetime(start_year, 1, 1, tzinfo=timezone.utc)
        end = None if end_str in ("present", "now", "current") else datetime(int(end_str), 12, 31, tzinfo=timezone.utc)
        anchors.append({
            "start": start, "end": end,
            "expression": m.group(0), "type": "range",
        })

    for m in _FROM_TO_RE.finditer(text):
        start_year = int(m.group(1))
        end_str = m.group(2).lower()
        start = datetime(start_year, 1, 1, tzinfo=timezone.utc)
        end = None if end_str in ("present", "now") else datetime(int(end_str), 12, 31, tzinfo=timezone.utc)
        anchors.append({
            "start": start, "end": end,
            "expression": m.group(0), "type": "range",
        })

    # Quarter: Q3 2023
    for m in _QUARTER_RE.finditer(text):
        quarter = int(m.group(1))
        year = int(m.group(2))
        start_month = (quarter - 1) * 3 + 1
        end_month = start_month + 2
        start = datetime(year, start_month, 1, tzinfo=timezone.utc)
        if end_month == 12:
            end = datetime(year, 12, 31, tzinfo=timezone.utc)
        else:
            end = datetime(year, end_month + 1, 1, tzinfo=timezone.utc) - timedelta(days=1)
        anchors.append({
            "start": start, "end": end,
            "expression": m.group(0), "type": "quarter",
        })

    # Relative: "last year", "past month"
    for m in _RELATIVE_RE.finditer(text):
        direction = m.group(1).lower()
        unit = m.group(2).lower()
        if direction in ("last", "past", "previous"):
            if unit == "year":
                start = datetime(reference_date.year - 1, 1, 1, tzinfo=timezone.utc)
                end = datetime(reference_date.year - 1, 12, 31, tzinfo=timezone.utc)
            elif unit == "month":
                if reference_date.month == 1:
                    start = datetime(reference_date.year - 1, 12, 1, tzinfo=timezone.utc)
                else:
                    start = datetime(reference_date.year, reference_date.month - 1, 1, tzinfo=timezone.utc)
                end = datetime(reference_date.year, reference_date.month, 1, tzinfo=timezone.utc) - timedelta(days=1)
            elif unit == "week":
                end = reference_date - timedelta(days=reference_date.weekday() + 1)
                start = end - timedelta(days=6)
            elif unit == "quarter":
                current_q = (reference_date.month - 1) // 3
                if current_q == 0:
                    start = datetime(reference_date.year - 1, 10, 1, tzinfo=timezone.utc)
                    end = datetime(reference_date.year - 1, 12, 31, tzinfo=timezone.utc)
                else:
                    sm = (current_q - 1) * 3 + 1
                    start = datetime(reference_date.year, sm, 1, tzinfo=timezone.utc)
                    em = sm + 2
                    if em == 12:
                        end = datetime(reference_date.year, 12, 31, tzinfo=timezone.utc)
                    else:
                        end = datetime(reference_date.year, em + 1, 1, tzinfo=timezone.utc) - timedelta(days=1)
            else:
                start = reference_date - timedelta(days=1)
                end = start
            anchors.append({
                "start": start, "end": end,
                "expression": m.group(0), "type": "relative",
            })

    # ISO dates as point-in-time anchors
    for m in _ISO_DATE_RE.finditer(text):
        try:
            dt = datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)), tzinfo=timezone.utc)
            anchors.append({
                "start": dt, "end": dt,
                "expression": m.group(0), "type": "absolute",
            })
        except ValueError:
            pass

    # Month Year as point-in-time
    for m in _MONTH_YEAR_RE.finditer(text):
        month = MONTH_MAP.get(m.group(1).lower(), 1)
        year = int(m.group(3))
        start = datetime(year, month, 1, tzinfo=timezone.utc)
        if month == 12:
            end = datetime(year, 12, 31, tzinfo=timezone.utc)
        else:
            end = datetime(year, month + 1, 1, tzinfo=timezone.utc) - timedelta(days=1)
        anchors.append({
            "start": start, "end": end,
            "expression": m.group(0), "type": "absolute",
        })

    return anchors


def assign_document_timestamps(
    doc: dict[str, Any],
    valid_time_field: str = "creation_date",
) -> tuple[datetime, datetime]:
    """Assign t_valid and t_tx to a document.

    t_valid: derived from the document's date field or header date expressions.
    t_tx: current system time (when ingestion occurs).

    Returns (t_valid, t_tx).
    """
    t_tx = datetime.now(timezone.utc)

    # Try the configured valid-time field
    raw_date = doc.get(valid_time_field)
    if raw_date:
        if isinstance(raw_date, datetime):
            t_valid = raw_date.replace(tzinfo=timezone.utc) if raw_date.tzinfo is None else raw_date
        elif isinstance(raw_date, str):
            parsed = parse_date_from_string(raw_date, t_tx)
            t_valid = parsed if parsed else t_tx
        else:
            t_valid = t_tx
    else:
        # Try to extract from the document text header (first 500 chars)
        text = doc.get("text", "")[:500]
        anchors = extract_temporal_anchors(text, t_tx)
        if anchors:
            t_valid = anchors[0]["start"]
        else:
            t_valid = t_tx

    return t_valid, t_tx


def is_late_arrival(
    t_valid: datetime,
    t_tx: datetime,
    threshold_days: int = 30,
) -> bool:
    """Check if a document is a late arrival.

    A document is late if t_tx - t_valid > threshold_days.
    """
    delta = t_tx - t_valid
    return delta > timedelta(days=threshold_days)
