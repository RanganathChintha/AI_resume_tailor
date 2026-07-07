"""Text normalization and section splitting for raw resume text.

These helpers turn messy PDF-extracted text into clean, section-aware data
that the parser and renderer can work with. Pure functions, no shared state.
"""
import re
from typing import List, Tuple

from app.core.resume_filter import ResumeFilter


# Maps the many header spellings resumes use onto a small set of canonical keys.
SECTION_MAP = {
    'professional summary': 'summary',
    'summary': 'summary',
    'education': 'education',
    'skills': 'skills',
    'technical skills': 'skills',
    'experience': 'experience',
    'work experience': 'experience',
    'academic projects': 'projects',
    'projects': 'projects',
    'achievements & certifications': 'achievements',
    'certifications & achievements': 'achievements',
    'certifications': 'achievements',
    'achievements': 'achievements',
    'awards & achievements': 'achievements',
}

# Longest headers first so multi-word headers match before their prefixes.
SECTION_HEADERS = sorted(SECTION_MAP.keys(), key=len, reverse=True)

# Matches a section header optionally followed by trailing content on the line.
_HEADER_PATTERN = re.compile(
    r'^(?P<header>' + '|'.join(re.escape(h) for h in SECTION_HEADERS) + r')(?P<rest>[\s:–—-].*)?$',
    re.IGNORECASE,
)

_CONTACT_RE = re.compile(
    r'[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}|\+?\d[\d\s\-]{7,}\d|linkedin\.com|github\.com',
    re.IGNORECASE,
)


def normalize_text(text: str) -> str:
    """Clean PDF-extracted text: fix line endings, preserve headers/bullets/
    contact lines, merge run-on lines, and collapse repeated blank lines."""
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    normalized: List[str] = []
    previous_blank = True

    for line in text.split('\n'):
        stripped = line.strip()

        if not stripped:
            normalized.append('')
            previous_blank = True
            continue

        header_match = _HEADER_PATTERN.match(stripped)
        if header_match:
            header = header_match.group('header').strip()
            rest = (header_match.group('rest') or '').lstrip(' :–—-').strip()
            normalized.append(header)
            if rest:
                normalized.append(rest)
            previous_blank = False
            continue

        is_contact_line = bool(_CONTACT_RE.search(stripped))
        is_bullet = stripped.startswith(('•', '-', '*'))
        is_category = bool(re.match(r'^[A-Za-z][A-Za-z /&]+:', stripped))

        # Replace stray replacement-char bullets from PDF extraction.
        if stripped.startswith('�'):
            normalized.append('•' + stripped[1:])
            previous_blank = False
            continue

        if is_bullet or is_contact_line or is_category:
            normalized.append(stripped)
            previous_blank = False
            continue

        # Merge a wrapped continuation line into the previous content line.
        if not previous_blank:
            prev = normalized[-1]
            if prev and prev[-1].isalnum() and stripped and stripped[0].isalnum():
                normalized[-1] = prev + ' ' + stripped
            else:
                normalized[-1] = prev + stripped
        else:
            normalized.append(stripped)
        previous_blank = False

    # Collapse runs of blank lines to a single separator and tidy whitespace.
    result: List[str] = []
    blank_count = 0
    for line in normalized:
        if line == '':
            blank_count += 1
        else:
            if blank_count > 0:
                result.append('')
                blank_count = 0
            result.append(re.sub(r'\s+', ' ', line).strip())
    return '\n'.join(result).strip()


def split_sections(text: str) -> List[Tuple[str, str]]:
    """Split normalized text into (section_key, section_body) tuples."""
    sections: List[Tuple[str, str]] = []
    current_name = 'front'  # everything before the first section header
    current_lines: List[str] = []

    for line in text.split('\n'):
        normalized_line = line.strip()
        if not normalized_line:
            continue

        header_match = _HEADER_PATTERN.match(normalized_line)
        if header_match:
            if current_lines:
                sections.append((current_name, '\n'.join(current_lines).strip()))
            key = header_match.group('header').strip().lower()
            current_name = SECTION_MAP.get(key, current_name)
            rest = (header_match.group('rest') or '').lstrip(' :–—-').strip()
            current_lines = [rest] if rest else []
            continue

        current_lines.append(normalized_line)

    if current_lines:
        sections.append((current_name, '\n'.join(current_lines).strip()))

    return sections


def split_list_items(text: str) -> List[str]:
    """Split a block into individual items, stripping any bullet markers."""
    items = []
    for line in text.split('\n'):
        line = line.strip()
        if not line:
            continue
        if line.startswith(('•', '-', '*')):
            items.append(line.lstrip('•-* ').strip())
        else:
            items.append(line)
    return items


def score_text(text: str, filterer: ResumeFilter) -> int:
    """JD-relevance score for a piece of text (delegates to ResumeFilter)."""
    return filterer._score(text)
