"""Deterministic extraction of resume sections from normalized text.

Each function pulls one kind of structured data (contact, summary, education,
skills, experience, projects, achievements) out of the section tuples produced
by :mod:`app.generator.text_utils`. Selection is JD-aware via ``ResumeFilter``.
Pure functions, no shared state.
"""
import re
from typing import Dict, List, Optional, Tuple

from app.core.resume_filter import ResumeFilter
from app.generator.text_utils import split_list_items, score_text


_CONTACT_RE = re.compile(
    r'[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}|\+?\d[\d\s\-]{7,}\d|linkedin\.com|github\.com',
    re.IGNORECASE,
)

# Shared vocabulary of technologies used when detecting project tech stacks.
_TECH_KEYWORDS = (
    r'(?:Flask|Python|Django|MongoDB|Express(?:\.js)?|React(?:\.js)?|Node(?:\.js)?|'
    r'PostgreSQL|FAISS|LangChain|RAG|SQL|REST APIs|REST API|REST|Docker|AWS|Azure|'
    r'GCP|PyTorch|TensorFlow|HTML|CSS|JWT)'
)


def extract_contact(text: str) -> Dict[str, str]:
    contact: Dict[str, str] = {'name': '', 'email': '', 'phone': '', 'linkedin': '', 'github': ''}
    lines = [line.strip() for line in text.split('\n') if line.strip()]

    email_match = re.search(r'[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}', text)
    if email_match:
        contact['email'] = email_match.group(0)

    phone_match = re.search(r'\+?\d[\d\s\-]{7,}\d', text)
    if phone_match:
        contact['phone'] = phone_match.group(0).strip()

    linkedin_match = re.search(r'(https?://)?(www\.)?linkedin\.com/[A-Za-z0-9_\-./]+', text, re.IGNORECASE)
    if linkedin_match:
        contact['linkedin'] = linkedin_match.group(0).strip()

    github_match = re.search(r'(https?://)?(www\.)?github\.com/[A-Za-z0-9_\-./]+', text, re.IGNORECASE)
    if github_match:
        contact['github'] = github_match.group(0).strip()

    if lines:
        name_candidates = [line for line in lines[:2] if not _CONTACT_RE.search(line)]
        contact['name'] = name_candidates[0] if name_candidates else lines[0]

    return contact


def extract_best_summary(sections: List[Tuple[str, str]], filterer: ResumeFilter) -> str:
    candidates = [text for name, text in sections if name == 'summary' and text]
    if not candidates:
        return ''

    best = max(candidates, key=lambda t: score_text(t, filterer))
    summary = re.sub(r'\s+', ' ', best.replace('\n', ' ').strip())
    words = summary.split()
    if len(words) > 70:
        summary = ' '.join(words[:70])
        last_period = summary.rfind('.')
        if last_period > 40:
            summary = summary[:last_period + 1]
    return summary


def extract_education(sections: List[Tuple[str, str]]) -> Dict[str, str]:
    education_blocks = [text for name, text in sections if name == 'education' and text]
    education = {'degree': '', 'institution': '', 'dates': '', 'location': ''}
    if not education_blocks:
        return education

    raw = re.sub(r'\s+', ' ', education_blocks[0].replace('\n', ' ').strip())
    raw = re.sub(r'Pre-University Course \(PUC\).*', '', raw, flags=re.IGNORECASE).strip()

    degree_match = re.search(
        r'((?:Bachelor|Master|B\.?Tech|M\.?Tech|B\.?E\.?|M\.?S\.?|B\.?Sc|M\.?Sc|MBA|Ph\.?D)[^,\n]*)',
        raw, re.IGNORECASE,
    )
    if degree_match:
        education['degree'] = degree_match.group(0).strip()

    institution_match = re.search(
        r'([A-Z][A-Za-z.&]*(?:\s+[A-Z][A-Za-z.&]*)*\s+(?:University|Institute|College|School|Academy)(?:\s+of\s+[A-Za-z ]+)?)',
        raw,
    )
    if institution_match:
        education['institution'] = institution_match.group(0).strip().rstrip(',')

    date_match = re.search(
        r'((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec) ?\d{4}) ?[–-] ?(?:Present|present|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec) ?\d{4})',
        raw,
    )
    if date_match:
        education['dates'] = date_match.group(0).replace('–', '--').replace('—', '--')

    location_match = re.search(
        r'([A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)*,\s*[A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)*)\s*$', raw)
    if location_match:
        education['location'] = location_match.group(1).strip()

    return education


# Canonical names for merging synonymous skill categories.
_SKILL_CANONICAL_MAP = {
    'languages': 'Languages',
    'programming languages': 'Languages',
    'programming': 'Languages',
    'web/backend': 'Web & Backend',
    'web development': 'Web & Backend',
    'web & backend': 'Web & Backend',
    'backend & apis': 'Backend & APIs',
    'backend and apis': 'Backend & APIs',
    'ai/ml': 'AI/ML',
    'ai / ml': 'AI/ML',
    'machine learning': 'AI/ML',
    'databases': 'Databases',
    'database': 'Databases',
    'tools': 'Tools & Platforms',
    'tools & platforms': 'Tools & Platforms',
    'tools & version control': 'Tools & Platforms',
    'cybersecurity concepts': 'Security',
    'security': 'Security',
}
_SKILL_PRIORITY = ['Languages', 'AI/ML', 'Web & Backend', 'Backend & APIs',
                   'Databases', 'Tools & Platforms', 'Security', 'Core Competencies']


def extract_skills(sections: List[Tuple[str, str]], filterer: ResumeFilter) -> List[str]:
    skills_blocks = [text for name, text in sections if name == 'skills' and text]
    lines: List[str] = []
    for block in skills_blocks:
        lines.extend(split_list_items(block))

    categories: Dict[str, str] = {}
    for line in lines:
        if ':' in line:
            key, value = line.split(':', 1)
            categories[key.strip()] = value.strip()
        elif ',' in line:
            categories[line] = line

    if not categories:
        return []

    # Merge duplicate/synonymous categories so skills are not listed twice.
    merged: Dict[str, List[str]] = {}
    for key, value in categories.items():
        canonical = _SKILL_CANONICAL_MAP.get(key.strip().lower(), key.strip())
        items = [item.strip() for item in re.split(r',\s*', value) if item.strip()]
        bucket = merged.setdefault(canonical, [])
        for item in items:
            if item.lower() not in {existing.lower() for existing in bucket}:
                bucket.append(item)
    categories = {name: ', '.join(items) for name, items in merged.items()}

    jd_keywords = filterer._get_jd_keywords()
    relevant = []
    for key, value in categories.items():
        combined_tokens = set(re.findall(r'[a-z0-9+#.]+', f'{key}: {value}'.lower()))
        if combined_tokens & jd_keywords:
            relevant.append(f'{key}: {value}')

    if not relevant:
        for name in _SKILL_PRIORITY:
            if name in categories:
                relevant.append(f'{name}: {categories[name]}')
        if not relevant:
            relevant = [f'{k}: {v}' for k, v in list(categories.items())[:5]]

    return relevant[:6]


def parse_experience_block(block: str, filterer: ResumeFilter) -> Optional[Dict[str, object]]:
    lines = [line.strip() for line in block.split('\n') if line.strip()]
    if not lines:
        return None

    bullets = []
    header_lines = []
    for line in lines:
        if line.startswith(('•', '-', '*')):
            bullets.append(line.lstrip('•-* ').strip())
        elif re.match(r'^[–—-]\s*', line):
            bullets.extend([item.strip() for item in re.split(r'[–—-]\s*', line) if item.strip()])
        else:
            header_lines.append(line)

    # Keep the raw header (original spacing) so 2+ space column separators can
    # split title / company / location / dates without hardcoding any values.
    raw_header = '  '.join(header_lines).strip() if header_lines else ''
    header = re.sub(r'\s+', ' ', raw_header)

    company = ''
    title = ''
    dates = ''
    location = ''

    date_re = re.compile(
        r'((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\s*\d{4})\s*[–—-]\s*(Present|present|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\s*\d{4})',
    )
    location_re = re.compile(
        r'^[A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+){0,2},\s*[A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+){0,2}$'
    )
    role_re = re.compile(
        r'Intern|Engineer|Developer|Scientist|Manager|Consultant|Analyst|Architect|Researcher|Trainee|Lead|Designer|Administrator|Specialist',
        re.IGNORECASE,
    )

    # Split the header into columns on runs of 2+ spaces.
    columns = [c.strip() for c in re.split(r'\s{2,}', raw_header) if c.strip()]
    text_cols: List[str] = []
    for col in columns:
        dm = date_re.search(col)
        if dm and not dates:
            dates = dm.group(0).replace('–', '--').replace('—', '--')
            remainder = col.replace(dm.group(0), '').strip(' ,|-')
            if remainder:
                text_cols.append(remainder)
            continue
        if not location and location_re.match(col):
            location = col
            continue
        if re.fullmatch(r'Remote', col, re.IGNORECASE) and not location:
            location = 'Remote'
            continue
        text_cols.append(col)

    def _split_trailing_location(value: str) -> str:
        nonlocal location
        if location:
            return value
        m = re.search(
            r'\s([A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)?,\s*[A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)?)\s*$',
            value,
        )
        if m:
            location = m.group(1).strip()
            return value[:m.start()].strip(' ,|-')
        return value

    for col in text_cols:
        if ',' in col:
            first, rest = col.split(',', 1)
            first, rest = first.strip(), rest.strip()
            if not title and role_re.search(first):
                title = first
                rest = _split_trailing_location(rest)
                if not company and rest:
                    company = rest
                continue
            if not company and role_re.search(rest):
                company = _split_trailing_location(first)
                if not title:
                    title = rest
                continue
        col = _split_trailing_location(col)
        if not title and role_re.search(col):
            title = col
        elif not company:
            company = col

    if not dates:
        date_match = date_re.search(header)
        if date_match:
            dates = date_match.group(0).replace('–', '--').replace('—', '--')

    scored_bullets = sorted(bullets, key=lambda b: score_text(b, filterer), reverse=True)
    chosen = [bullet for bullet in scored_bullets if bullet][:3]
    if not chosen and bullets:
        chosen = bullets[:3]

    return {
        'company': company,
        'role': title,
        'dates': dates,
        'location': location,
        'bullets': chosen,
    }


def extract_experience(sections: List[Tuple[str, str]], filterer: ResumeFilter) -> List[Dict[str, object]]:
    experience_blocks = [text for name, text in sections if name == 'experience' and text]
    entries: List[Tuple[int, int, Dict[str, object]]] = []
    for block in experience_blocks:
        entry = parse_experience_block(block, filterer)
        if not entry or not entry['company']:
            continue
        score = score_text(block, filterer)
        bullet_count = len(entry.get('bullets', []))
        entries.append((bullet_count, score, entry))

    entries.sort(key=lambda pair: (pair[0], pair[1]), reverse=True)
    return [entry for _, _, entry in entries][:1]


def parse_project_header(header: str) -> Tuple[str, str, str]:
    year = ''
    tech = ''
    title = header

    year_match = re.search(r'\b(20\d{2})\b', header)
    if year_match:
        year = year_match.group(1)
        title = header[:year_match.start()].strip()

    if '|' in title:
        title, tech = [part.strip() for part in title.split('|', 1)]
    elif ')' in title and ',' in title:
        header_parts = title.split(')')
        if len(header_parts) > 1:
            title = header_parts[0].strip() + ')'
            tech = header_parts[1].strip()
    elif '—' in title or '–' in title or '-' in title:
        parts = re.split(r'\s*[—–-]\s*', title)
        if len(parts) > 1:
            title = parts[0].strip()
            tech = parts[1].strip()

    if not tech:
        tech_match = re.search(
            r'(?P<tech>(?:Python|Flask|Django|MongoDB|Express(?:\.js)?|React(?:\.js)?|Node(?:\.js)?|PostgreSQL|FAISS|LangChain|RAG|SQL|REST APIs|REST API|REST|Docker|AWS|Azure|GCP|PyTorch|TensorFlow).*)$',
            title, re.IGNORECASE,
        )
        if tech_match:
            tech = tech_match.group('tech').strip()
            title = title[:tech_match.start()].strip()

    if tech and year and title.endswith(year):
        title = title[:title.rfind(year)].strip()

    return title, tech, year


def parse_all_project_blocks(sections: List[Tuple[str, str]]) -> List[Dict[str, object]]:
    """Parse the raw project section(s) into a deduped list of {header, bullets}."""
    project_blocks = [text for name, text in sections if name == 'projects' and text]
    projects: List[Dict[str, object]] = []

    header_boundary = re.compile(
        rf'(?<=[\.\n])\s*(?=[A-Z][A-Za-z0-9 &()\-]+.*?(?:\||{_TECH_KEYWORDS}))',
        re.IGNORECASE,
    )
    # A new project header can get merged into the tail of the previous bullet
    # during normalization (e.g. "...uptime NetScanner | Python | 2020"). Detect
    # a capitalized title followed by " | tech"/" | year" and split before it.
    new_project_header = re.compile(
        r'(?<=[a-z0-9%)])\s*'
        r'(?=[A-Z][A-Za-z0-9]+(?:\s+[A-Z][A-Za-z0-9]+){0,3}\s*\|\s*'
        rf'(?:{_TECH_KEYWORDS}|\d{{4}}))'
    )
    tech_separator = re.compile(rf'(?<=[^\s\|\-–—])(?={_TECH_KEYWORDS})', re.IGNORECASE)
    inline_project_bullet = re.compile(r'\s*[–—-]\s*(?=[A-Z][a-z]|•|\*)')

    for block in project_blocks:
        block = new_project_header.sub('\n', block)
        block = tech_separator.sub(' | ', block)
        block = header_boundary.sub('\n', block)
        block = inline_project_bullet.sub('\n• ', block)
        block = re.sub(r'(?<!\n)\s*(?=•)', '\n', block)
        lines = [line.strip() for line in block.split('\n') if line.strip()]
        current = None
        for line in lines:
            if line.startswith(('•', '-', '*')):
                if current is not None:
                    current['bullets'].append(line.lstrip('•-* ').strip())
                continue

            if current is None or current['bullets']:
                current = {'header': line, 'bullets': []}
                projects.append(current)
            else:
                current['header'] += ' ' + line

    # Remove duplicates by normalised title (ignore tech stack differences).
    seen_titles = set()
    deduped = []
    for p in projects:
        title_part = p['header'].split('|')[0].split('—')[0].split('–')[0].strip()
        norm_title = re.sub(r'[^a-z0-9]', '', title_part.lower())
        if norm_title not in seen_titles:
            seen_titles.add(norm_title)
            deduped.append(p)
    return deduped


def looks_like_tech_list(text: str) -> bool:
    """Heuristic: does this short line look like a tech stack, not a sentence?"""
    candidate = text.strip().rstrip('.')
    words = [w for w in re.split(r'[\s,]+', candidate) if w]
    if not words or len(words) > 6:
        return False
    tech_vocab = {
        'python', 'flask', 'django', 'mongodb', 'express', 'react', 'node',
        'nodejs', 'postgresql', 'mysql', 'sql', 'faiss', 'langchain', 'rag',
        'docker', 'aws', 'azure', 'gcp', 'pytorch', 'tensorflow', 'html',
        'css', 'jwt', 'javascript', 'typescript', 'fastapi', 'numpy',
        'pandas', 'rest', 'apis', 'api', 'mern', 'bm25', 'redis', 'kafka',
    }
    hits = sum(1 for w in words if re.sub(r'\.js$', '', w.lower()) in tech_vocab)
    return hits >= max(1, len(words) // 2)


def finalise_project(project: Dict[str, object]) -> Dict[str, object]:
    """Turn a raw {header, bullets} project into {title, tech, year, bullets}."""
    title, tech, year = parse_project_header(project['header'])
    bullets = list(project['bullets'])
    # If the tech stack leaked in as the first bullet, promote it to the tech field.
    if not tech and bullets and looks_like_tech_list(bullets[0]):
        tech = bullets.pop(0).strip().rstrip('.')
    return {
        'title': title or project['header'],
        'tech': tech,
        'year': year,
        'bullets': bullets[:3],
    }


def extract_projects(sections: List[Tuple[str, str]], filterer: ResumeFilter) -> List[Dict[str, object]]:
    projects = parse_all_project_blocks(sections)

    scored: List[Tuple[int, Dict[str, object]]] = []
    for project in projects:
        if not project['bullets']:
            continue
        score = score_text(project['header'] + ' ' + ' '.join(project['bullets']), filterer)
        scored.append((score, project))

    scored.sort(key=lambda pair: pair[0], reverse=True)

    # Keep only JD-relevant projects (positive score); never pad with an
    # unrelated project. Fall back to the single top project if nothing scores.
    relevant = [(score, project) for score, project in scored if score > 0]
    if relevant:
        selection = relevant[:2]
    elif scored:
        selection = scored[:1]
    else:
        selection = []

    return [finalise_project(project) for _, project in selection][:2]


# Category headers and noise to skip when scanning for real achievements.
_ACHIEVEMENT_CATEGORY_LABELS = {
    'problem solving', 'certifications', 'awards', 'achievements',
    'soft skills', 'soft skill', 'technical skills', 'technical', 'skills',
    'contact', 'profile', 'summary', 'experience', 'projects', 'education',
}
_ACHIEVEMENT_GENERIC_SKIP = {
    'technical research', 'team collaboration', 'problem solving',
    'communication skills', 'interpersonal skills', 'adaptability',
    'leadership', 'teamwork', 'time management', 'critical thinking',
}
_TRAILING_LABEL_RE = re.compile(
    r'\s*\.?\s*(soft skills?|technical skills?|core competencies)\s*$', re.IGNORECASE)


def extract_achievements(sections: List[Tuple[str, str]], filterer: ResumeFilter,
                         contact_name: str = '') -> List[str]:
    achievement_blocks = [text for name, text in sections if name == 'achievements' and text]
    lines: List[str] = []

    # Derive name tokens dynamically from the candidate's own name so we do not
    # hardcode any specific person.
    name_terms = {'email', 'phone', 'linkedin', 'github'}
    for part in re.split(r'\s+', contact_name.lower()):
        part = part.strip()
        if len(part) > 1:
            name_terms.add(part)

    for block in achievement_blocks:
        for line in block.split('\n'):
            stripped = line.strip()
            if not stripped:
                continue

            # Strip a leading bullet marker BEFORE noise checks.
            content = stripped.lstrip('•-* ').strip()
            lower = content.lower().rstrip(':').strip()

            if lower in _ACHIEVEMENT_CATEGORY_LABELS or lower + 's' in _ACHIEVEMENT_CATEGORY_LABELS:
                continue
            if lower in _ACHIEVEMENT_GENERIC_SKIP:
                continue
            if any(term in lower for term in name_terms) or re.search(r'\+?\d[\d\s\-]{7,}\d', content):
                continue
            if len(content) < 8:
                continue

            item = _TRAILING_LABEL_RE.sub('', content).strip().rstrip(',')
            if item and len(item) >= 8:
                lines.append(item)

    # Remove duplicates while preserving order.
    seen = set()
    unique_lines = []
    for line in lines:
        key = re.sub(r'[^a-z0-9]', '', line.lower())
        if key and key not in seen:
            seen.add(key)
            unique_lines.append(line)

    return sorted(unique_lines, key=lambda line: score_text(line, filterer), reverse=True)[:5]
