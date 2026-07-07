import re
from typing import Dict, List, Optional, Tuple
from app.core.resume_filter import ResumeFilter
from app.generator.llm_reviewer import LLMResumeReviewer, create_reviewer
from app.generator.llm_tailor import create_tailor, LLMResumeTailor


class ResumeGenerator:
    """Deterministic ATS resume generator that uses only the candidate's own data."""

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

    # Common section headers to help with splitting (case-insensitive)
    SECTION_HEADERS = list(sorted(SECTION_MAP.keys(), key=len, reverse=True))

    # Known glued tech terms used to re-insert spaces lost during PDF extraction.
    _PREFIX_FIX_TERMS = [
        'LangChain', 'LangGraph', 'FastAPI', 'FAISS', 'BM25', 'MiniLM', 'DevOps',
        'MongoDB', 'PostgreSQL', 'MySQL', 'JavaScript', 'TypeScript', 'PyTorch',
        'TensorFlow', 'NumPy', 'GitHub', 'Postman', 'Docker', 'Kubernetes',
        'Flask', 'Django', 'React', 'Express', 'OWASP', 'Azure',
    ]
    _SUFFIX_FIX_TERMS = ['FAISS', 'BM25', 'RAG', 'LLM', 'SSE', 'TTL', 'JWT', 'OWASP']

    def __init__(self):
        pass

    @staticmethod
    def _normalize_text(text: str) -> str:
        """
        Normalize extracted PDF text:
        - Fix line endings
        - Detect and preserve section headers on their own lines
        - Identify bullet points (•, -, *)
        - Identify contact info lines
        - Merge run-on lines that are not bullets, headers, or contact lines
        - Collapse multiple blanks to a single blank line
        """
        # Standardize line endings
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        lines = text.split('\n')
        normalized: List[str] = []
        previous_blank = True

        # Pattern for section headers
        section_headers = sorted(ResumeGenerator.SECTION_MAP.keys(), key=len, reverse=True)
        header_pattern = re.compile(
            r'^(?P<header>' + '|'.join(re.escape(header) for header in section_headers) + r')(?P<rest>[\s:–—-].*)?$',
            re.IGNORECASE
        )

        for line in lines:
            stripped = line.strip()

            # Preserve blank lines as a single blank line
            if not stripped:
                normalized.append('')
                previous_blank = True
                continue

            # Detect section headers
            header_match = header_pattern.match(stripped)
            if header_match:
                header = header_match.group('header').strip()
                rest = header_match.group('rest') or ''
                rest = rest.lstrip(' :–—-').strip()
                normalized.append(header)
                if rest:
                    normalized.append(rest)
                previous_blank = False
                continue

            # Detect contact lines and bullet points
            is_contact_line = bool(re.search(
                r'[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}|\+?\d[\d\s\-]{7,}\d|linkedin\.com|github\.com',
                stripped, re.IGNORECASE))
            is_bullet = stripped.startswith(('•', '-', '*'))
            is_category = bool(re.match(r'^[A-Za-z][A-Za-z /&]+:', stripped))

            # Handle special bullet characters from PDF extraction
            if stripped.startswith('�'):
                cleaned_stripped = '•' + stripped[1:]
                normalized.append(cleaned_stripped)
                previous_blank = False
                continue

            if is_bullet or is_contact_line or is_category:
                normalized.append(stripped)
                previous_blank = False
                continue

            # Merge with previous if not after a blank and previous line was content
            if not previous_blank:
                prev = normalized[-1]
                # Add a space if both prev and current are non-empty and joining word characters
                if prev and prev[-1].isalnum() and stripped and stripped[0].isalnum():
                    normalized[-1] = prev + ' ' + stripped
                else:
                    normalized[-1] = prev + stripped
            else:
                normalized.append(stripped)
            previous_blank = False

        # Collapse multiple blanks to a single blank, but preserve single
        # blank lines as paragraph/section separators
        result: List[str] = []
        blank_count = 0
        for line in normalized:
            if line == '':
                blank_count += 1
            else:
                if blank_count > 0:
                    result.append('')  # single separator
                    blank_count = 0
                # Normalize internal whitespace and common PDF artifacts
                cleaned = re.sub(r'\s+', ' ', line).strip()
                result.append(cleaned)
        return '\n'.join(result).strip()

    @staticmethod
    def _escape_latex(text: str) -> str:
        """
        Escape LaTeX special characters. Must NOT escape backslash first;
        do it in a single pass or with a proper ordered dict.
        """
        # IMPORTANT: backslash must be escaped first, but we must use
        # a replacement strategy that doesn't double-escape.
        text = text.replace('\\', r'\textbackslash{}')
        text = text.replace('&', r'\&')
        text = text.replace('%', r'\%')
        text = text.replace('$', r'\$')
        text = text.replace('#', r'\#')
        text = text.replace('_', r'\_')
        text = text.replace('{', r'\{')
        text = text.replace('}', r'\}')
        text = text.replace('~', r'\textasciitilde{}')
        text = text.replace('^', r'\textasciicircum{}')
        return text

    @staticmethod
    def _score_text(text: str, filterer: ResumeFilter) -> int:
        return filterer._score(text)

    def _split_sections(self, text: str) -> List[Tuple[str, str]]:
        """
        Split normalized text into sections based on known section headers.
        Returns a list of (section_key, section_body) tuples.
        """
        sections: List[Tuple[str, str]] = []
        current_name = 'front'  # everything before the first section header
        current_lines: List[str] = []

        section_headers = sorted(self.SECTION_MAP.keys(), key=len, reverse=True)
        header_pattern = re.compile(
            r'^(?P<header>' + '|'.join(re.escape(header) for header in section_headers) + r')(?P<rest>[\s:–—-].*)?$',
            re.IGNORECASE
        )

        for line in text.split('\n'):
            normalized_line = line.strip()
            if not normalized_line:
                continue

            header_match = header_pattern.match(normalized_line)
            if header_match:
                if current_lines:
                    sections.append((current_name, '\n'.join(current_lines).strip()))
                key = header_match.group('header').strip().lower()
                current_name = self.SECTION_MAP.get(key, current_name)
                rest = header_match.group('rest') or ''
                rest = rest.lstrip(' :–—-').strip()
                current_lines = [rest] if rest else []
                continue

            current_lines.append(normalized_line)

        if current_lines:
            sections.append((current_name, '\n'.join(current_lines).strip()))

        return sections

    def _extract_contact(self, text: str) -> Dict[str, str]:
        contact: Dict[str, str] = {'name': '', 'email': '', 'phone': '', 'linkedin': '', 'github': ''}
        lines = [line.strip() for line in text.split('\n') if line.strip()]

        email_match = re.search(r'[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}', text)
        if email_match:
            contact['email'] = email_match.group(0)

        phone_match = re.search(r'(?:\+?\d[\d\s\-]{7,}\d)', text)
        if phone_match:
            contact['phone'] = phone_match.group(0).strip()

        linkedin_match = re.search(r'(https?://)?(www\.)?linkedin\.com/[A-Za-z0-9_\-./]+', text, re.IGNORECASE)
        if linkedin_match:
            contact['linkedin'] = linkedin_match.group(0).strip()

        github_match = re.search(r'(https?://)?(www\.)?github\.com/[A-Za-z0-9_\-./]+', text, re.IGNORECASE)
        if github_match:
            contact['github'] = github_match.group(0).strip()

        if lines:
            name_candidates = [line for line in lines[:2] if not re.search(
                r'[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}|\+?\d[\d\s\-]{7,}\d|linkedin\.com|github\.com',
                line, re.IGNORECASE)]
            if name_candidates:
                contact['name'] = name_candidates[0]
            else:
                contact['name'] = lines[0]

        return contact

    def _extract_best_summary(self, sections: List[Tuple[str, str]], filterer: ResumeFilter) -> str:
        candidates = [text for name, text in sections if name == 'summary' and text]
        if not candidates:
            return ''

        best = max(candidates, key=lambda t: self._score_text(t, filterer))
        summary = best.replace('\n', ' ').strip()
        summary = re.sub(r'\s+', ' ', summary)
        words = summary.split()
        if len(words) > 70:
            summary = ' '.join(words[:70])
            last_period = summary.rfind('.')
            if last_period > 40:
                summary = summary[:last_period + 1]
        return summary

    def _extract_education(self, sections: List[Tuple[str, str]]) -> Dict[str, str]:
        education_blocks = [text for name, text in sections if name == 'education' and text]
        education = {'degree': '', 'institution': '', 'dates': '', 'location': ''}
        if not education_blocks:
            return education

        raw = education_blocks[0].replace('\n', ' ').strip()
        raw = re.sub(r'\s+', ' ', raw)
        raw = re.sub(r'Pre-University Course \(PUC\).*', '', raw, flags=re.IGNORECASE).strip()

        # Try to extract degree (e.g. "B.Tech", "Bachelor of Science", "Master of ...")
        degree_match = re.search(
            r'((?:Bachelor|Master|B\.?Tech|M\.?Tech|B\.?E\.?|M\.?S\.?|B\.?Sc|M\.?Sc|MBA|Ph\.?D)[^,\n]*)',
            raw, re.IGNORECASE
        )
        if degree_match:
            education['degree'] = degree_match.group(0).strip()

        # Institution: look for common institution keywords, otherwise leave blank.
        institution_match = re.search(
            r'([A-Z][A-Za-z.&]*(?:\s+[A-Z][A-Za-z.&]*)*\s+(?:University|Institute|College|School|Academy)(?:\s+of\s+[A-Za-z ]+)?)',
            raw
        )
        if institution_match:
            education['institution'] = institution_match.group(0).strip().rstrip(',')

        date_match = re.search(
            r'((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec) ?\d{4}) ?[–-] ?(?:Present|present|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec) ?\d{4})',
            raw
        )
        if date_match:
            education['dates'] = date_match.group(0).replace('–', '--').replace('—', '--')

        # Location: capture a trailing "City, Region"/"City, Country" style token.
        location_match = re.search(r'([A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)*,\s*[A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)*)\s*$', raw)
        if location_match:
            education['location'] = location_match.group(1).strip()

        return education

    def _split_list_items(self, text: str) -> List[str]:
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        items = []
        for line in lines:
            if line.startswith(('•', '-', '*')):
                items.append(line.lstrip('•-* ').strip())
            else:
                items.append(line)
        return items

    def _extract_skills(self, sections: List[Tuple[str, str]], filterer: ResumeFilter, jd_text: str) -> List[str]:
        skills_blocks = [text for name, text in sections if name == 'skills' and text]
        lines: List[str] = []
        for block in skills_blocks:
            lines.extend(self._split_list_items(block))

        categories: Dict[str, str] = {}
        for line in lines:
            if ':' in line:
                key, value = line.split(':', 1)
                categories[key.strip()] = value.strip()
            elif ',' in line:
                categories[line] = line

        if not categories:
            return []

        # Merge duplicate / synonymous categories (e.g. "Languages" and
        # "Programming Languages") so the same skills are not listed twice.
        canonical_map = {
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
        merged: Dict[str, List[str]] = {}
        for key, value in categories.items():
            canonical = canonical_map.get(key.strip().lower(), key.strip())
            items = [item.strip() for item in re.split(r',\s*', value) if item.strip()]
            bucket = merged.setdefault(canonical, [])
            for item in items:
                if item.lower() not in {existing.lower() for existing in bucket}:
                    bucket.append(item)
        categories = {name: ', '.join(items) for name, items in merged.items()}

        jd_keywords = filterer._get_jd_keywords()
        relevant = []
        for key, value in categories.items():
            combined = f'{key}: {value}'.lower()
            combined_tokens = set(re.findall(r'[a-z0-9+#.]+', combined))
            if combined_tokens & jd_keywords:
                relevant.append(f'{key}: {value}')

        if not relevant:
            priority = ['Languages', 'AI/ML', 'Web & Backend', 'Backend & APIs', 'Databases', 'Tools & Platforms', 'Security', 'Core Competencies']
            for name in priority:
                if name in categories:
                    relevant.append(f'{name}: {categories[name]}')
            if not relevant:
                relevant = [f'{k}: {v}' for k, v in list(categories.items())[:5]]

        return relevant[:6]

    def _parse_experience_block(self, block: str, filterer: ResumeFilter) -> Optional[Dict[str, object]]:
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

        # Preserve the raw header (with its original spacing) so we can use the
        # 2+ space column separators common in resumes to split title / company
        # / location / dates without hardcoding any specific values.
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

        # Split the header into columns on runs of 2+ spaces (falls back to the
        # whole header as a single column for single-spaced formats).
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

        # Derive title & company from the remaining text columns.
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

        scored_bullets = sorted(
            bullets, key=lambda b: self._score_text(b, filterer), reverse=True
        )
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

    def _extract_experience(self, sections: List[Tuple[str, str]], filterer: ResumeFilter) -> List[Dict[str, object]]:
        experience_blocks = [text for name, text in sections if name == 'experience' and text]
        entries: List[Tuple[int, int, Dict[str, object]]] = []
        for block in experience_blocks:
            entry = self._parse_experience_block(block, filterer)
            if not entry or not entry['company']:
                continue
            score = self._score_text(block, filterer)
            bullet_count = len(entry.get('bullets', []))
            entries.append((bullet_count, score, entry))

        entries.sort(key=lambda pair: (pair[0], pair[1]), reverse=True)
        return [entry for _, _, entry in entries][:1]

    def _parse_project_header(self, header: str) -> Tuple[str, str, str]:
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
                title, re.IGNORECASE
            )
            if tech_match:
                tech = tech_match.group('tech').strip()
                title = title[:tech_match.start()].strip()

        if tech and year and title.endswith(year):
            title = title[:title.rfind(year)].strip()

        return title, tech, year

    def _extract_projects(self, sections: List[Tuple[str, str]], filterer: ResumeFilter) -> List[Dict[str, object]]:
        projects = self._parse_all_project_blocks(sections)

        scored: List[Tuple[int, Dict[str, object]]] = []
        for project in projects:
            if not project['bullets']:
                continue
            score = self._score_text(project['header'] + ' ' + ' '.join(project['bullets']), filterer)
            scored.append((score, project))

        scored.sort(key=lambda pair: pair[0], reverse=True)

        # Keep only projects that are actually relevant to the JD (positive score).
        # Never pad the list with an unrelated project just to reach 2. If nothing
        # scores (e.g. a very short JD), fall back to the single top project so the
        # section is not empty.
        relevant = [(score, project) for score, project in scored if score > 0]
        if relevant:
            selection = relevant[:2]
        elif scored:
            selection = scored[:1]
        else:
            selection = []

        chosen = []
        for _, project in selection:
            chosen.append(self._finalise_project(project))

        return chosen[:2]

    def _parse_all_project_blocks(self, sections: List[Tuple[str, str]]) -> List[Dict[str, object]]:
        """Parse the raw project section(s) into a deduped list of {header, bullets}."""
        project_blocks = [text for name, text in sections if name == 'projects' and text]
        projects: List[Dict[str, object]] = []

        tech_keywords = r'(?:Flask|Python|Django|MongoDB|Express(?:\.js)?|React(?:\.js)?|Node(?:\.js)?|PostgreSQL|FAISS|LangChain|RAG|SQL|REST APIs|REST API|REST|Docker|AWS|Azure|GCP|PyTorch|TensorFlow|HTML|CSS|JWT)'
        header_boundary = re.compile(
            rf'(?<=[\.\n])\s*(?=[A-Z][A-Za-z0-9 &()\-]+.*?(?:\||{tech_keywords}))',
            re.IGNORECASE
        )
        # A new project header can get merged into the tail of the previous
        # bullet during normalization (e.g. "...uptime NetScanner | Python | 2020").
        # Detect a capitalized title followed by " | tech" or " | year" and split
        # a fresh line before it, even mid-sentence.
        new_project_header = re.compile(
            r'(?<=[a-z0-9%)])\s*'
            r'(?=[A-Z][A-Za-z0-9]+(?:\s+[A-Z][A-Za-z0-9]+){0,3}\s*\|\s*'
            rf'(?:{tech_keywords}|\d{{4}}))'
        )
        tech_separator = re.compile(rf'(?<=[^\s\|\-–—])(?={tech_keywords})', re.IGNORECASE)
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

        # Remove duplicates by normalised title (ignore tech stack differences)
        seen_titles = set()
        deduped = []
        for p in projects:
            # Extract title part before | or —
            raw_header = p['header']
            title_part = raw_header.split('|')[0].split('—')[0].split('–')[0].strip()
            norm_title = re.sub(r'[^a-z0-9]', '', title_part.lower())
            if norm_title not in seen_titles:
                seen_titles.add(norm_title)
                deduped.append(p)
        return deduped

    def _finalise_project(self, project: Dict[str, object]) -> Dict[str, object]:
        """Turn a raw {header, bullets} project into the rendered {title, tech, year, bullets}."""
        title, tech, year = self._parse_project_header(project['header'])
        bullets = list(project['bullets'])
        # If the tech stack leaked in as the first bullet (e.g. "Python, Flask"),
        # promote it to the tech field instead of showing it as a bullet.
        if not tech and bullets and self._looks_like_tech_list(bullets[0]):
            tech = bullets.pop(0).strip().rstrip('.')
        return {
            'title': title or project['header'],
            'tech': tech,
            'year': year,
            'bullets': bullets[:3],
        }


    @staticmethod
    def _looks_like_tech_list(text: str) -> bool:
        """Heuristic: does this short line look like a tech stack rather than a sentence?"""
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

    def _extract_achievements(self, sections: List[Tuple[str, str]], filterer: ResumeFilter, contact_name: str = '') -> List[str]:
        achievement_blocks = [text for name, text in sections if name == 'achievements' and text]
        lines: List[str] = []

        # Common category headers and noise to skip
        category_labels = {
            'problem solving', 'certifications', 'awards', 'achievements',
            'soft skills', 'soft skill', 'technical skills', 'technical', 'skills',
            'contact', 'profile', 'summary', 'experience', 'projects', 'education',
        }
        # Derive name tokens dynamically from the candidate's own name so we do
        # not hardcode any specific person.
        name_terms = {'email', 'phone', 'linkedin', 'github'}
        for part in re.split(r'\s+', contact_name.lower()):
            part = part.strip()
            if len(part) > 1:
                name_terms.add(part)
        # Generic soft-skill phrases that are not real achievements
        generic_skip = {
            'technical research', 'team collaboration', 'problem solving',
            'communication skills', 'interpersonal skills', 'adaptability',
            'leadership', 'teamwork', 'time management', 'critical thinking',
        }
        trailing_label_re = re.compile(
            r'\s*\.?\s*(soft skills?|technical skills?|core competencies)\s*$',
            re.IGNORECASE
        )

        def _clean_candidate(value: str) -> str:
            return trailing_label_re.sub('', value).strip().rstrip(',')

        for block in achievement_blocks:
            for line in block.split('\n'):
                stripped = line.strip()
                if not stripped:
                    continue

                # Strip any leading bullet marker BEFORE noise checks so that
                # "- Team Collaboration" is recognised as a generic soft skill.
                content = stripped.lstrip('•-* ').strip()
                lower = content.lower().rstrip(':').strip()

                # Skip pure category headers
                if lower in category_labels or lower + 's' in category_labels:
                    continue

                # Skip generic soft-skill phrases
                if lower in generic_skip:
                    continue

                # Skip the candidate's name or contact info
                if any(term in lower for term in name_terms) or re.search(r'\+?\d[\d\s\-]{7,}\d', content):
                    continue

                # Skip very short or single-word items
                if len(content) < 8:
                    continue

                item = _clean_candidate(content)
                if item and len(item) >= 8:
                    lines.append(item)

        # Remove duplicates while preserving order
        seen = set()
        unique_lines = []
        for line in lines:
            key = re.sub(r'[^a-z0-9]', '', line.lower())
            if key and key not in seen:
                seen.add(key)
                unique_lines.append(line)

        scored = sorted(unique_lines, key=lambda line: self._score_text(line, filterer), reverse=True)
        return scored[:5]

    @staticmethod
    def _render_contact(contact: Dict[str, str]) -> str:
        name = ResumeGenerator._escape_latex(contact.get('name', ''))
        phone = ResumeGenerator._escape_latex(contact.get('phone', ''))
        email = ResumeGenerator._escape_latex(contact.get('email', ''))
        linkedin = contact.get('linkedin', '')
        github = contact.get('github', '')

        def make_url(value: str) -> str:
            if not value:
                return ''
            if value.startswith('http'):
                return value
            return 'https://' + value.lstrip('/')

        linkedin = ResumeGenerator._escape_latex(make_url(linkedin)) if linkedin else ''
        github = ResumeGenerator._escape_latex(make_url(github)) if github else ''

        lines = [f'{{\\Huge \\textbf{{{name}}}}} \\\\']
        lines.append('\\small')
        contact_parts = []
        if phone:
            contact_parts.append(phone)
        if email:
            contact_parts.append('\\href{mailto:%s}{%s}' % (email, email))
        if linkedin:
            contact_parts.append('\\href{%s}{LinkedIn}' % linkedin)
        if github:
            contact_parts.append('\\href{%s}{GitHub}' % github)
        lines.append(' $\\mid$ '.join(contact_parts))
        return '\n'.join(lines)

    @staticmethod
    def _repair_text(text: str) -> str:
        """Safe, deterministic cleanup of common PDF-extraction artifacts."""
        if not text:
            return text
        # Normalise Unicode punctuation to LaTeX-safe ASCII so pdflatex compiles
        # cleanly even without inputenc loaded.
        unicode_map = {
            '\u2011': '-', '\u2012': '-', '\u2013': '--', '\u2014': '---',
            '\u2015': '---', '\u2018': "'", '\u2019': "'", '\u201c': '``',
            '\u201d': "''", '\u2026': '...', '\u00a0': ' ', '\u2022': '',
            '\u00b7': '', '\u2009': ' ', '\u200b': '',
        }
        for src, dst in unicode_map.items():
            text = text.replace(src, dst)
        # Re-insert spaces lost around known tech terms (e.g. "usingLangChain").
        for term in ResumeGenerator._PREFIX_FIX_TERMS:
            text = re.sub(r'(?<=[A-Za-z0-9])(' + re.escape(term) + r')', r' \1', text)
        for term in ResumeGenerator._SUFFIX_FIX_TERMS:
            # Split "RAGpipelines" -> "RAG pipelines" but NOT a plural like "LLMs".
            text = re.sub(r'(' + re.escape(term) + r')(?=[a-z])(?!s\b)', r'\1 ', text)
        # Add a space after a comma/semicolon when directly followed by a letter
        # (e.g. "Express.js,React.js" -> "Express.js, React.js"); leaves numbers
        # like "15,000" untouched.
        text = re.sub(r'([,;])(?=[A-Za-z])', r'\1 ', text)
        # Remove stray spaces before punctuation (e.g. "NLP ," -> "NLP,")
        text = re.sub(r'\s+([,.;:])', r'\1', text)
        # Drop unicode replacement/stray bullet artifacts
        text = text.replace('\ufffd', '').replace('●', '')
        # Collapse repeated whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    @classmethod
    def _clean_latex(cls, text: str) -> str:
        """Repair then LaTeX-escape a piece of text."""
        return cls._escape_latex(cls._repair_text(text))

    @staticmethod
    def _normalize_bullet(text: str) -> str:
        """
        Enforce bullet consistency for ATS: capitalized first letter and a single
        trailing period. Trims stray leading markers and duplicate end punctuation.
        """
        b = str(text).strip()
        if not b:
            return b
        # Strip any leading bullet markers the source may have carried in.
        b = re.sub(r'^[\s•\-\*\u2013\u2014]+', '', b).strip()
        if not b:
            return b
        # Capitalize the first alphabetic character.
        for i, ch in enumerate(b):
            if ch.isalpha():
                b = b[:i] + ch.upper() + b[i + 1:]
                break
        # Normalize trailing punctuation to exactly one period (leave ? and ! alone).
        b = b.rstrip()
        b = re.sub(r'[\s;,]+$', '', b)
        if not b.endswith(('.', '!', '?')):
            b += '.'
        else:
            # Collapse any run of trailing periods to a single one.
            b = re.sub(r'\.\.+$', '.', b)
        return b

    def _render_latex(
        self,
        contact: Dict[str, str],
        summary: str,
        education: Dict[str, str],
        skills: List[object],
        experience: List[Dict[str, object]],
        projects: List[Dict[str, object]],
        achievements: List[str]
    ) -> str:
        def resume_section_heading(title: str) -> str:
            return f'\\section*{{{title}}}'

        latex = [
            '\\documentclass[letterpaper,11pt]{article}',
            '\\usepackage[top=0.45in,bottom=0.45in,left=0.6in,right=0.6in]{geometry}',
            '\\usepackage{enumitem}',
            '\\usepackage[hidelinks]{hyperref}',
            '\\usepackage{titlesec}',
            '\\usepackage{xcolor}',
            '\\usepackage{microtype}',
            '\\pagestyle{empty}',
            '\\definecolor{accent}{HTML}{1f3a5f}',
            '\\newcommand{\\resumeItem}[1]{\\item\\small{#1}}',
            '\\newcommand{\\resumeSubheading}[4]{%',
            '  \\vspace{2pt}\\textbf{#1} \\hfill \\textbf{\\small #2} \\\\',
            '  \\textit{\\small #3} \\hfill \\textit{\\small #4}\\vspace{-1pt}}',
            '\\setlist[itemize]{leftmargin=1.4em, topsep=3pt, itemsep=2pt, parsep=0pt}',
            '\\setlength{\\parskip}{2pt}',
            '\\setlength{\\parindent}{0pt}',
            '\\raggedright',
            '\\titleformat{\\section}{\\vspace{4pt}\\scshape\\large\\bfseries\\color{accent}}'
            '{}{0em}{}[{\\color{accent}\\titlerule}\\vspace{-1pt}]',
            '\\titlespacing*{\\section}{0pt}{8pt}{5pt}',
            '\\begin{document}',
            '\\begin{center}',
            self._render_contact(contact),
            '\\end{center}',
            '\\vspace{-2pt}',
        ]

        if summary:
            latex.extend([
                resume_section_heading('Professional Summary'),
                ResumeGenerator._clean_latex(summary),
            ])

        if education.get('degree') or education.get('institution'):
            degree = self._repair_text(education.get('degree', ''))
            gpa = self._repair_text(education.get('gpa', ''))
            if gpa:
                degree = f'{degree} (GPA: {gpa})' if degree else f'GPA: {gpa}'
            latex.extend([
                resume_section_heading('Education'),
                '\\resumeSubheading{%s}{%s}{%s}{%s}' % (
                    ResumeGenerator._clean_latex(education.get('institution', '') or degree),
                    ResumeGenerator._clean_latex(education.get('dates', '')),
                    ResumeGenerator._escape_latex(degree),
                    ResumeGenerator._clean_latex(education.get('location', ''))
                )
            ])

        if skills:
            latex.extend([
                resume_section_heading('Technical Skills'),
                '\\begin{itemize}[leftmargin=1.4em,label={}]'
            ])
            for skill in skills:
                if isinstance(skill, dict):
                    category = skill.get('category', '')
                    items = skill.get('items', '')
                else:
                    text = str(skill)
                    if ':' in text:
                        category, items = text.split(':', 1)
                    else:
                        category, items = '', text
                category = category.strip()
                items = items.strip()
                if not items:
                    continue
                if category:
                    latex.append('  \\item \\small{\\textbf{%s:} %s}' % (
                        ResumeGenerator._clean_latex(category),
                        ResumeGenerator._clean_latex(items)
                    ))
                else:
                    latex.append('  \\item \\small{%s}' % ResumeGenerator._clean_latex(items))
            latex.append('\\end{itemize}')

        if experience:
            latex.append(resume_section_heading('Experience'))
            for entry in experience:
                latex.append('\\resumeSubheading{%s}{%s}{%s}{%s}' % (
                    ResumeGenerator._clean_latex(str(entry.get('role', ''))),
                    ResumeGenerator._clean_latex(str(entry.get('dates', ''))),
                    ResumeGenerator._clean_latex(str(entry.get('company', ''))),
                    ResumeGenerator._clean_latex(str(entry.get('location', '')))
                ))
                bullets = entry.get('bullets', []) or []
                if bullets:
                    latex.append('\\begin{itemize}')
                    for bullet in bullets:
                        latex.append('  \\resumeItem{%s}' % ResumeGenerator._clean_latex(
                            ResumeGenerator._normalize_bullet(str(bullet))))
                    latex.append('\\end{itemize}')

        if projects:
            latex.append(resume_section_heading('Projects'))
            for project in projects:
                title = ResumeGenerator._clean_latex(str(project.get('title', '')))
                tech = self._repair_text(str(project.get('tech', '')))
                year = ResumeGenerator._clean_latex(str(project.get('year', '')))
                heading = '\\textbf{%s}' % title
                if tech:
                    heading += ' $|$ \\textit{\\small %s}' % ResumeGenerator._escape_latex(tech)
                if year:
                    heading += ' \\hfill \\small %s' % year
                latex.append(heading + ' \\\\')
                bullets = project.get('bullets', []) or []
                if bullets:
                    latex.append('\\begin{itemize}')
                    for bullet in bullets:
                        latex.append('  \\resumeItem{%s}' % ResumeGenerator._clean_latex(
                            ResumeGenerator._normalize_bullet(str(bullet))))
                    latex.append('\\end{itemize}')

        if achievements:
            latex.extend([
                resume_section_heading('Achievements \\& Certifications'),
                '\\begin{itemize}'
            ])
            for achievement in achievements:
                latex.append('  \\resumeItem{%s}' % ResumeGenerator._clean_latex(
                    ResumeGenerator._normalize_bullet(str(achievement))))
            latex.append('\\end{itemize}')

        latex.append('\\end{document}')
        return '\n'.join(latex)

    def _report_match(self, jd_text: str, content: Dict[str, object]) -> None:
        """Print an ATS hard-keyword coverage report for user feedback."""
        keywords = LLMResumeTailor.extract_jd_keywords(jd_text)
        if not keywords:
            return

        blob_parts = [str(content.get('summary', ''))]
        for skill in content.get('skills', []) or []:
            if isinstance(skill, dict):
                blob_parts.append(f"{skill.get('category', '')} {skill.get('items', '')}")
            else:
                blob_parts.append(str(skill))
        for exp in content.get('experience', []) or []:
            blob_parts.append(str(exp.get('role', '')))
            blob_parts.extend(str(b) for b in exp.get('bullets', []) or [])
        for proj in content.get('projects', []) or []:
            blob_parts.append(f"{proj.get('title', '')} {proj.get('tech', '')}")
            blob_parts.extend(str(b) for b in proj.get('bullets', []) or [])
        blob = ' '.join(blob_parts).lower()

        matched = [k for k in keywords if k.lower() in blob]
        missing = [k for k in keywords if k.lower() not in blob]
        coverage = (len(matched) / len(keywords)) * 100 if keywords else 0

        print(f"[ATS] Hard-keyword coverage: {coverage:.0f}% ({len(matched)}/{len(keywords)})")
        if missing:
            preview = ', '.join(missing[:12])
            print(f"[ATS] Keywords not surfaced (verify if truthful): {preview}")

        self._report_quality(content)

    @staticmethod
    def _report_quality(content: Dict[str, object]) -> None:
        """Warn about the quality signals ATS sites score: metrics and repetition."""
        bullets: List[str] = []
        for exp in content.get('experience', []) or []:
            bullets.extend(str(b) for b in exp.get('bullets', []) or [])
        for proj in content.get('projects', []) or []:
            bullets.extend(str(b) for b in proj.get('bullets', []) or [])

        if not bullets:
            return

        # Quantify-impact check: how many bullets contain a number/metric.
        metric_re = re.compile(r'\d')
        quantified = sum(1 for b in bullets if metric_re.search(b))
        pct = (quantified / len(bullets)) * 100
        print(f"[ATS] Quantified bullets: {pct:.0f}% ({quantified}/{len(bullets)}) contain a metric.")
        if pct < 70:
            print("[ATS] Tip: add more numbers (%, counts, $, latency) to experience/project bullets.")

        # Repetition check: leading action-verb variety.
        leading = []
        for b in bullets:
            m = re.match(r'\s*([A-Za-z]+)', b)
            if m:
                leading.append(m.group(1).lower())
        dupes = {v for v in leading if leading.count(v) > 1}
        if dupes:
            print(f"[ATS] Repeated leading verbs (vary these): {', '.join(sorted(dupes))}")

    @staticmethod
    def _content_is_empty(content: Dict[str, object]) -> bool:
        """True when the tailored content has no substantive body sections."""
        if not content:
            return True
        for key in ('summary', 'skills', 'experience', 'projects', 'achievements'):
            value = content.get(key)
            if isinstance(value, str):
                if value.strip():
                    return False
            elif value:  # non-empty list/dict
                return False
        return True

    def _build_minimal_content(self, content: Dict[str, object], master_text: str) -> Dict[str, object]:
        """
        Build a minimal but non-empty resume body from the raw master resume text.

        Used only as a last resort when structured extraction fails, so that a
        new user still receives a usable output file instead of a blank resume.
        """
        base = dict(content or {})

        # Turn the most informative non-empty lines into summary + bullet points.
        lines = [ln.strip(' \t•-*') for ln in master_text.split('\n')]
        lines = [ln for ln in lines if len(ln) > 25]

        if not base.get('summary'):
            base['summary'] = lines[0] if lines else 'Professional resume.'

        if not base.get('skills'):
            base['skills'] = []
        if not base.get('education'):
            base['education'] = base.get('education', {}) or {}

        if not base.get('experience'):
            highlight_bullets = lines[1:7] if len(lines) > 1 else lines[:6]
            if highlight_bullets:
                base['experience'] = [{
                    'company': '',
                    'role': 'Highlights',
                    'dates': '',
                    'location': '',
                    'bullets': highlight_bullets[:5],
                }]
            else:
                base['experience'] = []

        if not base.get('projects'):
            base['projects'] = []
        if not base.get('achievements'):
            base['achievements'] = []

        return base

    @staticmethod
    def _norm_title(value: str) -> str:
        return re.sub(r'[^a-z0-9]', '', str(value).lower())

    def _select_projects_deterministic(
        self,
        sections: List[Tuple[str, str]],
        filterer: ResumeFilter,
        selected_projects: List[str],
    ) -> List[Dict[str, object]]:
        """Deterministic project selection that honours an explicit user choice."""
        if not selected_projects:
            return self._extract_projects(sections, filterer)

        wanted = {self._norm_title(t) for t in selected_projects}
        chosen = []
        for project in self._parse_all_project_blocks(sections):
            title, _tech, _year = self._parse_project_header(project['header'])
            norm = self._norm_title(title or project['header'])
            if any(norm and (norm in w or w in norm) for w in wanted):
                chosen.append(self._finalise_project(project))
        # If nothing matched (e.g. titles differ), fall back to scored selection.
        return chosen[:2] if chosen else self._extract_projects(sections, filterer)

    @staticmethod
    def _merge_extra_skills(skills: List[object], extra_skills: List[str]) -> List[object]:
        """Ensure user-approved JD skills appear in the skills section exactly once."""
        if not extra_skills:
            return skills

        # Collect everything already present (across all categories) for dedup.
        present: set = set()
        for item in skills:
            if isinstance(item, dict):
                blob = f"{item.get('category', '')} {item.get('items', '')}"
            else:
                blob = str(item)
            for tok in re.split(r'[,:/|]', blob.lower()):
                tok = tok.strip()
                if tok:
                    present.add(tok)

        to_add = [s for s in extra_skills if s.strip().lower() not in present]
        if not to_add:
            return skills

        merged = list(skills)
        # Reuse an existing "additional"/"other"/"core" category if one exists.
        for item in merged:
            if isinstance(item, dict) and re.search(r'other|additional|core', str(item.get('category', '')), re.IGNORECASE):
                existing = str(item.get('items', '')).strip()
                item['items'] = (existing + ', ' if existing else '') + ', '.join(to_add)
                return merged

        merged.append({'category': 'Additional Skills', 'items': ', '.join(to_add)})
        return merged

    def _enforce_selected_projects(
        self,
        projects: List[Dict[str, object]],
        selected_projects: List[str],
        sections: List[Tuple[str, str]],
    ) -> List[Dict[str, object]]:
        """Reorder/filter already-generated projects to match the user's selection."""
        if not selected_projects:
            return projects

        wanted = [self._norm_title(t) for t in selected_projects]

        def matches(title: str, want: str) -> bool:
            norm = self._norm_title(title)
            return bool(norm) and (norm in want or want in norm)

        ordered: List[Dict[str, object]] = []
        used = set()
        for want in wanted:
            for idx, proj in enumerate(projects):
                if idx in used:
                    continue
                if matches(str(proj.get('title', '')), want):
                    ordered.append(proj)
                    used.add(idx)
                    break

        # If the generated content did not include a selected project, pull it
        # straight from the parsed master resume so the user's choice is honoured.
        if len(ordered) < len(wanted):
            for want in wanted:
                if any(matches(str(p.get('title', '')), want) for p in ordered):
                    continue
                for raw in self._parse_all_project_blocks(sections):
                    title = self._parse_project_header(raw['header'])[0] or raw['header']
                    if matches(title, want):
                        ordered.append(self._finalise_project(raw))
                        break

        return (ordered or projects)[:2]

    def analyze_for_hitl(self, jd_text: str, master_resume_text: str) -> Dict[str, object]:
        """
        Produce data for a human-in-the-loop review step BEFORE generating the resume.

        Returns:
            {
                'missing_skills':  JD skills/keywords NOT found in the resume,
                'matched_skills':  JD skills/keywords already present in the resume,
                'projects':        [{'title', 'tech'}] for every project in the resume,
            }
        """
        normalized_text = self._normalize_text(master_resume_text)
        sections = self._split_sections(normalized_text)

        jd_keywords = LLMResumeTailor.extract_jd_keywords(jd_text)
        resume_lower = normalized_text.lower()
        matched_skills = [kw for kw in jd_keywords if kw.lower() in resume_lower]
        missing_skills = [kw for kw in jd_keywords if kw.lower() not in resume_lower]

        projects = []
        for project in self._parse_all_project_blocks(sections):
            title, tech, _year = self._parse_project_header(project['header'])
            bullets = list(project['bullets'])
            if not tech and bullets and self._looks_like_tech_list(bullets[0]):
                tech = bullets[0].strip().rstrip('.')
            clean_title = (title or project['header']).strip()
            if clean_title:
                projects.append({'title': clean_title, 'tech': tech.strip()})

        return {
            'missing_skills': missing_skills,
            'matched_skills': matched_skills,
            'projects': projects,
        }

    def tailor_resume(
        self,
        jd_text: str,
        master_resume_text: str,
        use_llm_review: bool = True,
        extra_skills: Optional[List[str]] = None,
        selected_projects: Optional[List[str]] = None,
    ) -> str:
        normalized_text = self._normalize_text(master_resume_text)
        sections = self._split_sections(normalized_text)
        filterer = ResumeFilter(normalized_text, jd_text)

        extra_skills = [s.strip() for s in (extra_skills or []) if s and s.strip()]
        selected_projects = [p.strip() for p in (selected_projects or []) if p and p.strip()]

        # Contact info is factual and safest to extract deterministically.
        contact = self._extract_contact(normalized_text)

        content: Optional[Dict[str, object]] = None

        # Preferred path: LLM-driven, JD-aware tailoring (grounded in the resume).
        if use_llm_review:
            tailor = create_tailor()
            if tailor is None:
                print("[WARN] GROQ_API_KEY not set - skipping AI tailoring. "
                      "Output will use keyword-based selection only and may be less JD-aligned. "
                      "Add your key to a .env file (see .env.example) for best results.")
            else:
                try:
                    content = tailor.tailor(
                        jd_text,
                        normalized_text,
                        extra_skills=extra_skills,
                        selected_projects=selected_projects,
                    )
                    if content is None:
                        print("[WARN] AI tailoring returned no usable content; "
                              "falling back to keyword-based extraction.")
                    else:
                        print("[INFO] AI tailoring applied (JD-aligned).")
                except Exception as exc:
                    print(f"[WARN] AI tailoring failed ({exc}); "
                          "falling back to keyword-based extraction.")
                    content = None

        # Fallback path: deterministic keyword-based extraction. This also runs
        # when the LLM produced a response but it was essentially empty, so a new
        # user always ends up with a populated resume.
        if content is None or self._content_is_empty(content):
            deterministic = {
                'summary': self._extract_best_summary(sections, filterer),
                'education': self._extract_education(sections),
                'skills': self._extract_skills(sections, filterer, jd_text),
                'experience': self._extract_experience(sections, filterer),
                'projects': self._select_projects_deterministic(sections, filterer, selected_projects),
                'achievements': self._extract_achievements(sections, filterer, contact.get('name', '')),
            }
            if content is not None and self._content_is_empty(content):
                print("[WARN] AI tailoring returned empty content; "
                      "using keyword-based extraction instead.")
            content = deterministic

        # Last-resort safety net: if we STILL have nothing usable (e.g. an
        # unstructured resume the parser could not segment), synthesise minimal
        # content from the raw master text so the output is never blank.
        if self._content_is_empty(content):
            print("[WARN] Could not extract structured sections; "
                  "generating a minimal resume from the raw resume text.")
            content = self._build_minimal_content(content, normalized_text)

        # Honour human-in-the-loop choices regardless of which path produced the
        # content: guarantee user-approved skills appear, and enforce the user's
        # project selection when they made one.
        if extra_skills:
            content['skills'] = self._merge_extra_skills(content.get('skills', []) or [], extra_skills)
        if selected_projects:
            content['projects'] = self._enforce_selected_projects(
                content.get('projects', []) or [], selected_projects, sections
            )

        try:
            self._report_match(jd_text, content)
        except Exception:
            pass

        latex = self._render_latex(
            contact,
            str(content.get('summary', '')),
            content.get('education', {}) or {},
            content.get('skills', []) or [],
            content.get('experience', []) or [],
            content.get('projects', []) or [],
            content.get('achievements', []) or [],
        )

        return latex
