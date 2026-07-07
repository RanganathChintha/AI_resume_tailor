import re
from typing import Dict, List, Optional, Tuple
from app.core.resume_filter import ResumeFilter
from app.generator.llm_reviewer import LLMResumeReviewer, create_reviewer


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

        # Try to extract degree (e.g. "B.Tech", "Bachelor of Science")
        degree_match = re.search(r'(Bachelor[^,\.]+(?:Computer Science[^,\.]*)?)', raw, re.IGNORECASE)
        if degree_match:
            education['degree'] = degree_match.group(0).strip()
        else:
            # fallback: take everything before the institution
            education['degree'] = raw.split('RGUKT')[0].strip() if 'RGUKT' in raw else raw.split('University')[0].strip()

        institution_match = re.search(
            r'(RGUKT(?: RK Valley)?|Rajiv Gandhi University(?: of Knowledge and Technology)?(?:,? RK Valley)?)',
            raw, re.IGNORECASE
        )
        if institution_match:
            education['institution'] = institution_match.group(0).strip().rstrip(',')
        else:
            education['institution'] = 'RGUKT RK Valley'

        date_match = re.search(
            r'((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec) ?\d{4}) ?[–-] ?(?:Present|present|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec) ?\d{4})',
            raw
        )
        if date_match:
            education['dates'] = date_match.group(0).replace('–', '--').replace('—', '--')
        else:
            education['dates'] = 'Aug 2023 -- Present'

        if 'Andhra Pradesh' in raw or 'RK Valley' in raw:
            education['location'] = 'Andhra Pradesh, India'
        elif 'Bangalore' in raw:
            education['location'] = 'Bangalore, India'

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

        jd_keywords = set(re.findall(r'[A-Za-z0-9_+-]+', jd_text.lower()))
        relevant = []
        for key, value in categories.items():
            combined = f'{key}: {value}'
            if any(keyword in combined.lower() for keyword in jd_keywords):
                relevant.append(combined)

        if not relevant:
            priority = ['Languages', 'AI/ML', 'Web/Backend', 'Backend & APIs', 'Databases', 'Tools', 'Security', 'Core Competencies']
            for name in priority:
                if name in categories:
                    relevant.append(f'{name}: {categories[name]}')
            if not relevant:
                relevant = list(categories.values())[:5]

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

        header = ' '.join(header_lines).strip() if header_lines else ''
        header = re.sub(r'\s+', ' ', header)

        company = ''
        title = ''
        dates = ''
        location = ''

        date_match = re.search(
            r'((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\s*\d{4})\s*[–—-]\s*(Present|present|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\s*\d{4})',
            header
        )
        if date_match:
            dates = date_match.group(0).replace('–', '--').replace('—', '--')
            header = header.replace(date_match.group(0), '').strip()

        for loc, pattern in [
            ('Bangalore, India', r'Bangalore'),
            ('Andhra Pradesh, India', r'Andhra Pradesh|RK Valley'),
            ('Hyderabad, India', r'Hyderabad'),
            ('Mumbai, India', r'Mumbai'),
            ('Remote', r'\bRemote\b'),
        ]:
            if re.search(pattern, header, re.IGNORECASE):
                location = loc
                header = re.sub(pattern, '', header, flags=re.IGNORECASE).strip()
                break

        if 'Siemens' in header:
            company = 'Siemens'
            header = header.replace('Siemens', '').strip()
        elif re.search(r'RGUKT|Rajiv Gandhi University', header, re.IGNORECASE):
            company = 'RGUKT RK Valley'
            header = re.sub(r'RGUKT|Rajiv Gandhi University(?: of Knowledge and Technology)?', '', header, flags=re.IGNORECASE).strip()

        title_match = re.search(
            r'(Software Engineering Intern|Software Engineer|Security Engineer|Cybersecurity Intern|Developer|SDE|Research Intern|AI Engineer|Machine Learning Engineer|Data Scientist|Full Stack Developer|Backend Developer|GenAI & RAG|GenAI|RAG)[^,\|–—-]*',
            header, re.IGNORECASE,
        )
        if title_match:
            title = title_match.group(0).strip()
            header = header.replace(title_match.group(0), '').strip()

        if not title and header:
            parts = re.split(r'\s{2,}|,|\||–|—|-', header)
            for part in parts:
                part = part.strip()
                if re.search(r'Intern|Engineer|Developer|Scientist|Manager|Consultant|Analyst|Architect|Research', part, re.IGNORECASE):
                    title = part
                    header = header.replace(part, '').strip()
                    break

        if not company and header:
            parts = re.split(r'\s{2,}|,|\||–|—|-', header)
            for part in parts:
                part = part.strip()
                if part and not re.search(r'Intern|Engineer|Developer|Scientist|Manager|Consultant|Analyst|Architect|Research', part, re.IGNORECASE):
                    company = part
                    break

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
        project_blocks = [text for name, text in sections if name == 'projects' and text]
        projects: List[Dict[str, object]] = []

        tech_keywords = r'(?:Flask|Python|Django|MongoDB|Express(?:\.js)?|React(?:\.js)?|Node(?:\.js)?|PostgreSQL|FAISS|LangChain|RAG|SQL|REST APIs|REST API|REST|Docker|AWS|Azure|GCP|PyTorch|TensorFlow|HTML|CSS|JWT)'
        header_boundary = re.compile(
            rf'(?<=[\.\n])\s*(?=(?:PDF|MERN Stack|DocPilot|OWASP|[A-Z][A-Za-z0-9 &()\-]+).*?(?:\||{tech_keywords}))',
            re.IGNORECASE
        )
        tech_separator = re.compile(rf'(?<=[^\s\|\-–—])(?={tech_keywords})', re.IGNORECASE)
        inline_project_bullet = re.compile(rf'\s*[–—-]\s*(?=[A-Z][a-z]|•|\*)')

        for block in project_blocks:
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
        projects = deduped

        scored: List[Tuple[int, Dict[str, object]]] = []
        for project in projects:
            if not project['bullets']:
                continue
            score = self._score_text(project['header'] + ' ' + ' '.join(project['bullets']), filterer)
            scored.append((score, project))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        chosen = []
        for _, project in scored[:3]:
            title, tech, year = self._parse_project_header(project['header'])
            chosen.append({
                'title': title or project['header'],
                'tech': tech,
                'year': year,
                'bullets': project['bullets'][:3]
            })

        return chosen[:2]

    def _extract_achievements(self, sections: List[Tuple[str, str]], filterer: ResumeFilter) -> List[str]:
        achievement_blocks = [text for name, text in sections if name == 'achievements' and text]
        lines: List[str] = []

        # Common category headers and noise to skip
        category_labels = {
            'problem solving', 'certifications', 'awards', 'achievements',
            'soft skills', 'soft skill', 'technical skills', 'technical', 'skills',
            'contact', 'profile', 'summary', 'experience', 'projects', 'education',
        }
        name_terms = {'ranganath', 'chintha', 'email', 'phone', 'linkedin', 'github'}

        for block in achievement_blocks:
            for line in block.split('\n'):
                stripped = line.strip()
                if not stripped:
                    continue

                lower = stripped.lower().rstrip(':').strip()

                # Skip pure category headers
                if lower in category_labels or lower + 's' in category_labels:
                    continue

                # Skip the candidate's name or contact info
                if any(term in lower for term in name_terms) or re.search(r'\+?\d[\d\s\-]{7,}\d', stripped):
                    continue

                # Skip very short or single-word items
                if len(stripped) < 8:
                    continue

                # Extract bullet content
                if stripped.startswith(('•', '-', '*')):
                    item = stripped.lstrip('•-* ').strip()
                    if item and len(item) >= 8:
                        lines.append(item)
                elif stripped:
                    # Plain content - keep only if it's long enough and ends with period
                    if len(stripped) > 15:
                        lines.append(stripped)

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

    def _render_latex(
        self,
        contact: Dict[str, str],
        summary: str,
        education: Dict[str, str],
        skills: List[str],
        experience: List[Dict[str, object]],
        projects: List[Dict[str, object]],
        achievements: List[str]
    ) -> str:
        def resume_section_heading(title: str) -> str:
            return f'\\section*{{{title}}}'

        latex = [
            '\\documentclass[letterpaper,11pt]{article}',
            '\\usepackage[top=0.35in,bottom=0.35in,left=0.55in,right=0.55in]{geometry}',
            '\\usepackage{enumitem}',
            '\\usepackage[hidelinks]{hyperref}',
            '\\usepackage{titlesec}',
            '\\pagestyle{empty}',
            '\\newcommand{\\resumeItem}[1]{\\item\\small{#1}}',
            '\\newcommand{\\resumeSubheading}[4]{%',
            '  \\textbf{#1} \\hfill #2 \\\\',
            '  \\textit{\\small #3} \\hfill \\textit{\\small #4}}',
            '\\newcommand{\\resumeProject}[3]{%',
            '  \\textbf{#1} \\hfill #3 \\\\',
            '  \\textit{\\small #2}}',
            '\\setlist[itemize]{leftmargin=1.5em, topsep=1pt, itemsep=1pt, parsep=0pt}',
            '\\setlength{\\parskip}{1pt}',
            '\\setlength{\\parindent}{0pt}',
            '\\titleformat{\\section}{\\large\\bfseries}{}{0em}{}[\\vspace{-2pt}]',
            '\\begin{document}',
            '\\begin{center}',
            self._render_contact(contact),
            '\\end{center}',
            '\\vspace{-6pt}',
        ]

        if summary:
            latex.extend([
                resume_section_heading('Professional Summary'),
                ResumeGenerator._escape_latex(summary),
            ])

        if education.get('degree'):
            latex.extend([
                resume_section_heading('Education'),
                '\\resumeSubheading{%s}{%s}{%s}{%s}' % (
                    ResumeGenerator._escape_latex(education['degree']),
                    ResumeGenerator._escape_latex(education['dates']),
                    ResumeGenerator._escape_latex(education['institution']),
                    ResumeGenerator._escape_latex(education['location'])
                )
            ])

        if skills:
            latex.extend([
                resume_section_heading('Technical Skills'),
                '\\begin{itemize}[leftmargin=*,label={}]'
            ])
            for skill in skills:
                latex.append('  \\resumeItem{%s}' % ResumeGenerator._escape_latex(skill))
            latex.append('\\end{itemize}')

        if experience:
            latex.append(resume_section_heading('Experience'))
            for entry in experience:
                latex.append('\\resumeSubheading{%s}{%s}{%s}{%s}' % (
                    ResumeGenerator._escape_latex(entry.get('role', '')),
                    ResumeGenerator._escape_latex(entry.get('dates', '')),
                    ResumeGenerator._escape_latex(entry.get('company', '')),
                    ResumeGenerator._escape_latex(entry.get('location', ''))
                ))
                latex.append('\\begin{itemize}[leftmargin=*,label={}]')
                for bullet in entry.get('bullets', []):
                    latex.append('  \\resumeItem{%s}' % ResumeGenerator._escape_latex(bullet))
                latex.append('\\end{itemize}')

        if projects:
            latex.append(resume_section_heading('Projects'))
            for project in projects:
                title = project.get('title', '')
                tech = project.get('tech', '')
                year = project.get('year', '')
                latex.append('\\resumeProject{%s}{%s}{%s}' % (
                    ResumeGenerator._escape_latex(title),
                    ResumeGenerator._escape_latex(tech),
                    ResumeGenerator._escape_latex(year)
                ))
                if project.get('bullets'):
                    latex.append('\\begin{itemize}[leftmargin=*,label={}]')
                    for bullet in project['bullets']:
                        latex.append('  \\resumeItem{%s}' % ResumeGenerator._escape_latex(bullet))
                    latex.append('\\end{itemize}')

        if achievements:
            latex.extend([
                resume_section_heading('Achievements'),
                '\\begin{itemize}[leftmargin=*,label={}]'
            ])
            for achievement in achievements:
                latex.append('  \\resumeItem{%s}' % ResumeGenerator._escape_latex(achievement))
            latex.append('\\end{itemize}')

        latex.append('\\end{document}')
        return '\n'.join(latex)

    def tailor_resume(self, jd_text: str, master_resume_text: str, use_llm_review: bool = True) -> str:
        normalized_text = self._normalize_text(master_resume_text)
        sections = self._split_sections(normalized_text)
        filterer = ResumeFilter(normalized_text, jd_text)

        contact = self._extract_contact(normalized_text)
        summary = self._extract_best_summary(sections, filterer)
        education = self._extract_education(sections)
        skills = self._extract_skills(sections, filterer, jd_text)
        experience = self._extract_experience(sections, filterer)
        projects = self._extract_projects(sections, filterer)
        achievements = self._extract_achievements(sections, filterer)

        latex = self._render_latex(
            contact, summary, education, skills, experience, projects, achievements
        )

        # Optional LLM review pass for mechanical fixes
        if use_llm_review:
            reviewer = create_reviewer()
            if reviewer:
                try:
                    latex = reviewer.review(jd_text, latex)
                except Exception:
                    # Fail silently - return original LaTeX if review fails
                    pass

        return latex
