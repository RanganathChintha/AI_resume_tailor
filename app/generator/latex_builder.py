"""LaTeX escaping and document rendering for the tailored resume.

Takes the structured resume content (contact, summary, education, skills,
experience, projects, achievements) and produces a compilable LaTeX document.
Pure functions, no shared state.
"""
import re
from typing import Dict, List


# Known glued tech terms used to re-insert spaces lost during PDF extraction.
_PREFIX_FIX_TERMS = [
    'LangChain', 'LangGraph', 'FastAPI', 'FAISS', 'BM25', 'MiniLM', 'DevOps',
    'MongoDB', 'PostgreSQL', 'MySQL', 'JavaScript', 'TypeScript', 'PyTorch',
    'TensorFlow', 'NumPy', 'GitHub', 'Postman', 'Docker', 'Kubernetes',
    'Flask', 'Django', 'React', 'Express', 'OWASP', 'Azure',
]
_SUFFIX_FIX_TERMS = ['FAISS', 'BM25', 'RAG', 'LLM', 'SSE', 'TTL', 'JWT', 'OWASP']


def escape_latex(text: str) -> str:
    """Escape LaTeX special characters (backslash first to avoid double-escaping)."""
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


def repair_text(text: str) -> str:
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
    for term in _PREFIX_FIX_TERMS:
        text = re.sub(r'(?<=[A-Za-z0-9])(' + re.escape(term) + r')', r' \1', text)
    for term in _SUFFIX_FIX_TERMS:
        # Split "RAGpipelines" -> "RAG pipelines" but NOT a plural like "LLMs".
        text = re.sub(r'(' + re.escape(term) + r')(?=[a-z])(?!s\b)', r'\1 ', text)
    # Add a space after a comma/semicolon directly followed by a letter, leaving
    # numbers like "15,000" untouched.
    text = re.sub(r'([,;])(?=[A-Za-z])', r'\1 ', text)
    # Remove stray spaces before punctuation (e.g. "NLP ," -> "NLP,").
    text = re.sub(r'\s+([,.;:])', r'\1', text)
    text = text.replace('\ufffd', '').replace('●', '')
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def clean_latex(text: str) -> str:
    """Repair then LaTeX-escape a piece of text."""
    return escape_latex(repair_text(text))


def normalize_bullet(text: str) -> str:
    """Enforce ATS bullet consistency: capitalized first letter and a single
    trailing period. Trims stray leading markers and duplicate end punctuation."""
    b = str(text).strip()
    if not b:
        return b
    b = re.sub(r'^[\s•\-\*\u2013\u2014]+', '', b).strip()
    if not b:
        return b
    # Capitalize the first alphabetic character.
    for i, ch in enumerate(b):
        if ch.isalpha():
            b = b[:i] + ch.upper() + b[i + 1:]
            break
    # Normalize trailing punctuation to exactly one period (leave ? and ! alone).
    b = re.sub(r'[\s;,]+$', '', b.rstrip())
    if not b.endswith(('.', '!', '?')):
        b += '.'
    else:
        b = re.sub(r'\.\.+$', '.', b)
    return b


def render_contact(contact: Dict[str, str]) -> str:
    """Render the name + contact line header block."""
    name = escape_latex(contact.get('name', ''))
    phone = escape_latex(contact.get('phone', ''))
    email = escape_latex(contact.get('email', ''))
    linkedin_raw = contact.get('linkedin', '')
    github_raw = contact.get('github', '')

    def make_url(value: str) -> str:
        if not value:
            return ''
        if value.startswith('http'):
            return value
        return 'https://' + value.lstrip('/')

    linkedin = escape_latex(make_url(linkedin_raw)) if linkedin_raw else ''
    github = escape_latex(make_url(github_raw)) if github_raw else ''

    lines = [f'{{\\Huge \\textbf{{{name}}}}} \\\\', '\\small']
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


def _render_bullets(latex: List[str], bullets: List[object]) -> None:
    """Append a normalized itemize block for a list of bullets."""
    if not bullets:
        return
    latex.append('\\begin{itemize}')
    for bullet in bullets:
        latex.append('  \\resumeItem{%s}' % clean_latex(normalize_bullet(str(bullet))))
    latex.append('\\end{itemize}')


def _render_preamble() -> List[str]:
    return [
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
    ]


def render_latex(
    contact: Dict[str, str],
    summary: str,
    education: Dict[str, str],
    skills: List[object],
    experience: List[Dict[str, object]],
    projects: List[Dict[str, object]],
    achievements: List[str],
) -> str:
    """Assemble the full LaTeX resume document from structured content."""
    def heading(title: str) -> str:
        return f'\\section*{{{title}}}'

    latex = _render_preamble()
    latex.extend([
        '\\begin{document}',
        '\\begin{center}',
        render_contact(contact),
        '\\end{center}',
        '\\vspace{-2pt}',
    ])

    if summary:
        latex.extend([heading('Professional Summary'), clean_latex(summary)])

    if education.get('degree') or education.get('institution'):
        degree = repair_text(education.get('degree', ''))
        gpa = repair_text(education.get('gpa', ''))
        if gpa:
            degree = f'{degree} (GPA: {gpa})' if degree else f'GPA: {gpa}'
        latex.extend([
            heading('Education'),
            '\\resumeSubheading{%s}{%s}{%s}{%s}' % (
                clean_latex(education.get('institution', '') or degree),
                clean_latex(education.get('dates', '')),
                escape_latex(degree),
                clean_latex(education.get('location', '')),
            ),
        ])

    if skills:
        latex.extend([
            heading('Technical Skills'),
            '\\begin{itemize}[leftmargin=1.4em,label={}]',
        ])
        for skill in skills:
            if isinstance(skill, dict):
                category = skill.get('category', '')
                items = skill.get('items', '')
            else:
                text = str(skill)
                category, items = text.split(':', 1) if ':' in text else ('', text)
            category = category.strip()
            items = items.strip()
            if not items:
                continue
            if category:
                latex.append('  \\item \\small{\\textbf{%s:} %s}' % (
                    clean_latex(category), clean_latex(items)))
            else:
                latex.append('  \\item \\small{%s}' % clean_latex(items))
        latex.append('\\end{itemize}')

    if experience:
        latex.append(heading('Experience'))
        for entry in experience:
            latex.append('\\resumeSubheading{%s}{%s}{%s}{%s}' % (
                clean_latex(str(entry.get('role', ''))),
                clean_latex(str(entry.get('dates', ''))),
                clean_latex(str(entry.get('company', ''))),
                clean_latex(str(entry.get('location', ''))),
            ))
            _render_bullets(latex, entry.get('bullets', []) or [])

    if projects:
        latex.append(heading('Projects'))
        for project in projects:
            title = clean_latex(str(project.get('title', '')))
            tech = repair_text(str(project.get('tech', '')))
            year = clean_latex(str(project.get('year', '')))
            heading_line = '\\textbf{%s}' % title
            if tech:
                heading_line += ' $|$ \\textit{\\small %s}' % escape_latex(tech)
            if year:
                heading_line += ' \\hfill \\small %s' % year
            latex.append(heading_line + ' \\\\')
            _render_bullets(latex, project.get('bullets', []) or [])

    if achievements:
        latex.append(heading('Achievements \\& Certifications'))
        _render_bullets(latex, achievements)

    latex.append('\\end{document}')
    return '\n'.join(latex)
