"""Resume tailoring orchestration.

`ResumeGenerator` ties together the parsing, rendering and LLM modules into the
two public entry points used by the CLI and Streamlit UI:

* :meth:`analyze_for_hitl` — inspect the JD vs. the resume for the human-in-the-
  loop review step (missing skills, matched skills, available projects).
* :meth:`tailor_resume` — produce the final tailored LaTeX resume, honouring any
  user-approved skills and project selections.
"""
import re
from typing import Dict, List, Optional, Tuple

from app.core.resume_filter import ResumeFilter
from app.generator.llm_tailor import create_tailor, LLMResumeTailor
from app.generator import text_utils, resume_parser, latex_builder


class ResumeGenerator:
    """Deterministic ATS resume generator that uses only the candidate's own data."""

    # ------------------------------------------------------------------ #
    # Human-in-the-loop helpers
    # ------------------------------------------------------------------ #
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
            return resume_parser.extract_projects(sections, filterer)

        wanted = {self._norm_title(t) for t in selected_projects}
        chosen = []
        for project in resume_parser.parse_all_project_blocks(sections):
            title, _tech, _year = resume_parser.parse_project_header(project['header'])
            norm = self._norm_title(title or project['header'])
            if any(norm and (norm in w or w in norm) for w in wanted):
                chosen.append(resume_parser.finalise_project(project))
        # If nothing matched (e.g. titles differ), fall back to scored selection.
        return chosen[:2] if chosen else resume_parser.extract_projects(sections, filterer)

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
                for raw in resume_parser.parse_all_project_blocks(sections):
                    title = resume_parser.parse_project_header(raw['header'])[0] or raw['header']
                    if matches(title, want):
                        ordered.append(resume_parser.finalise_project(raw))
                        break

        return (ordered or projects)[:2]

    # ------------------------------------------------------------------ #
    # ATS reporting
    # ------------------------------------------------------------------ #
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
            print(f"[ATS] Keywords not surfaced (verify if truthful): {', '.join(missing[:12])}")

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
        quantified = sum(1 for b in bullets if re.search(r'\d', b))
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

    # ------------------------------------------------------------------ #
    # Guaranteed-output helpers
    # ------------------------------------------------------------------ #
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

    @staticmethod
    def _build_minimal_content(content: Dict[str, object], master_text: str) -> Dict[str, object]:
        """Build a minimal but non-empty resume body from the raw master text.

        Used only as a last resort when structured extraction fails, so a new
        user still receives a usable file instead of a blank resume.
        """
        base = dict(content or {})

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
            base['experience'] = [{
                'company': '',
                'role': 'Highlights',
                'dates': '',
                'location': '',
                'bullets': highlight_bullets[:5],
            }] if highlight_bullets else []

        if not base.get('projects'):
            base['projects'] = []
        if not base.get('achievements'):
            base['achievements'] = []

        return base

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def analyze_for_hitl(self, jd_text: str, master_resume_text: str) -> Dict[str, object]:
        """Produce data for a human-in-the-loop review step BEFORE generating.

        Returns a dict with ``missing_skills`` (JD keywords absent from the
        resume), ``matched_skills`` (JD keywords already present) and
        ``projects`` (``[{'title', 'tech'}]`` for every project in the resume).
        """
        normalized_text = text_utils.normalize_text(master_resume_text)
        sections = text_utils.split_sections(normalized_text)

        jd_keywords = LLMResumeTailor.extract_jd_keywords(jd_text)
        resume_lower = normalized_text.lower()
        matched_skills = [kw for kw in jd_keywords if kw.lower() in resume_lower]
        missing_skills = [kw for kw in jd_keywords if kw.lower() not in resume_lower]

        projects = []
        for project in resume_parser.parse_all_project_blocks(sections):
            title, tech, _year = resume_parser.parse_project_header(project['header'])
            bullets = list(project['bullets'])
            if not tech and bullets and resume_parser.looks_like_tech_list(bullets[0]):
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
        normalized_text = text_utils.normalize_text(master_resume_text)
        sections = text_utils.split_sections(normalized_text)
        filterer = ResumeFilter(normalized_text, jd_text)

        extra_skills = [s.strip() for s in (extra_skills or []) if s and s.strip()]
        selected_projects = [p.strip() for p in (selected_projects or []) if p and p.strip()]

        # Contact info is factual and safest to extract deterministically.
        contact = resume_parser.extract_contact(normalized_text)

        content = self._tailor_content(
            jd_text, normalized_text, sections, filterer,
            contact, use_llm_review, extra_skills, selected_projects,
        )

        try:
            self._report_match(jd_text, content)
        except Exception:
            pass

        return latex_builder.render_latex(
            contact,
            str(content.get('summary', '')),
            content.get('education', {}) or {},
            content.get('skills', []) or [],
            content.get('experience', []) or [],
            content.get('projects', []) or [],
            content.get('achievements', []) or [],
        )

    def _tailor_content(
        self,
        jd_text: str,
        normalized_text: str,
        sections: List[Tuple[str, str]],
        filterer: ResumeFilter,
        contact: Dict[str, str],
        use_llm_review: bool,
        extra_skills: List[str],
        selected_projects: List[str],
    ) -> Dict[str, object]:
        """Build the structured resume content, guaranteeing a non-empty result."""
        content: Optional[Dict[str, object]] = None

        # Preferred path: LLM-driven, JD-aware tailoring (grounded in the resume).
        if use_llm_review:
            content = self._llm_tailor(jd_text, normalized_text, extra_skills, selected_projects)

        # Fallback path: deterministic keyword-based extraction. Runs when the
        # LLM is unavailable or returned essentially empty content.
        if content is None or self._content_is_empty(content):
            deterministic = {
                'summary': resume_parser.extract_best_summary(sections, filterer),
                'education': resume_parser.extract_education(sections),
                'skills': resume_parser.extract_skills(sections, filterer),
                'experience': resume_parser.extract_experience(sections, filterer),
                'projects': self._select_projects_deterministic(sections, filterer, selected_projects),
                'achievements': resume_parser.extract_achievements(sections, filterer, contact.get('name', '')),
            }
            if content is not None and self._content_is_empty(content):
                print("[WARN] AI tailoring returned empty content; "
                      "using keyword-based extraction instead.")
            content = deterministic

        # Last-resort safety net for unstructured resumes the parser can't segment.
        if self._content_is_empty(content):
            print("[WARN] Could not extract structured sections; "
                  "generating a minimal resume from the raw resume text.")
            content = self._build_minimal_content(content, normalized_text)

        # Honour human-in-the-loop choices regardless of which path produced the content.
        if extra_skills:
            content['skills'] = self._merge_extra_skills(content.get('skills', []) or [], extra_skills)
        if selected_projects:
            content['projects'] = self._enforce_selected_projects(
                content.get('projects', []) or [], selected_projects, sections
            )

        return content

    @staticmethod
    def _llm_tailor(
        jd_text: str,
        normalized_text: str,
        extra_skills: List[str],
        selected_projects: List[str],
    ) -> Optional[Dict[str, object]]:
        """Run the LLM tailoring path; returns None on any failure."""
        tailor = create_tailor()
        if tailor is None:
            print("[WARN] GROQ_API_KEY not set - skipping AI tailoring. "
                  "Output will use keyword-based selection only and may be less JD-aligned. "
                  "Add your key to a .env file (see .env.example) for best results.")
            return None
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
            return content
        except Exception as exc:
            print(f"[WARN] AI tailoring failed ({exc}); "
                  "falling back to keyword-based extraction.")
            return None
