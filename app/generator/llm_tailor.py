"""
LLM-based resume tailorer.

Takes the candidate's master resume text plus a target job description and
produces tailored, JD-aligned resume content as strict JSON. The LLM is used
to SELECT and REPHRASE the candidate's real content so it emphasises the job
description keywords, while being strictly forbidden from inventing new facts.
"""
import json
import re
from typing import Optional, Dict, Any, List

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from app.core.config import settings


# Placeholder "company" names the LLM sometimes invents to disguise a personal
# project as real work experience. Any experience entry using one of these as
# its company is dropped during normalisation.
_FABRICATED_COMPANY_TERMS = {
    'self-initiated', 'self initiated', 'self', 'personal', 'personal project',
    'academic', 'academic project', 'university project', 'college project',
    'freelance', 'independent', 'side project', 'hobby project', 'n/a', 'none',
    'various', 'self-employed', 'self employed',
}


class LLMResumeTailor:
    """Generates tailored, grounded resume content from a master resume + JD."""

    SYSTEM_PROMPT = r"""You are an expert technical resume writer and ATS (Applicant Tracking System) optimisation specialist. Your resumes routinely score 90+ on Jobscan-style ATS keyword scans.

You will receive:
1. A target JOB DESCRIPTION (JD).
2. A PRIORITY KEYWORD LIST automatically extracted from the JD.
3. The candidate's MASTER RESUME text (extracted from PDFs, may contain spacing/formatting artifacts).

Your job is to produce a SINGLE tailored, one-page resume as STRICT JSON that scores 90+ on ATS keyword matching for this JD.

HARD GROUNDING RULES (must follow exactly):
- Use ONLY facts that appear in the MASTER RESUME. Never invent employers, dates, metrics, degrees, projects, or achievements.
- You MAY rephrase, reorder, condense, and re-emphasise existing content to align with the JD.
- You MAY fix obvious extraction artifacts: run-together words ("usingLangChain" -> "using LangChain"), missing spaces after commas/periods, camelCase splits ("70/30FAISSsemantic" -> "70/30 FAISS semantic"), truncated words ("Cross" -> "Cross-Site Scripting" ONLY if clearly implied), and stray unicode.
- If a hard fact (employer, metric, date) is not in the master resume, omit it. Do NOT fabricate numbers.

EXPERIENCE SECTION - STRICT (do not fabricate work history):
- The "experience" array must contain ONLY real jobs/internships that explicitly appear under the master resume's Experience/Work section, with their REAL employer names, titles, dates, and locations exactly as written there.
- NEVER convert a PROJECT into an experience entry. NEVER invent an employer. NEVER use placeholder companies like "Self-initiated", "Personal", "Academic Project", "Freelance", or "Remote" as a company name unless that EXACT word is the real employer in the master resume.
- If the master resume lists a real internship (e.g. an internship at a named company), that entry MUST appear in "experience" - do not drop it in favour of projects.
- If the master resume genuinely has NO work experience at all, return an EMPTY "experience" array (the projects section will carry the weight). Do NOT manufacture experience to fill space.
- Keep each experience's employer, dates, and location byte-for-byte truthful; you may only rephrase the bullet wording.

JD-RELEVANCE SELECTION (choose what best matches THIS job - this is critical):
- The master resume may contain MORE projects, experiences, and skills than fit on one page. Your task is to SELECT the most JD-relevant subset.
- PROJECTS: Read every project in the master resume, then score each project from 0-10 by how well its tech stack AND problem domain match the JD. A project counts as RELEVANT if it shares ANY of the JD's core technologies, languages, or problem domains (e.g. a Python/Flask project IS relevant to a Python backend role). Include the 2 most relevant projects. Keep 2 projects whenever two are even loosely relevant - projects add valuable keyword surface for ATS. Only drop a project if it is CLEARLY unrelated to the JD (e.g. a pure cybersecurity/OWASP scanner for a front-end React role, or a game for a data-engineering role). If only one project is even loosely relevant, output that one; output an empty list ONLY if the candidate has no projects at all or none share anything with the JD.
- EXPERIENCE: Prioritise the experience(s) and the specific bullets that best demonstrate the JD's core requirements; lead with the most relevant bullet.
- SKILLS: Lead with the skill categories and individual skills the JD asks for. Do NOT output duplicate or near-duplicate categories (e.g. never have both "Languages" and "Programming Languages"). Merge overlapping categories into one clean set of 5-6 non-overlapping categories.
- ACHIEVEMENTS: Include ONLY achievements/certifications that LITERALLY appear in the master resume. Do NOT invent, embellish, or attach new numbers to them. Prefer the achievements most relevant to the JD. If fewer than 2 genuine achievements exist in the master resume, output fewer (or an empty list) rather than fabricating any. Never list soft-skill labels (e.g. "Team Collaboration", "Technical Research") as achievements.

ATS KEYWORD OPTIMISATION (this drives the score - be aggressive but truthful):
- Incorporate EVERY priority keyword that the candidate can TRUTHFULLY claim (i.e. it appears in, or is a clear synonym of something in, the master resume). Use the JD's EXACT wording/spelling for these keywords (e.g. if JD says "CI/CD", write "CI/CD" not "continuous integration").
- Mirror JD hard skills VERBATIM in the Technical Skills section. This section is the single biggest ATS scoring lever - make it comprehensive with every relevant tool, language, framework, and concept the candidate legitimately knows.
- Naturally weave the top priority keywords into the summary and experience/project bullets as well (keyword reinforcement across sections boosts ATS scores), but keep sentences readable - no keyword stuffing gibberish.
- Include the target JOB TITLE from the JD near the top: set the summary's first phrase to align the candidate with that role (only if truthful to their background).
- Use standard, ATS-recognised section names exactly as in the schema. Use consistent date format "MMM YYYY -- MMM YYYY" (or "-- Present").
- Do NOT invent skills the candidate lacks. If a priority keyword is clearly unrelated to the candidate's background, skip it.

STRONG ACTION VERBS, BUZZWORDS & METRICS (raise perceived impact and ATS score):
- Start EVERY experience and project bullet with a powerful, PAST-TENSE action verb: Architected, Engineered, Spearheaded, Optimized, Automated, Streamlined, Accelerated, Scaled, Designed, Developed, Built, Implemented, Delivered, Reduced, Increased, Led, Orchestrated, Deployed, Refactored, Migrated, Integrated, Launched, Boosted, Cut, Drove, Enhanced, Pioneered.
- NEVER start two bullets with the SAME action verb anywhere in the resume. Every bullet must open with a DIFFERENT verb. Never use present tense like "Write" - use "Wrote".

QUANTIFY IMPACT (this is the #1 recruiter/ATS complaint - fix it aggressively):
- EVERY experience bullet MUST contain at least one concrete number: a percentage (%), a count (e.g. "12 microservices", "3 teams"), a time/latency (e.g. "reduced from 4.5s to 0.7s"), a scale (e.g. "50K users", "500GB/day"), or money ($). Aim for a number in EVERY project bullet too.
- Prefer metrics that already appear in the master resume. If a bullet describes real work but the master resume gives no number, you MAY add a conservative, realistic, clearly-plausible metric (e.g. "15+ REST APIs", "reduced load time by ~30%", "across 3 environments"). Use "~", "+", or "up to" to signal an estimate when unsure. Prefer this over leaving a bullet with no number.
- NEVER add invented numbers to ACHIEVEMENTS, and never invent employers, dates, awards, or headline achievements.

AVOID REPETITION (ATS penalises repeated words):
- Do NOT overuse the same noun/keyword across bullets. If a keyword (e.g. "REST API", "Python", "RAG") already appears, use synonyms or restructure the next sentence (e.g. "RESTful endpoints", "HTTP services", "backend services"). Each important term should appear at most 2-3 times across the whole resume.
- Vary sentence structure - do not start every bullet with the same word or use the same connective phrasing repeatedly.

BULLET CONSISTENCY (formatting is scored):
- Every bullet MUST start with a capital letter and END WITH A PERIOD. Be consistent across ALL bullets.
- Do not end a bullet with a semicolon, comma, or no punctuation. One sentence per bullet.
- Use industry-standard buzzwords recruiters and ATS look for: "production-grade", "scalable", "end-to-end", "cross-functional", "microservices", "CI/CD", "responsive", "high-performance", "fault-tolerant", "cloud-native" - but ONLY where they truthfully fit the candidate's work, and do not repeat the same buzzword more than twice.

ONE STRICT PAGE (must fit on a single page - do NOT overflow):
- Output: one 3-line professional summary (50-75 words), 5-6 skill categories, at most 2 experiences (3 bullets each), 1-2 JD-RELEVANT projects (3 bullets each - only include a project if it truly matches the JD; never pad to 2 with an unrelated project), and 2-3 achievements. Never output more than 2 projects, and never output a project unrelated to the JD.
- Each bullet is a single sentence of roughly 16-26 words. Keep it tight so the whole resume stays on ONE page.

OUTPUT FORMAT:
Return ONLY valid JSON (no markdown fences, no commentary) with EXACTLY this schema:
{
  "summary": "string",
  "education": {"degree": "", "institution": "", "dates": "", "location": "", "gpa": ""},
  "skills": [{"category": "Languages", "items": "Comma, separated, skills"}],
  "experience": [{"role": "", "company": "", "dates": "", "location": "", "bullets": ["", ""]}],
  "projects": [{"title": "", "tech": "", "year": "", "bullets": ["", ""]}],
  "achievements": ["", ""]
}
Provide 1-2 JD-RELEVANT items in "projects" (never pad with an unrelated project). Every string must be plain text (NOT LaTeX). Do not escape characters. Do not include a "contact" field."""

    def __init__(self, model_name: Optional[str] = None, temperature: float = 0.2):
        api_key = settings.GROQ_API_KEY
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment. Please set it in .env")

        self.llm = ChatGroq(
            model=model_name or settings.MODEL_NAME,
            temperature=temperature,
            api_key=api_key,
            max_retries=2,
            max_tokens=3000,
            reasoning_effort="low",
        )

    # Groq free-tier models cap total tokens-per-minute; keep the master resume
    # input bounded so (input + output) stays under the limit.
    MAX_MASTER_CHARS = 6000

    # Curated tech vocabulary used to recognise hard skills in a JD even when
    # they are lowercase or embedded in prose.
    _TECH_VOCAB = {
        'python', 'java', 'javascript', 'typescript', 'c', 'c++', 'c#', 'go',
        'rust', 'sql', 'html', 'css', 'react', 'reactjs', 'angular', 'vue',
        'node', 'nodejs', 'express', 'expressjs', 'flask', 'django', 'fastapi',
        'spring', 'mongodb', 'postgresql', 'mysql', 'redis', 'sqlite', 'oracle',
        'docker', 'kubernetes', 'aws', 'azure', 'gcp', 'git', 'github', 'gitlab',
        'jenkins', 'ci/cd', 'rest', 'graphql', 'grpc', 'kafka', 'rabbitmq',
        'langchain', 'langgraph', 'llm', 'llms', 'rag', 'faiss', 'pinecone',
        'embeddings', 'nlp', 'pytorch', 'tensorflow', 'keras', 'scikit-learn',
        'pandas', 'numpy', 'matplotlib', 'jwt', 'oauth', 'agile', 'scrum',
        'microservices', 'linux', 'bash', 'jira', 'postman', 'selenium',
        'pytest', 'terraform', 'ansible', 'nginx', 'websocket', 'sse', 'mern',
        'vector', 'prompt', 'transformers', 'huggingface', 'openai', 'devops',
    }

    _KEYWORD_STOPWORDS = {
        'the', 'and', 'for', 'with', 'you', 'our', 'are', 'will', 'have', 'this',
        'that', 'your', 'who', 'about', 'from', 'their', 'they', 'them', 'was',
        'were', 'been', 'being', 'has', 'had', 'not', 'but', 'can', 'all', 'any',
        'job', 'role', 'work', 'team', 'teams', 'looking', 'candidate', 'ability',
        'experience', 'strong', 'good', 'great', 'plus', 'preferred', 'required',
        'responsibilities', 'requirements', 'skills', 'years', 'year', 'company',
        'including', 'etc', 'able', 'must', 'should', 'help', 'join', 'build',
        'using', 'across', 'within', 'into', 'other', 'more', 'well', 'new',
        'nice', 'hiring', 'hire', 'develop', 'developing', 'we', 'as', 'an',
        'is', 'to', 'of', 'in', 'on', 'or', 'a', 'requirement', 'responsibility',
        'include', 'includes', 'ideal', 'bonus', 'familiarity', 'knowledge',
        'understanding', 'proficiency', 'proficient', 'hands', 'end',
    }

    @classmethod
    def extract_jd_keywords(cls, jd_text: str, limit: int = 30) -> List[str]:
        """Extract likely ATS keywords (hard skills, tools, capitalised phrases)."""
        found: List[str] = []
        seen: set = set()

        def add(term: str) -> None:
            key = term.lower().strip()
            if key and key not in seen and len(key) > 1:
                seen.add(key)
                found.append(term.strip())

        # 1. Known tech tokens (preserve JD's own casing where possible)
        tokens = re.findall(r'[A-Za-z][A-Za-z0-9+#./-]*[A-Za-z0-9+#]|[A-Za-z]', jd_text)
        for tok in tokens:
            if tok.lower() in cls._TECH_VOCAB:
                add(tok)

        # 2. Acronyms / capitalised tech (e.g. "REST", "API", "CI/CD", "SQL")
        for m in re.findall(r'\b[A-Z][A-Za-z0-9]*(?:/[A-Z][A-Za-z0-9]*)?\b', jd_text):
            if 2 <= len(m) <= 12 and m.lower() not in cls._KEYWORD_STOPWORDS:
                add(m)

        # 3. Multi-word capitalised phrases (e.g. "Machine Learning", "REST APIs")
        for m in re.findall(r'\b(?:[A-Z][a-zA-Z]+)(?:\s+[A-Z][a-zA-Z]+){1,2}\b', jd_text):
            add(m)

        # 4. Remaining salient lowercase words that look like skills
        for w in re.findall(r'\b[a-z][a-z+#.-]{3,}\b', jd_text.lower()):
            if w in cls._TECH_VOCAB and w not in seen:
                add(w)

        return found[:limit]

    def _build_prompt(
        self,
        jd_text: str,
        master_resume_text: str,
        extra_skills: Optional[List[str]] = None,
        selected_projects: Optional[List[str]] = None,
    ) -> list:
        keywords = self.extract_jd_keywords(jd_text)
        keyword_block = ', '.join(keywords) if keywords else '(none detected)'
        master = master_resume_text.strip()
        if len(master) > self.MAX_MASTER_CHARS:
            master = master[:self.MAX_MASTER_CHARS].rsplit('\n', 1)[0]
        jd = jd_text.strip()
        if len(jd) > 2500:
            jd = jd[:2500]

        # Human-in-the-loop directives from the user's review step.
        hitl_parts = []
        extra_skills = [s.strip() for s in (extra_skills or []) if s and s.strip()]
        selected_projects = [p.strip() for p in (selected_projects or []) if p and p.strip()]
        if extra_skills:
            hitl_parts.append(
                "USER-CONFIRMED SKILLS (the candidate confirmed they genuinely have these JD skills - "
                "you MUST include them verbatim in the Technical Skills section, and weave them into "
                "summary/experience/project bullets where truthful): " + ', '.join(extra_skills)
            )
        if selected_projects:
            hitl_parts.append(
                "USER-SELECTED PROJECTS (include ONLY these projects, matched by title, and no others - "
                "this overrides your own relevance judgement): " + '; '.join(selected_projects)
            )
        hitl_block = ('\n\n' + '\n\n'.join(hitl_parts)) if hitl_parts else ''

        return [
            SystemMessage(content=self.SYSTEM_PROMPT),
            HumanMessage(content=f"""JOB DESCRIPTION:
{jd}

PRIORITY KEYWORD LIST (incorporate every one the candidate can truthfully claim, using this exact spelling):
{keyword_block}

MASTER RESUME:
{master}{hitl_block}

Return ONLY the tailored resume JSON:"""),
        ]

    @staticmethod
    def _extract_json(raw: str) -> Optional[Dict[str, Any]]:
        """Robustly pull a JSON object out of the model response."""
        text = raw.strip()

        # Strip markdown fences if present
        if text.startswith("```"):
            text = re.sub(r'^```[a-zA-Z]*\n?', '', text)
            text = re.sub(r'\n?```$', '', text).strip()

        # Fast path
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Fallback: grab the outermost {...} block
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1 and end > start:
            candidate = text[start:end + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass

        # Last resort: attempt to repair a truncated JSON object by trimming to
        # the last complete top-level element and closing open brackets.
        if start != -1:
            repaired = LLMResumeTailor._repair_truncated_json(text[start:])
            if repaired is not None:
                return repaired
        return None

    @staticmethod
    def _repair_truncated_json(fragment: str) -> Optional[Dict[str, Any]]:
        """Best-effort recovery of a JSON object that was cut off mid-stream."""
        # Trim trailing incomplete content after the last closed quote/brace.
        for cut in range(len(fragment), 0, -1):
            chunk = fragment[:cut].rstrip().rstrip(',')
            if not chunk.endswith(('}', ']', '"')):
                continue
            # Balance braces/brackets that are still open.
            opens = chunk.count('{') - chunk.count('}')
            open_sq = chunk.count('[') - chunk.count(']')
            if opens < 0 or open_sq < 0:
                continue
            balanced = chunk + (']' * open_sq) + ('}' * opens)
            try:
                return json.loads(balanced)
            except json.JSONDecodeError:
                continue
        return None

    @staticmethod
    def _normalise(data: Dict[str, Any]) -> Dict[str, Any]:
        """Coerce the parsed JSON into the expected shapes, tolerating omissions."""
        result: Dict[str, Any] = {}

        result['summary'] = str(data.get('summary', '') or '').strip()

        edu = data.get('education') or {}
        if isinstance(edu, list):
            edu = edu[0] if edu else {}
        result['education'] = {
            'degree': str(edu.get('degree', '') or '').strip(),
            'institution': str(edu.get('institution', '') or '').strip(),
            'dates': str(edu.get('dates', '') or '').strip(),
            'location': str(edu.get('location', '') or '').strip(),
            'gpa': str(edu.get('gpa', '') or '').strip(),
        }

        skills: List[Dict[str, str]] = []
        for item in data.get('skills', []) or []:
            if isinstance(item, dict):
                category = str(item.get('category', '') or '').strip()
                items = item.get('items', '')
                if isinstance(items, list):
                    items = ', '.join(str(i).strip() for i in items if str(i).strip())
                items = str(items).strip()
                if category and items:
                    skills.append({'category': category, 'items': items})
            elif isinstance(item, str) and item.strip():
                skills.append({'category': '', 'items': item.strip()})
        result['skills'] = skills

        def _clean_bullets(raw_bullets: Any) -> List[str]:
            out = []
            for b in raw_bullets or []:
                b = str(b).strip()
                if b:
                    out.append(b)
            return out

        experience = []
        for exp in data.get('experience', []) or []:
            if not isinstance(exp, dict):
                continue
            company = str(exp.get('company', '') or '').strip()
            # Drop entries whose "company" is a placeholder the model invented to
            # dress up a project as work experience.
            if company.lower() in _FABRICATED_COMPANY_TERMS:
                continue
            experience.append({
                'role': str(exp.get('role', '') or '').strip(),
                'company': company,
                'dates': str(exp.get('dates', '') or '').strip(),
                'location': str(exp.get('location', '') or '').strip(),
                'bullets': _clean_bullets(exp.get('bullets'))[:3],
            })
        result['experience'] = experience

        projects = []
        for proj in data.get('projects', []) or []:
            if not isinstance(proj, dict):
                continue
            projects.append({
                'title': str(proj.get('title', '') or '').strip(),
                'tech': str(proj.get('tech', '') or '').strip(),
                'year': str(proj.get('year', '') or '').strip(),
                'bullets': _clean_bullets(proj.get('bullets'))[:3],
            })
        result['projects'] = projects[:2]

        achievements = []
        for ach in data.get('achievements', []) or []:
            ach = str(ach).strip()
            if ach:
                achievements.append(ach)
        result['achievements'] = achievements[:3]

        return result

    def tailor(
        self,
        jd_text: str,
        master_resume_text: str,
        extra_skills: Optional[List[str]] = None,
        selected_projects: Optional[List[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Produce tailored resume content as a normalised dict, or None on failure.
        Raises no exceptions to the caller for content issues; returns None instead.
        """
        messages = self._build_prompt(
            jd_text,
            master_resume_text,
            extra_skills=extra_skills,
            selected_projects=selected_projects,
        )
        response = self.llm.invoke(messages)
        raw = response.content if isinstance(response.content, str) else str(response.content)

        parsed = self._extract_json(raw)
        if parsed is None:
            return None
        return self._normalise(parsed)


def create_tailor(model_name: Optional[str] = None) -> Optional[LLMResumeTailor]:
    """Factory that returns None if the API key is missing."""
    try:
        return LLMResumeTailor(model_name=model_name)
    except ValueError:
        return None
