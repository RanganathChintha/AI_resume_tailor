"""
LLM-based LaTeX resume reviewer and refiner.
Takes generated LaTeX, reviews it for issues, and returns improved LaTeX.
"""
import os
import json
import re
from typing import Optional
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from app.core.config import settings


class LLMResumeReviewer:
    """Reviews and refines LaTeX resumes using an LLM."""

    SYSTEM_PROMPT = """You are an expert LaTeX resume reviewer. Your task is to review a generated LaTeX resume and fix any issues.

You will receive:
1. The original job description (for context)
2. The generated LaTeX resume

Your goals:
- Fix spelling errors
- Fix spacing issues (missing spaces between words, missing spaces after punctuation)
- Fix LaTeX syntax errors (unescaped special characters: & % $ # _ { } ~ ^ \\)
- Fix punctuation/capitalization issues in bullet points
- Ensure consistent formatting across sections
- Fix any broken LaTeX commands or malformed macros
- Remove duplicate or redundant content
- Ensure the document will compile cleanly with pdflatex
- Replace Unicode replacement characters (�, ●, etc.) with proper bullet points (•) or remove them
- Ensure tech stack formatting is consistent (e.g., "Postgre | SQL" → "PostgreSQL", "js" → "JavaScripting aScript" or "JavaScript")

CRITICAL RULES:
- DO NOT add, invent, or hallucinate any content (no new skills, experiences, projects, achievements)
- DO NOT change the actual resume content (role titles, company names, dates, technologies, bullet substance)
- DO NOT restructure sections or change the LaTeX document class/packages
- ONLY fix mechanical errors: spelling, spacing, LaTeX escaping, punctuation, capitalization
- Preserve all \resumeItem, \resumeSubheading, \resumeProject commands exactly as-is (don't change their arguments)
- Return ONLY the corrected LaTeX, nothing else (no markdown, no explanation)

Common LaTeX special characters that MUST be escaped in text:
- & → \\&
- % → \\%
- $ → \\$
- # → \\#
- _ → \\_
- { → \\{
- } → \\}
- ~ → \\textasciitilde{}
- ^ → \\textasciicircum{}
- \\ → \\textbackslash{}

Also look for:
- Missing spaces after periods/commas in bullet points
- Run-together words (e.g., "usingLangChain" → "using LangChain")
- CamelCase artifacts from PDF extraction (e.g., "70/30FAISSsemantic" → "70/30 FAISS semantic")
- Unescaped percent signs in text (e.g., "80%" → "80\\%")
- Missing spaces around en-dashes in dates
- Unicode replacement characters (�, ●) in bullet lists → replace with proper formatting
- Inconsistent tech stack names (e.g., "Postgre | SQL" → "PostgreSQL", "js" → "JavaScript")"""

    def __init__(self, model_name: str = "openai/gpt-oss-120b", temperature: float = 0.1):
        """Initialize the LLM reviewer with Groq."""
        api_key = settings.GROQ_API_KEY
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment. Please set it in .env")

        self.llm = ChatGroq(
            model=model_name,
            temperature=temperature,
            api_key=api_key,
            max_retries=2,
            max_tokens=4000,
        )

    def _build_prompt(self, jd_text: str, latex_code: str) -> list:
        """Build the message list for the LLM."""
        return [
            SystemMessage(content=self.SYSTEM_PROMPT),
            HumanMessage(content=f"""Job Description:
{jd_text}

Generated LaTeX Resume:
{latex_code}

Review and return ONLY the corrected LaTeX:"""),
        ]

    def review(self, jd_text: str, latex_code: str) -> str:
        """
        Review the LaTeX resume and return corrected version.

        Args:
            jd_text: The job description for context
            latex_code: The generated LaTeX to review

        Returns:
            Corrected LaTeX string
        """
        messages = self._build_prompt(jd_text, latex_code)
        response = self.llm.invoke(messages)
        corrected = response.content.strip()

        # Strip markdown code fences if present
        if corrected.startswith("```latex"):
            corrected = corrected[8:]
        if corrected.startswith("```"):
            corrected = corrected[3:]
        if corrected.endswith("```"):
            corrected = corrected[:-3]

        corrected = corrected.strip()

        # Check for truncation (missing end of document)
        if not corrected.endswith('\\end{document}'):
            # Try to find the last complete section and append end document
            if '\\end{document}' in corrected:
                corrected = corrected[:corrected.rindex('\\end{document}')] + '\\end{document}'
            else:
                # Fallback: return original if severely truncated
                return latex_code

        return corrected


def create_reviewer(model_name: str = "openai/gpt-oss-120b") -> Optional[LLMResumeReviewer]:
    """Factory function that returns None if API key is missing."""
    try:
        return LLMResumeReviewer(model_name=model_name)
    except ValueError:
        return None