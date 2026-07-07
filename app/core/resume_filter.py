"""
Module: resume_filter
Purpose: Intelligently filters and condenses resume text based on a Job Description
to fit within LLM token limits while preserving high-signal content for ATS optimization.
"""
import re
from typing import List, Tuple


class ResumeFilter:
    """
    Filters raw resume text (often extracted from multiple PDFs) by scoring
    each line/paragraph for relevance to the Job Description, removing
    duplicates and noise, and returning a condensed summary that fits
    within a token budget.
    """

    # Common English stopwords (lowercased)
    STOPWORDS = {
        'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
        'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'need',
        'dare', 'ought', 'used', 'there', 'here', 'where', 'when', 'what',
        'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other',
        'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
        'than', 'too', 'very', 'just', 'now', 'also', 'etc'
    }

    # Boilerplate / noise phrases that add no signal
    FILLER = [
        r'responsible for', r'duties include', r'involved in',
        r'assisted with', r'helped with', r'participated in',
        r'contributed to', r'proficient in', r'experienced with',
        r'knowledge of', r'familiar with', r'skilled in',
        r'expertise in', r'competent in', r'adept at',
        r'strong background in', r'hands-on experience',
        r'proven ability', r'track record', r'demonstrated',
        r'extensive', r'solid', r'references available',
        r'\bphone\b', r'\bcell\b', r'\bfax\b',
    ]

    # Section priority for ATS (what to keep first)
    SECTION_PRIORITY = {
        'experience': 5,      # Most important for ATS
        'skills': 4,          # Second priority
        'projects': 3,        # Third if relevant
        'summary': 2,         # Less critical
        'education': 1,       # Only recent grads
        'contact': 0,         # Basic info
        'uncategorized': -1
    }

    # Metric indicators - boost chunks containing these
    METRIC_INDICATORS = {
        'improved', 'increased', 'decreased', 'reduced',
        'boosted', 'enhanced', 'optimized', 'delivered',
        'built', 'created', 'developed', 'implemented',
        'managed', 'led', 'solved', 'resolved', 'achieved',
        'saved', 'generated', 'accelerated', 'scaled',
        'launched', 'designed', 'architected', 'automated',
        '%', 'percent', 'percentage', 'x', 'times', 'million'
    }

    def __init__(self, raw_text: str, jd_text: str):
        self.raw_text = raw_text
        self.jd_text = jd_text
        self._jd_keywords = None

    # ── internal helpers ────────────────────────────────────────────────────

    def _get_jd_keywords(self) -> set:
        if self._jd_keywords is None:
            text = self.jd_text.lower()
            text = re.sub(r'[^a-z0-9\s]', ' ', text)
            words = set(text.split())
            words = {w for w in words if w not in self.STOPWORDS and len(w) > 2}
            self._jd_keywords = words
        return self._jd_keywords

    def _score(self, text: str) -> int:
        """Score text by JD keyword overlap"""
        t = text.lower()
        t = re.sub(r'[^a-z0-9\s]', ' ', t)
        tokens = set(t.split())
        return len(tokens & self._get_jd_keywords())

    def _has_metrics(self, text: str) -> bool:
        """Detect if text contains metrics/achievements"""
        text_lower = text.lower()
        return any(indicator in text_lower for indicator in self.METRIC_INDICATORS)

    def _clean(self, text: str) -> str:
        """Clean text - strip single chars, collapse whitespace"""
        text = re.sub(r'\b[a-zA-Z]\b', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _split_into_chunks(self, text: str) -> List[str]:
        """Split resume text into chunks (paragraphs / bullet points)"""
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        # Split on blank lines to get paragraphs/bullets
        chunks = re.split(r'\n\s*\n', text)
        chunks = [c.strip() for c in chunks if c.strip()]
        return chunks

    # ── public API ────────────────────────────────────────────────────────

    def filter(self, max_words: int = 400) -> str:
        """
        1. Split raw resume into chunks.
        2. Remove exact-duplicate chunks (common when multiple PDFs overlap).
        3. Score each remaining chunk by JD keyword overlap + section priority + metrics.
        4. Keep top-scoring chunks until *max_words* is reached.
        """
        chunks = self._split_into_chunks(self.raw_text)

        # Step 1 – deduplicate exact chunks
        seen = set()
        unique_chunks = []
        for c in chunks:
            norm = re.sub(r'\s+', ' ', c.strip().lower())
            if norm and norm not in seen:
                seen.add(norm)
                unique_chunks.append(c)

        # Step 2 – score with JD relevance, section priority, and metrics
        scored: List[Tuple[float, str, str, bool]] = []
        for c in unique_chunks:
            chunk_score = self._score(c)

            # Boost if contains metrics
            has_metrics = self._has_metrics(c)
            if has_metrics:
                chunk_score += 2  # Boost metric-containing chunks

            # Add section priority
            header_match = re.match(r'^([A-Za-z0-9]+)', c[:50], re.IGNORECASE)
            header = header_match.group(1).lower() if header_match else 'uncategorized'
            section_score = self.SECTION_PRIORITY.get(header, -1)

            total_score = chunk_score * 1.5 + section_score * 0.5
            scored.append((total_score, c, header, has_metrics))

        # Step 3 – sort by combined score
        scored.sort(key=lambda x: x[0], reverse=True)

        # Step 4 – greedily add chunks until word budget is consumed
        picked = []
        word_count = 0
        for _, chunk, section, _ in scored:
            clean_chunk = self._clean(chunk)
            if not clean_chunk:
                continue
            wc = len(clean_chunk.split())
            if word_count + wc > max_words:
                break
            picked.append(clean_chunk)
            word_count += wc

        # Step 5 – final fallback to ensure single page
        result = '\n\n'.join(picked)
        if len(result.split()) > 400:
            result = ' '.join(result.split()[:400])

        return result

    def quick_filter(self, max_chars: int = 8000) -> str:
        """Fallback: keep top-scoring chunks until char budget reached."""
        chunks = self._split_into_chunks(self.raw_text)
        # dedup
        seen = set()
        unique = []
        for c in chunks:
            n = re.sub(r'\s+', ' ', c.strip().lower())
            if n not in seen:
                seen.add(n)
                unique.append(c)
        # score
        scored = [(self._score(c), c) for c in unique]
        scored.sort(key=lambda x: x[0], reverse=True)

        result = []
        char_count = 0
        for _, chunk in scored:
            c = self._clean(chunk)
            if char_count + len(c) > max_chars:
                break
            result.append(c)
            char_count += len(c) + 1
        return '\n\n'.join(result) if result else self._clean(self.raw_text)[:max_chars]