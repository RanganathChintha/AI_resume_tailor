"""
Streamlit interface for the AI Resume Tailor.

Run with:  streamlit run streamlit_app.py

Features:
- Upload one or more master resume PDFs (saved to the data folder).
- Paste a target job description.
- Generate an ATS-optimised, JD-tailored LaTeX resume (LLM path or offline fallback).
- View ATS metrics (keyword coverage, quantified bullets, repeated verbs).
- Preview the LaTeX source and download the .tex file.
- Optionally compile to PDF (if a LaTeX distribution / pdflatex is installed).
"""
import io
import os
import glob
import shutil
import contextlib

import streamlit as st

from app.core.config import settings
from app.data_manager.cache import ResumeCacheManager
from app.generator.tailor import ResumeGenerator
from main import compile_pdf, _is_valid_latex, _minimal_placeholder_latex


st.set_page_config(page_title="AI Resume Tailor", page_icon="📄", layout="wide")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _apply_api_key(key: str) -> None:
    """Push a user-entered API key into both the environment and settings."""
    key = (key or "").strip()
    if key:
        os.environ["GROQ_API_KEY"] = key
        settings.GROQ_API_KEY = key


def _list_saved_pdfs() -> list[str]:
    return sorted(glob.glob(os.path.join(settings.DATA_DIR, "*.pdf")))


def _save_uploaded_pdfs(uploaded_files) -> list[str]:
    """Persist uploaded PDFs to the data directory so the cache manager sees them."""
    saved = []
    os.makedirs(settings.DATA_DIR, exist_ok=True)
    for uf in uploaded_files:
        dest = os.path.join(settings.DATA_DIR, uf.name)
        with open(dest, "wb") as f:
            f.write(uf.getbuffer())
        saved.append(dest)
    return saved


def _parse_ats_report(log_text: str) -> dict:
    """Extract the [ATS] metric lines from captured stdout into a small dict."""
    report = {"lines": [], "coverage": None, "quantified": None, "repeated_verbs": None}
    for line in log_text.splitlines():
        line = line.strip()
        if not line.startswith("[ATS]"):
            continue
        report["lines"].append(line)
        if "Hard-keyword coverage:" in line:
            report["coverage"] = line.split("coverage:", 1)[1].strip()
        elif "Quantified bullets:" in line:
            report["quantified"] = line.split("Quantified bullets:", 1)[1].strip()
        elif "Repeated leading verbs" in line:
            report["repeated_verbs"] = line.split(":", 1)[1].strip()
    return report


def _analyze(job_description: str) -> dict:
    """Human-in-the-loop step 1: inspect the JD vs. resume before generating."""
    cache_manager = ResumeCacheManager()
    generator = ResumeGenerator()
    master_data = cache_manager.get_resume_data()
    analysis = generator.analyze_for_hitl(job_description, master_data)
    analysis["master_data"] = master_data
    return analysis


def _generate(
    job_description: str,
    use_llm: bool,
    master_data: str | None = None,
    extra_skills: list[str] | None = None,
    selected_projects: list[str] | None = None,
) -> dict:
    """Run the tailoring pipeline, capturing logs and returning results."""
    cache_manager = ResumeCacheManager()
    generator = ResumeGenerator()

    log_buffer = io.StringIO()
    with contextlib.redirect_stdout(log_buffer):
        if master_data is None:
            master_data = cache_manager.get_resume_data()
        latex_code = generator.tailor_resume(
            jd_text=job_description,
            master_resume_text=master_data,
            use_llm_review=use_llm,
            extra_skills=extra_skills,
            selected_projects=selected_projects,
        )
        if not _is_valid_latex(latex_code):
            latex_code = _minimal_placeholder_latex(master_data)

    logs = log_buffer.getvalue()
    return {
        "latex": latex_code,
        "logs": logs,
        "report": _parse_ats_report(logs),
    }


def _try_compile_pdf(latex_code: str) -> str | None:
    """Write the .tex, attempt PDF compilation, and return the PDF path if built."""
    os.makedirs(settings.OUTPUT_DIR, exist_ok=True)
    tex_path = os.path.join(settings.OUTPUT_DIR, "tailored_resume.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(latex_code)

    if not shutil.which("pdflatex"):
        return None

    log_buffer = io.StringIO()
    with contextlib.redirect_stdout(log_buffer):
        ok = compile_pdf(tex_path)
    pdf_path = os.path.splitext(tex_path)[0] + ".pdf"
    return pdf_path if ok and os.path.exists(pdf_path) else None


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("⚙️ Configuration")

    key_present = bool(settings.GROQ_API_KEY)
    st.caption("Groq API key")
    if key_present:
        st.success("API key detected from environment / .env")
    else:
        st.warning("No API key found. Enter one below or run offline (keyword mode).")

    api_key_input = st.text_input(
        "GROQ_API_KEY",
        type="password",
        placeholder="gsk_...",
        help="Get a free key at https://console.groq.com/keys",
    )
    if api_key_input:
        _apply_api_key(api_key_input)
        key_present = True

    st.divider()
    use_llm = st.toggle(
        "AI tailoring (recommended)",
        value=True,
        help="Uses the LLM to tailor and ATS-optimise. Turn off for offline keyword-only mode.",
    )
    want_pdf = st.toggle(
        "Compile to PDF",
        value=False,
        help="Requires a LaTeX distribution (pdflatex) installed on this machine.",
    )

    st.divider()
    st.caption(f"Model: `{settings.MODEL_NAME}`")
    st.caption(f"Data folder: `{settings.DATA_DIR}`")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
st.title("📄 AI Resume Tailor")
st.write(
    "Upload your master resume, paste a job description, and generate an "
    "ATS-optimised, one-page resume tailored to that role."
)

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("1. Master resume")
    uploaded = st.file_uploader(
        "Upload resume PDF(s)",
        type=["pdf"],
        accept_multiple_files=True,
        help="These are saved to the data folder and reused on later runs.",
    )
    if uploaded:
        saved = _save_uploaded_pdfs(uploaded)
        st.success(f"Saved {len(saved)} file(s).")

    existing = _list_saved_pdfs()
    if existing:
        st.caption("Resumes on file:")
        for p in existing:
            st.markdown(f"- `{os.path.basename(p)}`")
    else:
        st.info("No resume PDFs found yet. Upload at least one to begin.")

with col_right:
    st.subheader("2. Job description")
    job_description = st.text_area(
        "Paste the target job description",
        height=280,
        placeholder="Paste the full job description here...",
    )

st.subheader("3. Review & generate")
st.caption(
    "Human-in-the-loop: first analyze the job description, then choose which "
    "missing skills to add and which projects to feature before generating."
)
analyze = st.button("🔍 Analyze job description", use_container_width=True)

if analyze:
    if not _list_saved_pdfs():
        st.error("Please upload at least one resume PDF first.")
    elif not job_description.strip():
        st.error("Please paste a job description.")
    else:
        try:
            with st.spinner("Analyzing job description against your resume..."):
                st.session_state["analysis"] = _analyze(job_description)
                st.session_state["analysis_jd"] = job_description
                st.session_state.pop("result", None)
                st.session_state.pop("pdf_path", None)
        except FileNotFoundError as e:
            st.error(f"{e}")
        except Exception as e:  # noqa: BLE001
            st.error(f"Something went wrong during analysis: {e}")

analysis = st.session_state.get("analysis")
if analysis:
    # Invalidate stale analysis if the job description changed.
    if st.session_state.get("analysis_jd", "") != job_description:
        st.warning("Job description changed since the last analysis. Re-run 'Analyze' to refresh.")

    st.markdown("#### 🧩 Skills review")
    matched = analysis.get("matched_skills", [])
    missing = analysis.get("missing_skills", [])

    if matched:
        st.markdown("**Already in your resume:** " + ", ".join(f"`{s}`" for s in matched))
    else:
        st.caption("No JD keywords matched your current resume.")

    chosen_skills: list[str] = []
    if missing:
        st.markdown(
            "**Missing JD skills** — select the ones you genuinely have so they can "
            "be added to your resume. Only pick skills you can honestly claim."
        )
        chosen_skills = st.multiselect(
            "Skills to add",
            options=missing,
            default=[],
            help="These will be inserted into your Technical Skills section.",
        )
    else:
        st.success("Great — your resume already covers the key JD skills.")

    st.markdown("#### 📌 Project selection")
    project_titles = [p.get("title", "") for p in analysis.get("projects", []) if p.get("title")]
    chosen_projects: list[str] = []
    if project_titles:
        labels = {
            p["title"]: (f"{p['title']} — {p['tech']}" if p.get("tech") else p["title"])
            for p in analysis.get("projects", [])
            if p.get("title")
        }
        st.markdown("Choose up to **2** projects to feature (most relevant to this role):")
        chosen_projects = st.multiselect(
            "Projects to include",
            options=project_titles,
            default=[],
            format_func=lambda t: labels.get(t, t),
            help="Leave empty to let the tailor pick the most JD-relevant projects.",
        )
        if len(chosen_projects) > 2:
            st.warning("Only the first 2 selected projects will be used.")
            chosen_projects = chosen_projects[:2]
    else:
        st.caption("No distinct projects detected in your resume.")

    st.markdown("#### 🚀 Generate")
    generate = st.button("Generate tailored resume", type="primary", use_container_width=True)

    if generate:
        if use_llm and not key_present:
            st.error(
                "AI tailoring is on but no API key is set. Enter a key in the sidebar, "
                "or switch off AI tailoring to use offline keyword mode."
            )
        else:
            try:
                with st.spinner("Tailoring your resume..."):
                    result = _generate(
                        job_description,
                        use_llm,
                        master_data=analysis.get("master_data"),
                        extra_skills=chosen_skills,
                        selected_projects=chosen_projects,
                    )
                st.session_state["result"] = result
                st.session_state["pdf_path"] = (
                    _try_compile_pdf(result["latex"]) if want_pdf else None
                )
                st.success("Done! See results below.")
            except FileNotFoundError as e:
                st.error(f"{e}")
            except Exception as e:  # noqa: BLE001 - surface any pipeline error to the user
                st.error(f"Something went wrong: {e}")


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #
result = st.session_state.get("result")
if result:
    st.divider()
    st.subheader("📊 ATS report")
    report = result["report"]

    m1, m2, m3 = st.columns(3)
    m1.metric("Keyword coverage", report.get("coverage") or "—")
    m2.metric("Quantified bullets", report.get("quantified") or "—")
    m3.metric("Repeated verbs", report.get("repeated_verbs") or "None")

    with st.expander("Full generation log"):
        st.code(result["logs"] or "(no log output)", language="text")

    st.subheader("📄 Output")
    tab_preview, tab_source = st.tabs(["Download", "LaTeX source"])

    with tab_preview:
        st.download_button(
            "⬇️ Download .tex",
            data=result["latex"],
            file_name="tailored_resume.tex",
            mime="text/x-tex",
            use_container_width=True,
        )
        pdf_path = st.session_state.get("pdf_path")
        if want_pdf:
            if pdf_path and os.path.exists(pdf_path):
                with open(pdf_path, "rb") as f:
                    st.download_button(
                        "⬇️ Download PDF",
                        data=f.read(),
                        file_name="tailored_resume.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                    )
            else:
                st.info(
                    "PDF not compiled. Install a LaTeX distribution (e.g. MiKTeX / TeX Live) "
                    "so `pdflatex` is on PATH, or compile the downloaded .tex on Overleaf."
                )
        st.caption(
            "Tip: paste the .tex into https://overleaf.com to render a PDF instantly."
        )

    with tab_source:
        st.code(result["latex"], language="latex")
