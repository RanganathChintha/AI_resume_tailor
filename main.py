import os
import sys
import shutil
import subprocess
from app.data_manager.cache import ResumeCacheManager
from app.generator.tailor import ResumeGenerator
from app.core.config import settings


def compile_pdf(tex_path: str) -> bool:
    """Attempt to compile the .tex file to PDF using pdflatex, if available."""
    pdflatex = shutil.which("pdflatex")
    if not pdflatex:
        print("[INFO] pdflatex not found on PATH - skipping PDF compilation. "
              "Install a LaTeX distribution (e.g. MiKTeX/TeX Live) to auto-generate the PDF.")
        return False

    output_dir = os.path.dirname(tex_path)
    try:
        result = subprocess.run(
            [pdflatex, "-interaction=nonstopmode", "-halt-on-error",
             os.path.basename(tex_path)],
            cwd=output_dir,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        print("[WARN] pdflatex timed out.")
        return False

    pdf_path = os.path.splitext(tex_path)[0] + ".pdf"
    if result.returncode == 0 and os.path.exists(pdf_path):
        print(f"[SUCCESS] PDF compiled: {pdf_path}")
        return True

    # Surface the most relevant error line for debugging
    error_lines = [ln for ln in result.stdout.splitlines() if ln.startswith("!")]
    detail = error_lines[0] if error_lines else "see pdflatex log in the output folder"
    print(f"[WARN] PDF compilation failed: {detail}")
    return False


def _is_valid_latex(latex_code: str) -> bool:
    """Return True only if the generated LaTeX looks like a complete document."""
    if not latex_code or not latex_code.strip():
        return False
    return "\\begin{document}" in latex_code and "\\end{document}" in latex_code


def _minimal_placeholder_latex(master_data: str) -> str:
    """
    Build a minimal but valid LaTeX document from the raw resume text.

    This is a last-resort safety net so that a first-time user always receives a
    usable output file, even if tailoring could not produce structured content.
    """
    def esc(text: str) -> str:
        for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                     ("$", r"\$"), ("#", r"\#"), ("_", r"\_"), ("{", r"\{"),
                     ("}", r"\}"), ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}")):
            text = text.replace(a, b)
        return text

    lines = [ln.strip() for ln in (master_data or "").split("\n") if ln.strip()]
    body = "\n".join(r"\item " + esc(ln) for ln in lines[:40]) or r"\item No resume content was available."

    return (
        "\\documentclass[letterpaper,11pt]{article}\n"
        "\\usepackage[margin=0.7in]{geometry}\n"
        "\\usepackage{enumitem}\n"
        "\\pagestyle{empty}\n"
        "\\begin{document}\n"
        "\\section*{Resume}\n"
        "\\begin{itemize}[leftmargin=1.4em]\n"
        f"{body}\n"
        "\\end{itemize}\n"
        "\\end{document}\n"
    )


def main():
    print("\n=== AI Resume Tailor (Modular Edition) ===")
    use_llm_review = '--no-llm-review' not in sys.argv
    # Check for --pdf flag to auto-compile
    compile_to_pdf = '--pdf' in sys.argv

    # Initialize Managers
    try:
        cache_manager = ResumeCacheManager()
        generator = ResumeGenerator()
    except ValueError as e:
        print(f"Configuration Error: {e}")
        return

    # Get Job Description
    print("\nPlease paste the Job Description (press Enter twice to finish):")
    jd_lines = []
    while True:
        line = input()
        if not line:
            break
        jd_lines.append(line)
    job_description = "\n".join(jd_lines)

    if not job_description.strip():
        print("Job Description cannot be empty. Exiting.")
        return

    if not use_llm_review:
        print("[INFO] LLM tailoring disabled (--no-llm-review flag)")

    # Execute Pipeline
    try:
        master_data = cache_manager.get_resume_data()
        latex_code = generator.tailor_resume(jd_text=job_description, master_resume_text=master_data, use_llm_review=use_llm_review)

        output_filename = os.path.join(settings.OUTPUT_DIR, "tailored_resume.tex")

        # Guarantee the user always gets a usable output file. If tailoring
        # somehow produced an empty/broken document, fall back to a minimal
        # placeholder built from the raw resume text rather than writing nothing.
        if not _is_valid_latex(latex_code):
            print("\n[WARN] Tailoring produced no usable content - "
                  "writing a minimal placeholder resume so you still get an output. "
                  "Check that GROQ_API_KEY is set and the master resume PDF has extractable text, then retry.")
            latex_code = _minimal_placeholder_latex(master_data)

        os.makedirs(settings.OUTPUT_DIR, exist_ok=True)
        with open(output_filename, "w", encoding="utf-8") as f:
            f.write(latex_code)

        print(f"\n[SUCCESS] Tailored resume saved to: {output_filename}")

        if compile_to_pdf:
            compile_pdf(output_filename)
        else:
            print("Compile it by navigating to the output folder and running: pdflatex tailored_resume.tex")
            print("(or re-run with the --pdf flag to auto-compile)")

    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
    except Exception as e:
        print(f"\n[CRITICAL ERROR] An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()