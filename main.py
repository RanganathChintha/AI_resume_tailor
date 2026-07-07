import os
import sys
from app.data_manager.cache import ResumeCacheManager
from app.generator.tailor import ResumeGenerator
from app.core.config import settings

def main():
    print("\n=== AI Resume Tailor (Modular Edition) ===")

    # Check for --no-llm-review flag
    use_llm_review = '--no-llm-review' not in sys.argv

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
        print("[INFO] LLM review disabled (--no-llm-review flag)")

    # Execute Pipeline
    try:
        master_data = cache_manager.get_resume_data()
        latex_code = generator.tailor_resume(jd_text=job_description, master_resume_text=master_data, use_llm_review=use_llm_review)

        output_filename = os.path.join(settings.OUTPUT_DIR, "tailored_resume.tex")
        with open(output_filename, "w", encoding="utf-8") as f:
            f.write(latex_code)

        print(f"\n[SUCCESS] Tailored resume saved to: {output_filename}")
        print("Compile it by navigating to the output folder and running: pdflatex tailored_resume.tex")

    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
    except Exception as e:
        print(f"\n[CRITICAL ERROR] An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()