"""Build Swastham_RAG_Project_Report.pdf from report.html with headless Chrome, then add page numbers.

Usage: python docs/report/build_report.py --repo-url https://github.com/<user>/swastham-rag-demo
"""
import argparse
import subprocess
import tempfile
from pathlib import Path

import pymupdf

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
OUTPUT = ROOT / "Swastham_RAG_Project_Report.pdf"
BROWSERS = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
]


def render(html_path: Path, pdf_path: Path) -> None:
    browser = next(b for b in BROWSERS if Path(b).exists())
    subprocess.run(
        [browser, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
         f"--print-to-pdf={pdf_path}", html_path.as_uri()],
        check=True, capture_output=True, timeout=120,
    )


def add_page_numbers(src: Path, dst: Path) -> None:
    doc = pymupdf.open(src)
    total = len(doc)
    for i, page in enumerate(doc):
        if i == 0:
            continue  # cover
        w, h = page.rect.width, page.rect.height
        page.insert_text((48, h - 28), "Swastham Multilingual Health RAG · Project Report",
                         fontsize=7.5, color=(0.35, 0.4, 0.45))
        label = f"{i + 1} / {total}"
        page.insert_text((w - 48 - pymupdf.get_text_length(label, fontsize=7.5), h - 28), label,
                         fontsize=7.5, color=(0.35, 0.4, 0.45))
    doc.set_metadata({"title": "Swastham Multilingual Health RAG: Project Report",
                      "author": "Chetanya Jain", "subject": "Proof of concept report"})
    doc.save(dst, garbage=3, deflate=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-url", required=True)
    args = parser.parse_args()

    html = (HERE / "report.html").read_text(encoding="utf-8").replace("{{REPO_URL}}", args.repo_url)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_html = Path(tmp) / "report.html"
        tmp_pdf = Path(tmp) / "raw.pdf"
        tmp_html.write_text(html, encoding="utf-8")
        render(tmp_html, tmp_pdf)
        add_page_numbers(tmp_pdf, OUTPUT)
    print(f"Wrote {OUTPUT} ({len(pymupdf.open(OUTPUT))} pages)")


if __name__ == "__main__":
    main()
