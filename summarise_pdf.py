"""Week 1 script: summarise a PDF with an LLM and log tokens and cost.

Setup (once):
    pip install anthropic pypdf python-dotenv
    Create a file named .env containing:  ANTHROPIC_API_KEY=your-key-here
    Add .env to .gitignore so the key never reaches GitHub.

Run:
    python summarise_pdf.py path/to/paper.pdf
"""

import csv
import sys
from datetime import datetime
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv
from pypdf import PdfReader

MODEL = "claude-sonnet-5-5"
# Prices change: check the provider's pricing page and update these (USD per million tokens).
PRICE_INPUT_PER_M = 3.00
PRICE_OUTPUT_PER_M = 15.00
LOG_FILE = Path("usage_log.csv")


def read_pdf(path: str) -> str:
    reader = PdfReader(path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def summarise(text: str, client: Anthropic):
    prompt = (
        "Summarise this document for a bank's AI risk committee in 5 bullet points, "
        "then list 3 claims a reviewer should verify against the source.\n\n"
        f"<document>\n{text}\n</document>"
    )
    response = client.messages.create(
        model=MODEL,
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(block.text for block in response.content if block.type == "text")
    return text, response.usage


def log_usage(pdf: str, usage) -> float:
    cost = (usage.input_tokens * PRICE_INPUT_PER_M + usage.output_tokens * PRICE_OUTPUT_PER_M) / 1_000_000
    is_new = not LOG_FILE.exists()
    with LOG_FILE.open("a", newline="") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["timestamp", "file", "model", "input_tokens", "output_tokens", "cost_usd"])
        writer.writerow([datetime.now().isoformat(timespec="seconds"), pdf, MODEL,
                         usage.input_tokens, usage.output_tokens, f"{cost:.4f}"])
    return cost


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python summarise_pdf.py path/to/file.pdf")
    load_dotenv()
    client = Anthropic()  # reads ANTHROPIC_API_KEY from the environment
    pdf_path = sys.argv[1]
    summary, usage = summarise(read_pdf(pdf_path), client)
    cost = log_usage(pdf_path, usage)
    print(summary)
    print(f"\nTokens: {usage.input_tokens} in, {usage.output_tokens} out. Approx cost: USD {cost:.4f}")
