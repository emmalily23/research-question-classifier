"""
PDF Text Extractor using GROBID

This script processes a collection of PDF papers using the GROBID server to extract structured full-text content. 
It uses metadata generated from a previous scraping step (downloadPapers.py) and outputs the extracted texts in JSONL format,
grouped by conference (ACL, EMNLP, EACL).

Main features:
- Sends each PDF to GROBID for TEI XML conversion.
- Extracts the title and main body text from the XML.
- Saves results to `Data/extracted_texts-MODEL/{conference}.jsonl`.

Prerequisites:
- GROBID server running locally at http://localhost:8070
- PDF files downloaded to `Data/pdfs-MODEL/`
- Metadata CSV file at `Data/metadata-MODEL.csv`
"""

import os
import csv
import time
import json
import logging
from pathlib import Path
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
from datetime import datetime
import argparse

# GROBID endpoint
GROBID_URL = "http://localhost:8070/api/processFulltextDocument"

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pdf_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def check_grobid_health():
    """Check if the GROBID server is up and responding."""
    try:
        health_url = GROBID_URL.replace("/api/processFulltextDocument", "/api/isalive")
        resp = requests.get(health_url, timeout=15)
        if resp.status_code == 200 and resp.text.strip() == "true":
            logger.info("GROBID server is alive.")
            return True
        else:
            logger.error("GROBID server is not responding properly.")
            return False
    except Exception as e:
        logger.error(f"GROBID health check failed: {e}")
        return False


def get_conference_from_id(paper_id):
    """Extract conference name from the paper ID."""
    paper_id = paper_id.lower()
    if "emnlp" in paper_id:
        return "emnlp"
    elif "eacl" in paper_id:
        return "eacl"
    elif "acl" in paper_id:
        return "acl"
    else:
        return "unknown"


def load_existing_ids(output_dir):
    """Load paper IDs that have already been processed and saved."""
    existing_ids = set()
    if not output_dir.exists():
        return existing_ids

    for file in output_dir.glob("*.jsonl"):
        with open(file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    existing_ids.add(json.loads(line)["paper_id"])
                except Exception:
                    continue
    return existing_ids


def write_result_to_conference_file(result, conference, output_dir):
    """Append extracted result to the appropriate conference file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"{conference}.jsonl"
    with open(out_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(result) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Process PDFs with GROBID and extract texts.")
    parser.add_argument("model", nargs="?", choices=["bert", "scibert"], default="bert",
                        help="Model name, determines input/output dirs (default: %(default)s)")
    args = parser.parse_args()

    model = args.model.upper()
    pdf_dir = Path(f"Data/pdfs-{model}")
    metadata_csv = Path(f"Data/metadata-{model}.csv")
    output_dir = Path(f"Data/extracted_texts-{model}")

    if not check_grobid_health():
        return

    existing_ids = load_existing_ids(output_dir)

    # Load metadata from CSV
    papers = []
    with open(metadata_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["id"] not in existing_ids:
                papers.append(row)

    logger.info(f"Processing {len(papers)} new PDFs...")

    processed = 0
    written = 0

    for paper in tqdm(papers, desc="Processing PDFs", unit="paper"):
        pdf_path = Path(paper["local_path"])
        if not pdf_path.exists():
            logger.warning(f"Missing PDF: {pdf_path}")
            continue

        try:
            logger.info(f"Processing {paper['id']}...")

            with open(pdf_path, "rb") as f:
                response = requests.post(
                    GROBID_URL,
                    files={"input": f},
                    headers={"Accept": "application/xml"},
                    timeout=50
                )
            time.sleep(2)

            if response.status_code != 200:
                logger.error(f"GROBID failed on {paper['id']} (status {response.status_code})")
                continue

            soup = BeautifulSoup(response.text, "xml")
            body = soup.find("body")
            if not body:
                logger.warning(f"No body found for {paper['id']}")
                continue

            paragraphs = [p.get_text(separator=" ", strip=True) for p in body.find_all("p")]
            body_text = "\n\n".join(paragraphs)

            title_stmt = soup.find("titleStmt")
            title = title_stmt.find("title").get_text(strip=True) if title_stmt else ""

            result = {
                "paper_id": paper["id"],
                "title": title,
                "pdf_url": paper["url"],
                "body_text": body_text
            }

            conference = get_conference_from_id(paper["id"])
            write_result_to_conference_file(result, conference, output_dir)
            written += 1
            logger.info(f"Written: {paper['id']} to {conference}.jsonl")

        except requests.exceptions.Timeout:
            logger.error(f"Timeout processing {paper['id']}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error for {paper['id']}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error for {paper['id']}: {type(e).__name__}: {e}")
        processed += 1

    logger.info(f"\nProcessing complete. {written} written, {processed - written} skipped.")
    logger.info(f"Output saved in: {output_dir.resolve()}")


if __name__ == "__main__":
    logger.info(f"Started: {datetime.now()}")
    main()
    logger.info(f"Finished: {datetime.now()}")
