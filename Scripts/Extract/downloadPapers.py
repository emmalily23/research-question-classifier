"""
PDF Downloader for ACL Anthology Conference Papers

This script:
1. Scrapes a specific ACL Anthology event page for PDF paper links.
2. Downloades a number of papers (default 100).
3. Saves downloaded PDFs into a local directory.
4. Generates and saves metadata (paper ID, URL, file path) to a CSV.

Customise:
- `event_url`: the ACL event page to scrape (e.g. ACL 2022, EMNLP 2024)
- `max_papers`: limit the number of papers to download for speed and storage control

Output:
- `Data/pdfs-MODEL/`: folder containing downloaded PDFs.
- `Data/metadata-MODEL.csv`: file with metadata of downloaded papers.

Usage:
- Default run (uses config inside script):
    python downloadPapers.py
- Custom run (override link + model type):
    python downloadPapers.py "https://aclanthology.org/events/acl-2022/" scibert
"""

import os
import re
import pandas as pd
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import argparse

def setup_directories(model):
    """Create data directories depending on model."""
    os.makedirs(f"Data/pdfs-{model.upper()}", exist_ok=True)
    os.makedirs("Data", exist_ok=True)

def detect_paper_prefix(event_url):
    """
    Get paper prefix (e.g. '/2021.eacl') from the event URL.
    Examples:
        https://aclanthology.org/events/eacl-2021/: /2021.eacl
        https://aclanthology.org/events/acl-2022/ : /2022.acl
    """
    match = re.search(r"/events/([a-z]+)-(\d{4})/?", event_url)
    if not match:
        raise ValueError(f"Could not detect prefix from event URL: {event_url}")
    
    conf, year = match.groups()
    return f"/{year}.{conf}"

def get_pdf_links(base_url, paper_prefix):
    """Scrape the HTML of the conference page and return a list of valid PDF links."""
    print(f"\nCONFIRMED FINAL URL: {base_url}\n")
    response = requests.get(base_url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser')
    
    pdf_links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.endswith(".pdf") and paper_prefix in href and not href.endswith(".0.pdf"):
            paper_id = href.split("/")[-1].replace(".pdf", "")
            if paper_id.split(".")[-1].isdigit():
                pdf_links.append(href)

    print(f"Found {len(pdf_links)} PDF links")
    return pdf_links

def download_papers(pdf_links, model, max_downloads=100):
    """Download PDFs and collect metadata."""
    metadata = []
    downloaded_count = 0

    for href in pdf_links:
        if downloaded_count >= max_downloads:
            break

        paper_id = href.split("/")[-1].replace(".pdf", "")
        if paper_id.endswith(".0"):
            continue

        filename = f"Data/pdfs-{model.upper()}/{paper_id}.pdf"
        pdf_url = urljoin("https://aclanthology.org", href)

        if os.path.exists(filename):
            print(f"Skipping {paper_id} (already exists)")
            continue

        print(f"Downloading {paper_id} ({downloaded_count + 1}/{max_downloads})")
        try:
            response = requests.get(pdf_url, timeout=10)
            response.raise_for_status()
            with open(filename, "wb") as f:
                f.write(response.content)

            metadata.append({
                "id": paper_id,
                "url": pdf_url,
                "local_path": filename
            })
            downloaded_count += 1

        except Exception as e:
            print(f"Failed to download {paper_id}: {e}")
    
    return metadata

def save_metadata(metadata, model):
    """Append new PDF metadata to a CSV."""
    output_path = f"Data/metadata-{model.upper()}.csv"

    if os.path.exists(output_path):
        existing_df = pd.read_csv(output_path)
        existing_ids = set(existing_df["id"])
    else:
        existing_df = pd.DataFrame(columns=["id", "url", "local_path"])
        existing_ids = set()

    new_entries = [entry for entry in metadata if entry["id"] not in existing_ids]

    if new_entries:
        new_df = pd.DataFrame(new_entries)
        updated_df = pd.concat([existing_df, new_df], ignore_index=True)
        updated_df.to_csv(output_path, index=False, encoding="utf-8")
        print(f"\nAppended {len(new_entries)} new entries to metadata CSV.")
    else:
        print("\nNo new entries to append. Metadata is up to date.")

    print(f"Metadata saved to: {os.path.abspath(output_path)}")

def main():
    # CLI parser
    parser = argparse.ArgumentParser(description="ACL Anthology PDF Downloader")
    parser.add_argument("event_url", nargs="?", default="https://aclanthology.org/events/eacl-2021/",
                        help="ACL Anthology event page URL (default: %(default)s)")
    parser.add_argument("model", nargs="?", choices=["bert", "scibert"], default="bert",
                        help="Model type, affects output dirs/files (default: %(default)s)")
    args = parser.parse_args()

    # CONFIGURE defaults
    event_url = args.event_url
    paper_prefix = detect_paper_prefix(event_url)
    max_papers = 100
    model = args.model

    setup_directories(model)
    pdf_links = get_pdf_links(event_url, paper_prefix)
    metadata = download_papers(pdf_links, model, max_downloads=max_papers)
    
    print(f"\nDownload summary:")
    print(f"Total PDFs available: {len(pdf_links)}")
    print(f"Attempted downloads: {min(max_papers, len(pdf_links))}")
    print(f"Successful downloads: {len(metadata)}")
    
    save_metadata(metadata, model)

if __name__ == "__main__":
    main()
