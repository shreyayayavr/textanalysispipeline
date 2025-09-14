# scripts/extract_articles.py
"""
extract_articles.py
objective:
    This script extracts clean textual content from a set of URLs listed in an Excel file.
    each article is saved as a separate `.txt` file.A CSV log is also generated to record
    whether extraction was successful for each URL.
features:
    - fetches HTML with retry mechanism and custom headers
    - removes non-content HTML elements [ads,navigation,scripts]
    - attempts to find the main article container intelligently
    - filters out footer-like patterns [e.g.,copyright,contact info]
    - saves each article as a `.txt` file with its `URL_ID` as the filename

"""

import argparse, time, re
from pathlib import Path
import pandas as pd
import requests
from bs4 import BeautifulSoup

# HTTP request headers to avoid being blocked by some sites
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; BlackcofferTest/1.0)"}

def fetch_html(url, retries=2, timeout=15):
    """
    fetch raw HTML from a URL with retry support.
    Args:
        url (str): The target webpage URL
        retries (int): Number of retry attempts if request fails
        timeout (int): Max seconds to wait for a response.
    returns:
        str | None: HTML text if successful, else None.
    """
    for attempt in range(retries + 1):
        try:
            r = requests.get(url, headers=HEADERS, timeout=timeout)
            # Only proceed if HTTP status is OK and content type is HTML
            if 200 <= r.status_code < 300 and "text/html" in r.headers.get("Content-Type", ""):
                return r.text
        except requests.RequestException:
            pass  # Ignore and retry
        time.sleep(1 + attempt)  # Incremental delay before retry
    return None

def clean_soup(soup):
    """
    remove unwanted HTML elements such as scripts,styles,navigation bars,etc.
    this helps isolate meaningful content
    Args:
        soup (BeautifulSoup): Parsed HTML document
    Returns:
        BeautifulSoup: Cleaned HTML
    """
    for tag in soup(["script", "style", "noscript", "header", "footer", "svg", "img", "aside", "form", "nav"]):
        tag.decompose()
    return soup

def pick_title(soup):
    """
    determine the best title for the article
    Args:
        soup (BeautifulSoup): Parsed HTML document
    Returns:
        str: Extracted title or empty string if not found
    """
    h1 = soup.find("h1")
    if h1 and h1.get_text(strip=True):
        return h1.get_text(strip=True)

    meta = soup.find("meta", property="og:title")
    if meta and meta.get("content"):
        return meta["content"].strip()

    if soup.title and soup.title.get_text():
        return soup.title.get_text(strip=True)

    return ""

def pick_article_container(soup):
    """
    Attempt to locate the main body of the article.
    Strategy:
        - Search for common content wrapper elements.
        - Select the one with the largest text length (likely main content).
    Args:
        soup (BeautifulSoup): Parsed HTML document.
    Returns:
        Tag: The most likely HTML container for article text
    """
    candidates = []
    selectors = [
        "article", "[class*='content']", "[class*='post']", "[class*='article']",
        "main", "div[itemprop='articleBody']"
    ]
    for sel in selectors:
        for node in soup.select(sel):
            text = node.get_text(" ", strip=True)
            if len(text.split()) > 100:  # Only consider sufficiently long bodies
                candidates.append((len(text), node))

    # Return the container with the most text, else return the whole soup
    return max(candidates, key=lambda x: x[0])[1] if candidates else soup

def extract_article_text(html):
    """
    extract clean text content from raw HTML.
    steps:
        1.Parse HTML.
        2.Clean out irrelevant tags.
        3.Identify title and main article container.
        4.Extract paragraphs (<p>) from the chosen container.
        5.Remove unwanted lines using blacklist patterns.
        6.Join into final article text.
    Args:
        html (str): Raw HTML content
    Returns:
        str: Cleaned article text (title + body)
    """
    soup = BeautifulSoup(html, "lxml")
    soup = clean_soup(soup)

    title = pick_title(soup)
    container = pick_article_container(soup)

    # Get paragraph text from container
    ps = [p.get_text(" ", strip=True) for p in (
        container.find_all("p") if container is not soup else soup.find_all("p")
    )]

    # Regex patterns to remove common footer/header clutter
    blacklist_patterns = [
        r"blackcoffer",
        r"©",
        r"contact\s+us?",
        r"firm\s+name",
        r"all\s+rights?\s+reserved",
        r"client\s*:",
        r"industry\s*type\s*:",
        r"products?\s*&\s*services?\s*:",
        r"organization\s+size\s*:",
        r"we\s+provide\s+intelligence",
        r"rising\s+it\s+cities.*\d{4}",
        r"https?://\S+",
    ]

    # Keep only paragraphs that do not match blacklist patterns
    filtered_ps = [p for p in ps if not any(re.search(pattern, p.lower()) for pattern in blacklist_patterns)]

    # Combine title and body
    text = (title + "\n\n" + "\n".join(filtered_ps)).strip()

    # Remove excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text

def main(input_xlsx, outdir, log_csv):
    """
    orchestrates reading URLs from Excel, extracting each article,
    saving to .txt files, and writing a log
    Args:
        input_xlsx (str): Path to input Excel file
        outdir (str): Directory to store extracted text files
        log_csv (str): Path to output log CSV file
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(input_xlsx)
    logs = []

    for _, row in df.iterrows():
        url_id, url = str(row["URL_ID"]), row["URL"]
        html = fetch_html(url)

        if not html:
            logs.append({"URL_ID": url_id, "URL": url, "status": "fail"})
            continue

        text = extract_article_text(html)
        (outdir / f"{url_id}.txt").write_text(text, encoding="utf-8")

        logs.append({"URL_ID": url_id, "URL": url, "status": "ok", "words": len(text.split())})

    if log_csv:
        pd.DataFrame(logs).to_csv(log_csv, index=False)

    print("done")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Extract articles from URLs in Excel file.")
    ap.add_argument("--input", required=True, help="Path to the input Excel file.")
    ap.add_argument("--outdir", default="articles", help="Directory for saving article .txt files.")
    ap.add_argument("--log", default="output/extract_log.csv", help="Path to the CSV log file.")
    args = ap.parse_args()

    Path("output").mkdir(exist_ok=True)  # Ensure output folder exists
    main(args.input, args.outdir, args.log)
