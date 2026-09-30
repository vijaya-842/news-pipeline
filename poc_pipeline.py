"""Pipeline: same RSS/BeautifulSoup link discovery as before, now with:
  - a Playwright fallback when a plain `requests` fetch fails (403 / timeout),
    since some sites block non-browser requests outright
  - news-fetch (fetch_many) for full-article content extraction, replacing
    trafilatura
  - Excel (.xlsx) output instead of JSON

Everything about HOW sources are found (RSS discovery, HTML scrape fallback,
article-link heuristics) is unchanged from the previous version -- only the
page-fetch step (added Playwright fallback) and the content-extraction /
output steps changed.

Usage:
  python poc_pipeline.py --hours 24 --workers 8
"""
import argparse
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import urljoin, urlparse

import feedparser
import pandas as pd
import requests
import yaml
from bs4 import BeautifulSoup

from newsfetch import fetch_many

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
SOURCES_FILE = BASE_DIR / "config" / "sources.yaml"
OUTPUT_FILE = BASE_DIR / "output" / "news_content.xlsx"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0 Safari/537.36"
}
TIMEOUT = 20
ARTICLE_PATH_WORDS = (
    "article", "news", "press-release", "pressrelease", "release", "story", "post", "media"
)
NAV_PATH_BLOCKLIST = (
    "privacy", "terms", "cookie", "contact", "about-us", "about", "careers",
    "jobs", "login", "signin", "sign-in", "signup", "sign-up", "subscribe",
    "newsletter", "advertise", "sitemap", "search", "tag/", "tags/",
    "category/", "author/", "page/", "archive",
)
MAX_SCRAPE_CANDIDATES = 20


# ---------------------------------------------------------------------------
# Page fetching: requests first, Playwright fallback on failure
# ---------------------------------------------------------------------------
def fetch_with_playwright(url: str, timeout_ms: int = 25000) -> str | None:
    """Renders the page with a real headless browser. Used only when a plain
    `requests` GET fails (403, timeout, etc.) -- a real browser fingerprint
    gets past a good chunk of basic bot-blocking that a raw HTTP request
    trips. Returns None (rather than raising) if it still fails, so callers
    can treat it the same as any other unreachable source."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return None
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            try:
                context = browser.new_context(user_agent=HEADERS["User-Agent"])
                page = context.new_page()
                page.goto(url, timeout=timeout_ms, wait_until="domcontentloaded")
                return page.content()
            finally:
                browser.close()
    except Exception:
        return None


def request_text(url: str) -> str:
    """Plain HTTP fetch with retries, then a single Playwright attempt if
    every retry fails. Raises requests.RequestException only if BOTH the
    plain fetch and the Playwright fallback fail."""
    last_error = None
    for attempt in range(3):
        try:
            response = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
            response.raise_for_status()
            return response.text
        except requests.HTTPError as error:
            if error.response is not None and error.response.status_code in {401, 403, 404, 410}:
                last_error = error
                break  # no point retrying a hard block, go straight to Playwright
            last_error = error
        except requests.RequestException as error:
            last_error = error
        if attempt < 2:
            time.sleep(2 ** attempt)

    rendered = fetch_with_playwright(url)
    if rendered is not None:
        return rendered
    raise last_error


# ---------------------------------------------------------------------------
# Datetime parsing
# ---------------------------------------------------------------------------
def parse_datetime(value) -> datetime | None:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        text = value.strip()
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            try:
                parsed = parsedate_to_datetime(text)
            except (TypeError, ValueError, IndexError):
                return None
    else:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


# ---------------------------------------------------------------------------
# Feed discovery + parsing (unchanged logic)
# ---------------------------------------------------------------------------
def discover_feed_url(page_url: str, html: str) -> str | None:
    soup = BeautifulSoup(html, "html.parser")
    for link_tag in soup.find_all("link", rel=lambda value: value and "alternate" in value):
        type_attr = (link_tag.get("type") or "").lower()
        href = link_tag.get("href")
        if href and ("rss" in type_attr or "atom" in type_attr or "feed" in type_attr):
            return urljoin(page_url, href)
    return None


def guess_feed_url(page_url: str) -> str | None:
    parsed = urlparse(page_url)
    base = f"{parsed.scheme}://{parsed.netloc}"
    for suffix in ("/feed", "/rss", "/rss.xml", "/feed.xml", "/atom.xml"):
        candidate = base + suffix
        try:
            response = requests.get(candidate, headers=HEADERS, timeout=TIMEOUT)
            content_type = response.headers.get("content-type", "").lower()
            if response.ok and ("xml" in content_type or "<rss" in response.text[:2000].lower() or "<feed" in response.text[:2000].lower()):
                return candidate
        except requests.RequestException:
            continue
    return None


def collect_from_feed(source: dict, feed_url: str) -> list[dict]:
    parsed = feedparser.parse(feed_url, request_headers=HEADERS)
    articles = []
    for entry in parsed.entries:
        published = parse_datetime(entry.get("published") or entry.get("updated"))
        if not published or not entry.get("link"):
            continue
        articles.append({
            "source": source["name"],
            "title": entry.get("title", ""),
            "url": entry["link"],
            "published": published.isoformat(),
            "description": BeautifulSoup(entry.get("summary", ""), "html.parser").get_text(" ", strip=True),
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        })
    return articles


# ---------------------------------------------------------------------------
# HTML scrape fallback (unchanged logic)
# ---------------------------------------------------------------------------
def is_article_link(source_url: str, link: str, text: str) -> bool:
    parsed_source = urlparse(source_url)
    parsed_link = urlparse(link)
    if parsed_link.scheme not in ("", "http", "https"):
        return False
    if parsed_link.netloc and parsed_link.netloc != parsed_source.netloc:
        return False
    path_lower = parsed_link.path.lower()
    if not parsed_link.path or parsed_link.path in ("/", parsed_source.path):
        return False
    if any(nav_word in path_lower for nav_word in NAV_PATH_BLOCKLIST):
        return False
    if any(word in path_lower for word in ARTICLE_PATH_WORDS):
        return True
    return len(text.strip()) >= 25 and bool(re.search(r"/[^/]+$", parsed_link.path))


def discover_article_links(source_url: str, html: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    links = []
    for anchor in soup.find_all("a", href=True):
        href = urljoin(source_url, anchor["href"].split("#", 1)[0])
        if is_article_link(source_url, href, anchor.get_text(" ", strip=True)) and href not in links:
            links.append(href)
        if len(links) >= MAX_SCRAPE_CANDIDATES:
            break
    return links


def first_meta(soup: BeautifulSoup, selectors: list[tuple[str, str]]) -> str | None:
    for attribute, value in selectors:
        tag = soup.find("meta", attrs={attribute: value})
        if tag and tag.get("content"):
            return tag["content"].strip()
    return None


def extract_article_stub(source_name: str, article_url: str, html: str) -> dict | None:
    """Cheap metadata-only extraction used during link discovery/scraping --
    NOT the final content. Full body text comes later from news-fetch."""
    soup = BeautifulSoup(html, "html.parser")
    title = first_meta(soup, [("property", "og:title"), ("name", "twitter:title")])
    if not title and soup.title:
        title = soup.title.get_text(" ", strip=True)
    published = first_meta(soup, [
        ("property", "article:published_time"), ("name", "date"),
        ("name", "pubdate"), ("itemprop", "datePublished"),
    ])
    if not published:
        time_tag = soup.find("time", datetime=True)
        published = time_tag.get("datetime") if time_tag else None
    description = first_meta(soup, [("property", "og:description"), ("name", "description")])
    canonical = soup.find("link", rel="canonical")
    if not title or not published:
        return None
    return {
        "source": source_name,
        "title": title,
        "url": urljoin(article_url, canonical.get("href")) if canonical and canonical.get("href") else article_url,
        "published": published,
        "description": description,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }


def collect_source(source: dict, cutoff: datetime) -> list[dict]:
    page_url = source["url"]
    html = request_text(page_url)
    fetch_mode = source.get("fetch_mode", "auto").lower()
    feed_url = source.get("feed_url")
    if fetch_mode != "scrape" and not feed_url:
        feed_url = discover_feed_url(page_url, html) or guess_feed_url(page_url)
    if fetch_mode != "scrape" and feed_url:
        try:
            articles = collect_from_feed(source, feed_url)
            if articles:
                return [a for a in articles if parse_datetime(a["published"]) >= cutoff]
        except Exception as error:
            print(f"  feed failed: {error}")
            if fetch_mode == "feed":
                return []
    if fetch_mode == "feed":
        return []

    articles = []
    for article_url in discover_article_links(page_url, html):
        try:
            article = extract_article_stub(source["name"], article_url, request_text(article_url))
            published = parse_datetime(article["published"]) if article else None
            if article and published and published >= cutoff:
                article["published"] = published.isoformat()
                articles.append(article)
        except requests.RequestException as error:
            print(f"  article failed {article_url}: {error}")
    return articles


def collect_source_safe(source: dict, cutoff: datetime) -> tuple[str, list[dict], str | None]:
    try:
        return source["name"], collect_source(source, cutoff), None
    except Exception as error:
        return source.get("name", "unnamed source"), [], str(error)


# ---------------------------------------------------------------------------
# Content extraction via news-fetch
# ---------------------------------------------------------------------------
def add_full_content(articles: list[dict], max_workers: int) -> list[dict]:
    urls = [a["url"] for a in articles]
    fetched = fetch_many(urls, max_workers=max_workers)  # list[Article | None], same order as urls

    for stub, article in zip(articles, fetched):
        if article is None:
            stub["content"] = None
            stub["authors"] = None
            stub["content_error"] = "news-fetch returned no result (fetch failed or low confidence)"
            continue
        stub["content"] = article.text
        stub["authors"] = ", ".join(article.authors) if article.authors else None
        # Prefer news-fetch's own published_at when our stub's date parsing
        # was weak (e.g. from a raw meta tag string) but keep our value if
        # news-fetch didn't find one.
        if article.published_at and not stub.get("published"):
            stub["published"] = article.published_at.isoformat() if hasattr(article.published_at, "isoformat") else str(article.published_at)
    return articles


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch current news + full content from a curated source list.")
    parser.add_argument("--hours", type=float, default=24)
    parser.add_argument("--workers", type=int, default=8, help="Concurrent source fetches.")
    parser.add_argument("--content-workers", type=int, default=8, help="Concurrent article-content fetches (news-fetch).")
    parser.add_argument("--output", type=Path, default=OUTPUT_FILE)
    args = parser.parse_args()

    sources = yaml.safe_load(SOURCES_FILE.read_text(encoding="utf-8")).get("sources", [])
    cutoff = datetime.now(timezone.utc) - timedelta(hours=args.hours)

    print(f"Collecting from {len(sources)} sources (last {args.hours}h)...")
    all_articles = []
    worker_count = max(1, min(args.workers, 16))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(collect_source_safe, source, cutoff) for source in sources]
        for future in as_completed(futures):
            source_name, articles, error = future.result()
            if error:
                print(f"  {source_name}: failed ({error})")
                continue
            print(f"  {source_name}: {len(articles)} article(s)")
            all_articles.extend(articles)

    unique_articles = list({a["url"]: a for a in all_articles if a.get("url")}.values())
    print(f"\n{len(unique_articles)} unique articles collected. Fetching full content via news-fetch...")

    content_worker_count = max(1, min(args.content_workers, 16))
    results = add_full_content(unique_articles, content_worker_count)

    with_content = sum(1 for a in results if a.get("content"))
    print(f"{with_content}/{len(results)} articles got usable full content "
          f"({len(results) - with_content} had no content extracted or failed to fetch).")

    results.sort(key=lambda a: a["published"], reverse=True)
    output_rows = [
        {
            "source": a["source"],
            "title": a["title"],
            "url": a["url"],
            "description": a.get("description"),
            "authors": a.get("authors"),
            "content": a.get("content"),
            "published": a["published"],
            "fetched_at": a["fetched_at"],
        }
        for a in results
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(output_rows)
    df.to_excel(args.output, index=False, engine="openpyxl")
    print(f"\nSaved {len(output_rows)} article(s) to {args.output}")


if __name__ == "__main__":
    main()