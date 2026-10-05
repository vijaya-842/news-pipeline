"""Pharma/biotech news aggregation pipeline.

Key behaviors:

  1. Fetch order per source/page: PLAYWRIGHT FIRST (real headless browser,
     gets past basic bot-blocking that a plain HTTP request trips). Only if
     Playwright fails do we fall back to a plain `requests` GET. For the
     SOURCE's own listing/feed page specifically, if BOTH Playwright and
     plain requests fail to load the page at all, we make one more attempt:
     guess common RSS feed URLs (/feed, /rss.xml, etc.) directly via plain
     requests, without needing the page HTML. Only if THAT also fails do we
     give up on the source and log it as a failure -- i.e. keep trying the
     next method rather than stopping at the first failure.

  2. Timeout is now a CLI flag (--timeout, default 30s) instead of a
     hardcoded 20s, since 20s was producing false-failures on sources that
     are just slow rather than actually blocked.

Output: two-sheet Excel file --
  Sheet1 "Articles"  -- every article successfully found AND with usable
                        full content from news-fetch (or the BS4 fallback).
  Sheet2 "Failures"  -- everything else, with url + stage + reason:
                          - source_fetch        page unreachable via BOTH
                                                 Playwright and requests,
                                                 AND the RSS-guess fallback
                                                 also failed/found nothing
                          - article_scrape      an individual article link
                                                 couldn't be fetched or had
                                                 no usable title/date
                          - content_extraction  news-fetch AND the BS4
                                                 fallback both couldn't
                                                 extract usable body text

Usage:
  python poc_pipeline.py --hours 24 --workers 8 --timeout 30
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
DEFAULT_TIMEOUT = 30  # seconds; was 20, bumped because 20s was flagging
                       # slow-but-working sources as errors
# STRONG words are specific enough that finding them in a path reliably
# means "this is an individual article", not a category/hub page.
STRONG_ARTICLE_PATH_WORDS = ("article",)
# WEAK words show up in both real articles AND category/hub pages on sites
# like GlobeNewswire, PR Newswire, and Newswire Canada (e.g. a hub page at
# "/news-releases/health-latest-news/" contains "release" and "news" just
# like a real article does). A weak-word match only counts as an article
# if the path ALSO has the telltale signature of those sites' real article
# URLs: a long numeric ID baked into the last path segment (e.g.
# "...-302894593.html").
WEAK_ARTICLE_PATH_WORDS = ("news", "press-release", "pressrelease", "release", "story", "post", "media")
NAV_PATH_BLOCKLIST = (
    "privacy", "terms", "cookie", "contact", "about-us", "about", "careers",
    "jobs", "login", "signin", "sign-in", "signup", "sign-up", "subscribe",
    "newsletter", "advertise", "sitemap", "search", "tag/", "tags/",
    "category/", "author/", "page/", "archive",
    # GlobeNewswire/PR Newswire/Newswire Canada name their CATEGORY pages
    # with long, descriptive, hyphenated slugs too (e.g.
    # "automotive-transportation-latest-news-list"), so a long slug alone
    # doesn't distinguish a category page from a real article the way it
    # does on other sites. These two patterns are specific enough to their
    # category-page naming convention to block safely.
    "-list/", "-list", "latest-news",
    # EIN Presswire's own site-nav/marketing pages -- these don't contain
    # any of the article keywords above, so they were slipping through the
    # generic fallback check instead (long link text + path ending in a
    # segment, which is true for "Trustpilot Reviews", "Pricing", etc.).
    "why-us", "trustpilot", "pricing", "guide-pdf", "knowledge-base",
    "channel/", "newsroom/einpresswire", "press-release-distribution-guide",
    "ai/press-release-generator",
    # GlobeNewswire's specific category/sub-category hub paths. NOTE: these
    # are already rejected by the has_long_numeric_id check in
    # is_article_link() below (hub pages have no numeric article ID in the
    # URL) -- listed explicitly here too for clarity/defense-in-depth, but
    # this does NOT put any new real Globe Newswire articles in the output.
    # Its listing page only links to these category hubs, never to
    # individual press releases directly, so there's nothing beyond this to
    # scrape without a second crawl hop (visit each hub, then find articles
    # on it). See note in collect_source().
    "/news/consumer-products-services", "/news/banks-financial-services",
    "/news/technology-telecom", "/news/industrials-utilities",
    "/news/heathcare/",
    # Fierce Pharma's gated whitepaper/webinar pages -- real pages, but not
    # news articles, and news-fetch correctly finds no article body on them.
    "resource/",
)
MAX_SCRAPE_CANDIDATES = 20

# Set by main() from --timeout so every function below shares one value
# without threading an argument through every call.
TIMEOUT = DEFAULT_TIMEOUT


# ---------------------------------------------------------------------------
# Page fetching: Playwright FIRST, plain requests as fallback
# ---------------------------------------------------------------------------
def fetch_with_playwright(url: str, timeout_ms: int | None = None) -> str | None:
    """Renders the page with a real headless browser. This is now the FIRST
    thing tried for any page, not a last resort -- a real browser
    fingerprint gets past a good chunk of basic bot-blocking that a raw
    HTTP request trips immediately. Returns None (never raises) if it
    fails, so callers can cleanly fall through to the next method."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return None
    timeout_ms = timeout_ms or (TIMEOUT * 1000)
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


def fetch_with_requests(url: str) -> str:
    """Plain HTTP fetch with retries. Raises requests.RequestException if
    every retry fails. This is now the FALLBACK, used only after Playwright
    has already failed."""
    last_error = None
    for attempt in range(3):
        try:
            response = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
            response.raise_for_status()
            return response.text
        except requests.HTTPError as error:
            if error.response is not None and error.response.status_code in {401, 403, 404, 410}:
                raise
            last_error = error
        except requests.RequestException as error:
            last_error = error
        if attempt < 2:
            time.sleep(2 ** attempt)
    raise last_error


def fetch_page(url: str) -> str:
    """Playwright first; plain requests fallback if Playwright fails.
    Raises requests.RequestException only if BOTH fail. Used for every page
    fetch (source listing pages AND individual article pages)."""
    html = fetch_with_playwright(url)
    if html is not None:
        return html
    return fetch_with_requests(url)  # raises on failure, caller handles it


# ---------------------------------------------------------------------------
# Datetime parsing
# ---------------------------------------------------------------------------
# Non-standard date formats seen in the wild that neither ISO-format nor
# RFC-2822 parsing handles, keyed by a short description for readability.
# Add more patterns here as new sources turn up new formats in the
# Failures sheet (look for "unparseable" rows with a stage of
# article_scrape or source_fetch).
EXTRA_DATE_FORMATS = (
    "%a, %m/%d/%Y - %H:%M",  # FDA.gov's Drupal-generated dates, e.g.
                              # "Tue, 10/01/2024 - 07:43"
)


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
                parsed = None
                for fmt in EXTRA_DATE_FORMATS:
                    try:
                        parsed = datetime.strptime(text, fmt)
                        break
                    except ValueError:
                        continue
                if parsed is None:
                    return None
    else:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


# ---------------------------------------------------------------------------
# Feed discovery + parsing
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
    """Tries common feed paths directly via plain requests, WITHOUT needing
    the page HTML at all. This is what gets tried when Playwright AND a
    plain fetch of the page itself have both already failed."""
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
# HTML scrape fallback
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
    if any(word in path_lower for word in STRONG_ARTICLE_PATH_WORDS):
        return True
    if any(word in path_lower for word in WEAK_ARTICLE_PATH_WORDS):
        last_segment = parsed_link.path.rstrip("/").rsplit("/", 1)[-1]
        has_long_numeric_id = bool(re.search(r"\d{5,}", last_segment))
        # A weak word (news/release/press-release/etc.) without a numeric
        # article ID is a category/hub page on these sites (GlobeNewswire,
        # PR Newswire, Newswire Canada, EIN Presswire), not an individual
        # article -- reject it outright instead of falling through to the
        # generic heuristic below, which was previously letting these back
        # in whenever the link text happened to be long (e.g. a category
        # name like "Consumer Products & Services").
        return has_long_numeric_id
    if path_lower.endswith(".pdf"):
        return False
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


# ---------------------------------------------------------------------------
# Per-source collection: Playwright -> requests -> RSS-guess, in that order
# ---------------------------------------------------------------------------
def collect_source(source: dict, cutoff: datetime) -> tuple[list[dict], list[dict]]:
    """Returns (articles, failures).

    Order of attempts for the source's own page:
      1. Playwright render of the page.
      2. Plain `requests` fetch of the page (if Playwright failed).
      3. If BOTH of those failed: guess common RSS feed URLs directly via
         plain requests (no page HTML needed) and try those.
      4. Only if step 3 also fails/finds nothing do we give up on this
         source entirely (raised to collect_source_safe as one failure).

    Once we DO have the page HTML (from step 1 or 2), the existing
    feed-vs-scrape logic runs as before: look for an RSS feed first, fall
    back to scraping article links out of the page.
    """
    failures: list[dict] = []
    page_url = source["url"]
    fetch_mode = source.get("fetch_mode", "auto").lower()
    feed_url = source.get("feed_url")

    html = None
    page_fetch_error = None
    try:
        html = fetch_page(page_url)
    except requests.RequestException as error:
        page_fetch_error = error

    if html is None:
        # Both Playwright and plain requests failed to load the page itself.
        # Last resort: try guessing an RSS feed URL directly.
        if fetch_mode == "scrape":
            raise page_fetch_error  # scrape-only source with no page to scrape -- true failure

        guessed_feed = feed_url or guess_feed_url(page_url)
        if not guessed_feed:
            raise page_fetch_error  # nothing left to try -- true failure

        try:
            articles = collect_from_feed(source, guessed_feed)
        except Exception as feed_error:
            raise RuntimeError(
                f"page unreachable ({page_fetch_error}); RSS fallback also failed ({feed_error})"
            ) from feed_error

        if not articles:
            raise RuntimeError(
                f"page unreachable ({page_fetch_error}); RSS fallback found the feed but it had no entries"
            )
        return [a for a in articles if parse_datetime(a["published"]) >= cutoff], failures

    # We have the page HTML (via Playwright or requests) -- normal flow.
    if fetch_mode != "scrape" and not feed_url:
        feed_url = discover_feed_url(page_url, html) or guess_feed_url(page_url)
    if fetch_mode != "scrape" and feed_url:
        try:
            articles = collect_from_feed(source, feed_url)
            if articles:
                return [a for a in articles if parse_datetime(a["published"]) >= cutoff], failures
        except Exception as error:
            failures.append({
                "source": source["name"], "url": feed_url, "stage": "source_fetch",
                "reason": f"feed parse failed: {error}",
            })
            if fetch_mode == "feed":
                return [], failures
    if fetch_mode == "feed":
        return [], failures

    articles = []
    for article_url in discover_article_links(page_url, html):
        try:
            html_article = fetch_page(article_url)
            article = extract_article_stub(source["name"], article_url, html_article)
            if article is None:
                failures.append({
                    "source": source["name"], "url": article_url, "stage": "article_scrape",
                    "reason": "page fetched but no usable title/published date found (meta tags missing)",
                })
                continue
            published = parse_datetime(article["published"])
            if not published:
                failures.append({
                    "source": source["name"], "url": article_url, "stage": "article_scrape",
                    "reason": f"published date found but unparseable: {article['published']!r}",
                })
                continue
            if published < cutoff:
                continue  # not a failure, just outside the time window
            article["published"] = published.isoformat()
            articles.append(article)
        except requests.RequestException as error:
            failures.append({
                "source": source["name"], "url": article_url, "stage": "article_scrape",
                "reason": f"fetch failed (Playwright and requests both failed): {error}",
            })
    return articles, failures


def collect_source_safe(source: dict, cutoff: datetime) -> tuple[str, list[dict], list[dict]]:
    """Returns (source_name, articles, failures). A whole-source failure
    (page + requests + RSS-guess all failed) is recorded as a single
    failure row rather than raising."""
    try:
        articles, failures = collect_source(source, cutoff)
        return source["name"], articles, failures
    except Exception as error:
        return source.get("name", "unnamed source"), [], [{
            "source": source.get("name", "unnamed source"),
            "url": source.get("url"),
            "stage": "source_fetch",
            "reason": str(error),
        }]


# ---------------------------------------------------------------------------
# Content extraction via news-fetch, with a manual BeautifulSoup fallback
# ---------------------------------------------------------------------------
# Minimum size for the BS4 fallback to accept a page as "real content" rather
# than a paywall teaser. CAUTION: this fallback is a deliberate tradeoff --
# the previous pipeline (trafilatura) was already grabbing content from
# paywalled sources like Endpoint News, and it turned out to be truncated
# teaser text, not the real article (part of why news-fetch replaced it,
# since news-fetch correctly refuses low-confidence/thin content). A raw
# <p>-tag scrape can reintroduce that same problem. To limit the risk:
#   1. It only runs when news-fetch has already given up.
#   2. It requires a reasonably long combined paragraph text before
#      accepting the result (short teaser pages won't clear this bar).
#   3. Every article recovered this way is tagged "content_source": "bs4_fallback"
#      in the output so it's visually distinguishable from a normal
#      news-fetch extraction -- treat these as "verify before trusting."
FALLBACK_MIN_CONTENT_CHARS = 600


def fallback_extract_content(article_url: str) -> str | None:
    """Last-resort manual extraction when news-fetch returns nothing usable.
    Returns None (never raises) if the page can't be fetched or doesn't
    clear the minimum-length bar."""
    try:
        html = fetch_page(article_url)
    except requests.RequestException:
        return None
    soup = BeautifulSoup(html, "html.parser")
    paragraphs = soup.find_all("p")
    text = "\n\n".join(
        p.get_text(" ", strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 40
    )
    if len(text.strip()) < FALLBACK_MIN_CONTENT_CHARS:
        return None
    return text.strip()


def add_full_content(articles: list[dict], max_workers: int) -> tuple[list[dict], list[dict]]:
    """Returns (successful_articles, content_failures). An article only
    counts as successful if news-fetch returned non-empty body text, OR
    (failing that) the BS4 fallback found a long-enough block of paragraph
    text. Fallback successes are tagged so they're easy to spot/audit."""
    urls = [a["url"] for a in articles]
    fetched = fetch_many(urls, max_workers=max_workers)  # list[Article | None], same order as urls

    successes = []
    failures = []
    for stub, article in zip(articles, fetched):
        content = article.text.strip() if article is not None and article.text else None
        if content:
            stub["content"] = content
            stub["content_source"] = "news_fetch"
            stub["authors"] = ", ".join(article.authors) if article.authors else None
            if article.published_at and not stub.get("published"):
                stub["published"] = article.published_at.isoformat() if hasattr(article.published_at, "isoformat") else str(article.published_at)
            successes.append(stub)
            continue

        # news-fetch gave us nothing usable -- try the manual fallback.
        fallback_content = fallback_extract_content(stub["url"])
        if fallback_content:
            stub["content"] = fallback_content
            stub["content_source"] = "bs4_fallback"  # flagged for manual spot-check
            stub["authors"] = None
            successes.append(stub)
            continue

        reason = (
            "news-fetch returned no result (fetch failed or below confidence threshold)"
            if article is None else
            "news-fetch found the page but extracted no body text"
        )
        failures.append({
            "source": stub["source"], "url": stub["url"], "stage": "content_extraction",
            "reason": f"{reason}; BS4 fallback also found no usable content (page may be paywalled/gated)",
        })
    return successes, failures


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    global TIMEOUT

    parser = argparse.ArgumentParser(description="Fetch current news + full content from a curated source list.")
    parser.add_argument("--hours", type=float, default=24)
    parser.add_argument("--workers", type=int, default=8, help="Concurrent source fetches.")
    parser.add_argument("--content-workers", type=int, default=8, help="Concurrent article-content fetches (news-fetch).")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="Per-request timeout in seconds (default: 30).")
    parser.add_argument("--output", type=Path, default=OUTPUT_FILE)
    args = parser.parse_args()
    TIMEOUT = args.timeout

    sources = yaml.safe_load(SOURCES_FILE.read_text(encoding="utf-8")).get("sources", [])
    cutoff = datetime.now(timezone.utc) - timedelta(hours=args.hours)

    print(f"Collecting from {len(sources)} sources (last {args.hours}h, timeout={TIMEOUT}s, Playwright-first)...")
    all_articles = []
    all_failures = []
    worker_count = max(1, min(args.workers, 16))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(collect_source_safe, source, cutoff) for source in sources]
        for future in as_completed(futures):
            source_name, articles, failures = future.result()
            if failures and not articles:
                print(f"  {source_name}: failed ({failures[0]['reason']})" if len(failures) == 1 and failures[0]["stage"] == "source_fetch" else f"  {source_name}: 0 article(s), {len(failures)} failure(s)")
            else:
                print(f"  {source_name}: {len(articles)} article(s)" + (f", {len(failures)} failure(s)" if failures else ""))
            all_articles.extend(articles)
            all_failures.extend(failures)

    unique_articles = list({a["url"]: a for a in all_articles if a.get("url")}.values())
    print(f"\n{len(unique_articles)} unique articles collected. Fetching full content via news-fetch "
          f"(falling back to manual extraction when news-fetch comes up empty)...")

    content_worker_count = max(1, min(args.content_workers, 16))
    successful_articles, content_failures = add_full_content(unique_articles, content_worker_count)
    all_failures.extend(content_failures)

    print(f"{len(successful_articles)}/{len(unique_articles)} articles got usable full content "
          f"({len(content_failures)} failed content extraction).")
    print(f"Total failures across all stages: {len(all_failures)}")

    successful_articles.sort(key=lambda a: a["published"], reverse=True)
    success_rows = [
        {
            "source": a["source"],
            "title": a["title"],
            "url": a["url"],
            "description": a.get("description"),
            "authors": a.get("authors"),
            "content": a.get("content"),
            "content_source": a.get("content_source", "news_fetch"),  # "bs4_fallback" = spot-check this one
            "published": a["published"],
            "fetched_at": a["fetched_at"],
        }
        for a in successful_articles
    ]
    failure_rows = [
        {
            "source": f.get("source"),
            "url": f.get("url"),
            "stage": f.get("stage"),
            "reason": f.get("reason"),
            "logged_at": datetime.now(timezone.utc).isoformat(),
        }
        for f in all_failures
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    success_df = pd.DataFrame(success_rows)
    failure_df = pd.DataFrame(failure_rows)
    with pd.ExcelWriter(args.output, engine="openpyxl") as writer:
        success_df.to_excel(writer, sheet_name="Articles", index=False)
        failure_df.to_excel(writer, sheet_name="Failures", index=False)

    print(f"\nSaved {len(success_rows)} successful article(s) to Sheet 'Articles'")
    print(f"Saved {len(failure_rows)} failure(s) to Sheet 'Failures'")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()