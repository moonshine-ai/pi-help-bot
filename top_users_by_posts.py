#!/usr/bin/env python3
"""
Find the top N users by post count across all forum topic JSON files.

Reads every .json file in forum_pages/, counts posts per username (from the
"posts" array in each topic), aggregates across all files, and prints the
top 100 users by total post count.

With --engineers-only: only counts users with the Engineer rank in the ranking.

With --doc-links-tsv: writes a TSV of documentation links from all posts, with a
column marking whether the link appeared in at least one engineer post in that topic.

Usage:
  python top_users_by_posts.py
  python top_users_by_posts.py --engineers-only
  python top_users_by_posts.py --doc-links-tsv links.tsv   # all users + engineer column
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from urllib.parse import urlparse, urlunparse

FORUM_BASE = "https://forums.raspberrypi.com"
RANK_IMAGE_ENGINEER = "./images/ranks/Forum-Banners_Engineer.png"
DOC_BASE = "https://www.raspberrypi.com/documentation"
HREF_RE = re.compile(r'href\s*=\s*["\']([^"\']+)["\']', re.I)


def make_profile_url(relative_link: str | None) -> str:
    """Convert relative profile link to absolute URL; strip sid for stable link."""
    if not relative_link or not isinstance(relative_link, str):
        return ""
    # Remove leading ./
    path = relative_link.strip().removeprefix("./")
    # Extract u= for a stable URL without session id
    match = re.search(r"[?&]u=(\d+)", path.replace("&amp;", "&"))
    if match:
        return f"{FORUM_BASE}/memberlist.php?mode=viewprofile&u={match.group(1)}"
    return f"{FORUM_BASE}/{path.replace('&amp;', '&')}" if path else ""


def extract_doc_links(content: str) -> list[str]:
    """Extract and normalize documentation URLs from HTML content.
    Accepts both www.raspberrypi.com/documentation and www.raspberrypi.org/documentation.
    """
    if not content or not isinstance(content, str):
        return []
    seen: set[str] = set()
    result: list[str] = []
    for m in HREF_RE.finditer(content):
        raw = m.group(1).replace("&amp;", "&").strip()
        if not raw:
            continue
        # Resolve relative URLs (assume .com documentation)
        if raw.startswith("/"):
            url = f"https://www.raspberrypi.com{raw}"
        elif not raw.startswith(("http://", "https://")):
            continue
        else:
            url = raw
        parsed = urlparse(url)
        # Only documentation on raspberrypi.com or .org (with or without www)
        netloc = (parsed.netloc or "").lower()
        if netloc not in (
            "www.raspberrypi.com",
            "www.raspberrypi.org",
            "raspberrypi.com",
            "raspberrypi.org",
        ):
            continue
        if not (parsed.path or "").startswith("/documentation"):
            continue
        # Normalize host to www so the same doc URL dedupes across redirects
        host = "www.raspberrypi.com" if netloc.endswith("raspberrypi.com") else "www.raspberrypi.org"
        canonical = urlunparse(("https", host, parsed.path, "", parsed.query, parsed.fragment))
        if canonical not in seen:
            seen.add(canonical)
            result.append(canonical)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find top users by post count from forum_pages JSON files."
    )
    parser.add_argument(
        "--forum-pages",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "forum_pages",
        help="Directory containing topic_*.json files (default: forum_pages in repo root)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Number of top users to show (default: 100)",
    )
    parser.add_argument(
        "--engineers-only",
        action="store_true",
        help="Only count posts from users with profile_rank_image Engineer banner",
    )
    parser.add_argument(
        "--doc-links-tsv",
        type=Path,
        nargs="?",
        const=Path("doc_links.tsv"),
        default=None,
        help="Write TSV of doc links from all posts: topic URL, title, doc link, post_from_engineer (yes/no). "
        "Default file if flag given with no path: doc_links.tsv",
    )
    args = parser.parse_args()

    forum_dir = args.forum_pages
    if not forum_dir.is_dir():
        raise SystemExit(f"Not a directory: {forum_dir}")

    post_counts: dict[str, int] = {}
    profile_links: dict[str, str] = {}
    # (canonical_url, doc_link) -> (headline, from_engineer) — engineer True if any post with that link is engineer
    doc_link_info: dict[tuple[str, str], tuple[str, bool]] = {}
    files_processed = 0
    total_posts = 0

    for path in sorted(forum_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            print(f"Warning: skip {path.name}: {e}", file=sys.stderr)
            continue

        posts = data.get("posts")
        if not isinstance(posts, list):
            continue

        canonical_url = data.get("canonical_url") or ""
        headline = (data.get("headline") or "").replace("\t", " ").replace("\n", " ").strip()

        for post in posts:
            if not isinstance(post, dict):
                continue
            username = post.get("username")
            if username is None or not isinstance(username, str):
                continue
            username = username.strip()
            if username:
                if args.engineers_only:
                    if post.get("profile_rank_image") != RANK_IMAGE_ENGINEER:
                        continue
                post_counts[username] = post_counts.get(username, 0) + 1
                total_posts += 1
                if username not in profile_links:
                    profile_links[username] = make_profile_url(post.get("user_profile_link"))
                # Doc links from all posts when TSV output requested
                if args.doc_links_tsv is not None and canonical_url:
                    is_engineer = post.get("profile_rank_image") == RANK_IMAGE_ENGINEER
                    for doc_link in extract_doc_links(post.get("content")):
                        key = (canonical_url, doc_link)
                        if key not in doc_link_info:
                            doc_link_info[key] = (headline, is_engineer)
                        else:
                            h, eng = doc_link_info[key]
                            doc_link_info[key] = (h, eng or is_engineer)

        files_processed += 1

    # Sort by count descending, then by username for ties
    sorted_users = sorted(
        post_counts.items(),
        key=lambda x: (-x[1], x[0].lower()),
    )
    top = sorted_users[: args.limit]

    print(f"Processed {files_processed} JSON files, {total_posts} total posts, {len(post_counts)} unique users.\n")
    if args.engineers_only:
        print("(Filtered to users with Engineer rank only.)\n")
    if args.doc_links_tsv is not None:
        tsv_path = args.doc_links_tsv
        if doc_link_info:
            rows = sorted(
                (
                    (url, title, link, "yes" if eng else "no")
                    for (url, link), (title, eng) in doc_link_info.items()
                ),
                key=lambda r: (r[0], r[2]),
            )
            with open(tsv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
                w.writerow(
                    ["canonical_url", "topic_title", "documentation_link", "post_from_engineer"]
                )
                w.writerows(rows)
            print(f"Wrote {len(rows)} documentation links to {tsv_path}\n")
        else:
            print("No documentation links found in posts.\n")
    print(f"Top {len(top)} users by post count:\n")
    print(f"{'Rank':<6} {'Posts':<8} Username")
    print("-" * 60)
    for rank, (username, count) in enumerate(top, start=1):
        url = profile_links.get(username, "")
        print(f"{rank:<6} {count:<8} {username}")
        if url:
            print(f"       {url}")


if __name__ == "__main__":
    main()
