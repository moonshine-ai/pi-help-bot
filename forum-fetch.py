import cdx_toolkit
import json
import logging
from urllib.parse import urlparse, parse_qs
from bs4 import BeautifulSoup

# Uncomment to see which CDX URL is being fetched (helps debug hangs):
# logging.basicConfig(level=logging.INFO)

SUBFORUMS = {
    "91": "Beginners",
    "28": "Troubleshooting",
}

# Crawls to try (newest first). Latest crawl often has only robots.txt for forums;
# older crawls may have actual viewforum.php pages.
CRAWLS_TO_TRY = [
    "CC-MAIN-2026-08",
    "CC-MAIN-2026-04",
    "CC-MAIN-2025-51",
    "CC-MAIN-2025-47",
    "CC-MAIN-2025-43",
    "CC-MAIN-2025-38",
    "CC-MAIN-2025-33",
    "CC-MAIN-2025-30",
]

def fetch_rpi_subforum_titles(
    output_file="rpi_topics.json",
    limit=None,
    debug=False,
):
    """
    Fetch topic titles from RPi forum listing pages via Common Crawl index.

    Args:
        output_file: Output JSONL file path.
        limit: Max CDX records to fetch per subforum (default None = no limit).
               Use e.g. limit=500 to avoid slow/unbounded index queries.
        debug: If True, print why records are skipped (no CDX hits, non-200, no topictitle, etc.).
    """
    # Try multiple crawls (newest first) until we get CDX records; latest crawl
    # often has only robots.txt for forums.raspberrypi.com, no viewforum.php.
    cdx = cdx_toolkit.CDXFetcher(source="ia")
    total = 0

    with open(output_file, "w") as f:
        # Target only listing pages for this subforum.
        # These look like viewforum.php?f=91&start=0, ?f=91&start=25, etc.
        # url_pattern = f"forums.raspberrypi.com/viewforum.php?f={subforum_id}*"
        url_pattern = "forums.raspberrypi.com/viewtopic.php*"
        print(f"\nFetching listing pages for '{url_pattern}'")
        if limit is not None:
            print(f"  (limit={limit} CDX records)")

        cdx_count = 0
        kwargs = {}
        if limit is not None:
            kwargs["limit"] = int(limit)
        for obj in cdx.iter(url_pattern, **kwargs):
            if debug:
                print(f"  [debug] CDX #{cdx_count}: url={obj.get('url', '')}")

            cdx_count += 1
            print(dir(obj))
            print(obj.data)
            f.write(json.dumps(obj.data) + "\n")

        if debug or cdx_count == 0:
            print(f"  CDX records seen: {cdx_count}")

    print(f"\nDone. {total} unique topics written to {output_file}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Fetch RPi forum topic titles from Common Crawl index.")
    p.add_argument("--limit", type=int, default=None,
                   help="Max CDX records per subforum (e.g. 500). Use if the first request hangs.")
    p.add_argument("-v", "--verbose", action="store_true", help="Log CDX requests (shows endpoint URLs).")
    p.add_argument("--debug", action="store_true", help="Print why CDX records are skipped (no hits, non-200, no topictitle).")
    args = p.parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    fetch_rpi_subforum_titles(limit=args.limit, debug=args.debug)