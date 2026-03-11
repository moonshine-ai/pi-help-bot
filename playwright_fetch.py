#!/usr/bin/env python3
"""
Fetch a single URL headlessly with Playwright and save HTML, optional HAR and screenshot.
Usage:
  python playwright_fetch.py <url> <html_path> [--har har_path] [--screenshot screenshot_path]
"""
import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="Fetch URL with Playwright (headless) and save HTML/HAR/screenshot.")
    parser.add_argument("url", help="URL to fetch")
    parser.add_argument("html_path", help="Output path for the HTML file")
    parser.add_argument("--har", metavar="path", default=None, help="Optional path to save HAR")
    parser.add_argument("--screenshot", metavar="path", default=None, help="Optional path to save PNG screenshot")
    parser.add_argument("--wait-until", default="load", help="Wait until (e.g. domcontentloaded, load, networkidle)")
    args = parser.parse_args()

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("playwright not installed. Run: pip install playwright && playwright install chromium", file=sys.stderr)
        sys.exit(1)

    har_ctx = {}
    if args.har:
        har_ctx["record_har_path"] = args.har

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(**har_ctx)
        page = context.new_page()
        try:
            page.goto(args.url, wait_until=args.wait_until, timeout=60000)
            html = page.content()
            with open(args.html_path, "w", encoding="utf-8") as f:
                f.write(html)
            if args.screenshot:
                page.screenshot(path=args.screenshot)
        finally:
            context.close()
            browser.close()

    print(f"Saved HTML to {args.html_path}" + (f", HAR to {args.har}" if args.har else "") + (f", screenshot to {args.screenshot}" if args.screenshot else ""))


if __name__ == "__main__":
    main()
