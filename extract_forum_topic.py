#!/usr/bin/env python3
"""
Extract structured data from Raspberry Pi forum topic pages.

Each file in forum_pages/ is a JSON object with a "body" key containing the HTML
of a topic page. This script extracts:
- Headline (h2.topic-title)
- Canonical topic URL (https://forums.raspberrypi.com/viewtopic.php?t=...)
- Forum name and id (e.g. "Beginners", "Troubleshooting")
- Per-post: content, time, username, profile link, profile posts count, joined date.

Usage:
  python extract_forum_topic.py forum_pages/topic_106441.html
  python extract_forum_topic.py forum_pages/  # process all .html in directory
  python extract_forum_topic.py forum_pages/ --output topics.json
"""

import argparse
import json
import os
import re
import sys

from bs4 import BeautifulSoup


def extract_headline(soup: BeautifulSoup) -> str | None:
    """Extract text from first h2 with class topic-title."""
    h2 = soup.find('h2', class_=re.compile(r'topic-title'))
    if not h2:
        return None
    # Get text; if there's an inner <a>, use its text
    link = h2.find('a')
    if link:
        text = link.get_text(strip=True)
    else:
        text = h2.get_text(strip=True)
    return text or None


def extract_canonical_url(soup: BeautifulSoup) -> str | None:
    """Extract href from <link rel="canonical" href="..."> in the document head."""
    link = soup.find('link', rel='canonical')
    if not link or not link.get('href'):
        return None
    return link['href'].strip() or None


def extract_forum(soup: BeautifulSoup) -> dict:
    """Extract forum id and name (e.g. Beginners, Troubleshooting).
    Returns {'forum_id': str|None, 'forum_name': str|None}.
    """
    result = {'forum_id': None, 'forum_name': None}
    # From feed link: "Feed - Forum - NAME" href=".../feed/forum/ID"
    for link in soup.find_all('link', rel='alternate', type='application/atom+xml'):
        title = link.get('title') or ''
        href = link.get('href') or ''
        if 'Feed - Forum - ' in title:
            m = re.search(r'feed/forum/(\d+)', href)
            if m:
                result['forum_id'] = m.group(1)
                result['forum_name'] = title.replace('Feed - Forum - ', '').strip()
                return result
    # Fallback: breadcrumb - last data-forum-id with span itemprop="name"
    for elem in soup.find_all(attrs={'data-forum-id': True}):
        forum_id = elem.get('data-forum-id')
        span = elem.find('span', itemprop='name')
        if span:
            result['forum_id'] = forum_id
            result['forum_name'] = span.get_text(strip=True)
    return result


def extract_post_blocks(soup: BeautifulSoup) -> list:
    """Return list of post block elements (div with id p123 and class containing post)."""
    post_divs = []
    for div in soup.find_all('div', id=re.compile(r'^p\d+$')):
        classes = div.get('class') or []
        if any('post' in c for c in classes):
            post_divs.append(div)
    return post_divs


def _content_inner_html(content_div) -> str | None:
    """Get inner HTML of a content div as string."""
    if not content_div:
        return None
    return content_div.decode_contents().strip() or None


def extract_post_data(post_block) -> dict:
    """From a single post block (BeautifulSoup Tag), extract all requested fields."""
    out = {
        'content': None,
        'time': None,
        'username': None,
        'user_profile_link': None,
        'profile_posts': None,
        'profile_rank_text': None,
        'profile_rank_image': None,
        'joined': None,
    }
    # Username: <a ... class="username">USER</a>
    username_a = post_block.find('a', class_=re.compile(r'username'))
    if username_a:
        out['username'] = username_a.get_text(strip=True)
        out['user_profile_link'] = username_a.get('href', '').strip() or None

    # Profile posts: <dd class="profile-posts">...<a ...>N</a></dd>
    pp_dd = post_block.find('dd', class_=re.compile(r'profile-posts'))
    if pp_dd:
        pp_a = pp_dd.find('a')
        if pp_a:
            out['profile_posts'] = pp_a.get_text(strip=True)

    # Joined: <dd class="profile-joined"><strong>Joined:</strong> DATE</dd>
    j_dd = post_block.find('dd', class_=re.compile(r'profile-joined'))
    if j_dd:
        text = j_dd.get_text()
        if 'Joined:' in text:
            out['joined'] = text.split('Joined:', 1)[-1].strip()

    # Time (post time): <p class="author"><a ...>DATE TIME</a>
    author_p = post_block.find('p', class_=re.compile(r'author'))
    if author_p:
        author_a = author_p.find('a')
        if author_a:
            out['time'] = author_a.get_text(strip=True)

    # <dd class="profile-rank">...</dd>: text and first <img> src
    pr_dd = post_block.find('dd', class_=re.compile(r'profile-rank'))
    if pr_dd:
        out['profile_rank_text'] = pr_dd.get_text(strip=True) or None
        img = pr_dd.find('img', src=True)
        if img:
            out['profile_rank_image'] = img['src'].strip()

    # Content: first <div class="content">...</div>
    content_div = post_block.find('div', class_=re.compile(r'^content$'))
    out['content'] = _content_inner_html(content_div)

    return out


def extract_topic(filepath: str) -> dict | None:
    """Load one topic file (JSON with body key) and return extracted data."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        return {'error': str(e), 'file': filepath}

    body = data.get('body')
    if not body or not isinstance(body, str):
        return {'error': 'Missing or invalid "body"', 'file': filepath}

    soup = BeautifulSoup(body, 'html.parser')

    headline = extract_headline(soup)
    post_blocks = extract_post_blocks(soup)
    posts = [extract_post_data(block) for block in post_blocks]

    forum = extract_forum(soup)

    topic_id = os.path.basename(filepath)
    if topic_id.endswith('.html'):
        topic_id = topic_id[:-5]  # topic_106441

    # Canonical URL from <link rel="canonical" href="...">, fallback to filename-derived
    canonical_url = extract_canonical_url(soup)
    if canonical_url is None:
        numeric_id = topic_id.replace('topic_', '', 1) if topic_id.startswith('topic_') else topic_id
        canonical_url = f"https://forums.raspberrypi.com/viewtopic.php?t={numeric_id}" if numeric_id.isdigit() else None

    return {
        'topic_id': topic_id,
        'canonical_url': canonical_url,
        'forum_id': forum['forum_id'],
        'forum_name': forum['forum_name'],
        'headline': headline,
        'posts': posts,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Extract headline and post data from forum topic HTML (JSON) files.',
    )
    parser.add_argument(
        'path',
        nargs='+',
        help='Path(s) to .html file(s) or a directory containing them.',
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force extraction even if the output file already exists.',
    )
    args = parser.parse_args()

    files = []
    for p in args.path:
        if os.path.isfile(p) and p.endswith('.html'):
            files.append(p)
        elif os.path.isdir(p):
            for name in sorted(os.listdir(p)):
                if name.endswith('.html'):
                    files.append(os.path.join(p, name))

    if not files:
        print('No .html files found.', file=sys.stderr)
        sys.exit(1)

    for index, f in enumerate(files):
        output_file = f.replace('.html', '.json')
        if os.path.exists(output_file) and not args.force:
            print(f'Skipping {f} because {output_file} already exists')
            continue
        if index > 0 and index % 100 == 0:
            print(f'Extracted {index}/{len(files)} topics')
        topic = extract_topic(f)
        if topic:
            with open(output_file, 'w', encoding='utf-8') as out:
                json.dump(topic, out, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    main()
