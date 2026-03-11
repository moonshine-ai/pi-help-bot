import requests
import os
from concurrent.futures import ThreadPoolExecutor
import threading

MAX_WORKERS = 50

pages_fetched = 0
pages_fetched_lock = threading.Lock()

def is_valid_topic_url(topic_url):
    if 'p=' not in topic_url and 't=' not in topic_url:
        return False

    if 'p=' in topic_url:
        topic_id = topic_url.split('p=')[1].split('&')[0]
    else:
        topic_id = topic_url.split('t=')[1].split('&')[0]
    filename = f"forum_pages/topic_{topic_id}.html"

    if os.path.exists(filename):
        return False

    if not os.path.exists(os.path.dirname(filename)):
        return False

    return True

def fetch_topic(topic_url):
    global pages_fetched
    if not is_valid_topic_url(topic_url):
        return

    response = requests.get(
        url='https://app.scrapingbee.com/api/v1',
        params={
            'api_key': os.getenv('SCRAPINGBEE_API_KEY'),
            'url': topic_url,
            'json_response': 'true',
        },
    )
    if response.status_code != 200:
        print('Error fetching topic: ', response.status_code)
        return
    with pages_fetched_lock:
        pages_fetched += 1
        if (pages_fetched % 10 == 0):
            print(f'Fetched {pages_fetched} pages')

    if 'p=' in topic_url:
        topic_id = topic_url.split('p=')[1].split('&')[0]
    else:
        topic_id = topic_url.split('t=')[1].split('&')[0]
    filename = f"forum_pages/topic_{topic_id}.html"

    with open(filename, 'w') as f:
        f.write(response.content.decode('utf-8'))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Fetch a topic from the Raspberry Pi forums')
    parser.add_argument("--url-file", type=str, help='The file containing the URLs to fetch')
    args = parser.parse_args()
    urls = []
    with open(args.url_file, 'r') as f:
        for line in f:
            url = line.strip()
            if url and is_valid_topic_url(url):
                urls.append(url)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        list(executor.map(fetch_topic, urls))
    print(f'Done. Fetched {pages_fetched} pages total.')
