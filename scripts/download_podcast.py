import feedparser
import requests
import os
import re

def download_podcast(rss_url, save_dir="downloads"):
    # 1. Parse the RSS feed
    feed = feedparser.parse(rss_url)
    
    if not feed.entries:
        print("No episodes found in this feed.")
        return

    # 2. Get the latest episode
    latest_episode = feed.entries[0]
    title = latest_episode.title
    
    # 3. Find the audio URL (found in 'enclosures')
    audio_url = None
    for link in latest_episode.links:
        if link.get('type') == 'audio/mpeg' or 'enclosure' in link.get('rel', ''):
            audio_url = link.get('href')
            break
    
    if not audio_url:
        print(f"Could not find an audio file for: {title}")
        return

    # 4. Clean the title for a safe filename
    clean_title = re.sub(r'[^\w\s-]', '', title).strip().replace(' ', '_')
    filename = f"{clean_title}.mp3"
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    save_path = os.path.join(save_dir, filename)

    # 5. Download the file
    print(f"Downloading: {title}...")

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    response = requests.get(audio_url, stream=True, headers=headers)
    
    if response.status_code == 200:
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        print(f"Successfully saved to: {save_path}")
    else:
        print(f"Failed to download. Status code: {response.status_code}")

if __name__ == "__main__":
    # Example: Test with a valid podcast RSS URL
    PODCAST_RSS = "https://rss.buzzsprout.com/1426696.rss" # Example: The Daily
    download_podcast(PODCAST_RSS)