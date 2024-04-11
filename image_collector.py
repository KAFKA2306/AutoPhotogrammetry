import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urlencode
from concurrent.futures import ThreadPoolExecutor, as_completed
from flickrapi import FlickrAPI
from fivehundredpx import FiveHundredPX

def download_image(img_url, output_dir):
    try:
        response = requests.get(img_url)
        file_name = os.path.basename(urlparse(img_url).path)
        file_path = os.path.join(output_dir, file_name)
        with open(file_path, 'wb') as f:
            f.write(response.content)
        return img_url, True
    except Exception as e:
        print(f"Error downloading {img_url}: {e}")
        return img_url, False

def collect_images_from_web(keywords, num_images, output_dir, max_workers=10):
    search_engines = [
        f"https://www.google.com/search?q={urlencode(keywords)}&tbm=isch",
        f"https://www.bing.com/images/search?q={urlencode(keywords)}",
    ]
    
    image_urls = []
    for url in search_engines:
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            for img in soup.find_all('img'):
                img_url = img.get('src')
                if img_url and not img_url.startswith('data:'):
                    image_urls.append(img_url)
                    if len(image_urls) >= num_images:
                        break
        except Exception as e:
            print(f"Error fetching {url}: {e}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    downloaded_images = []
    failed_images = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(download_image, img_url, output_dir): img_url for img_url in image_urls}
        for future in as_completed(futures):
            img_url, success = future.result()
            if success:
                downloaded_images.append(img_url)
            else:
                failed_images.append(img_url)
    
    return downloaded_images, failed_images

def collect_images_from_flickr(keywords, num_images, output_dir, api_key, api_secret):
    flickr = FlickrAPI(api_key, api_secret)
    
    photos = flickr.walk(
        text=keywords,
        tag_mode='all',
        tags=keywords,
        extras='url_c',
        per_page=100,
        sort='relevance'
    )
    
    image_urls = []
    for photo in photos:
        if len(image_urls) >= num_images:
            break
        image_urls.append(photo.get('url_c'))
    
    os.makedirs(output_dir, exist_ok=True)
    
    downloaded_images = []
    failed_images = []
    
    for img_url in image_urls:
        img_url, success = download_image(img_url, output_dir)
        if success:
            downloaded_images.append(img_url)
        else:
            failed_images.append(img_url)
    
    return downloaded_images, failed_images

def collect_images_from_500px(keywords, num_images, output_dir, consumer_key):
    fivehundredpx = FiveHundredPX(consumer_key)
    
    photos = fivehundredpx.photos.search(
        term=keywords,
        image_size=4,
        rpp=100,
        sort='relevance'
    )
    
    image_urls = []
    for photo in photos['photos']:
        if len(image_urls) >= num_images:
            break
        image_urls.append(photo['image_url'])
    
    os.makedirs(output_dir, exist_ok=True)
    
    downloaded_images = []
    failed_images = []
    
    for img_url in image_urls:
        img_url, success = download_image(img_url, output_dir)
        if success:
            downloaded_images.append(img_url)
        else:
            failed_images.append(img_url)
    
    return downloaded_images, failed_images

def collect_images(keywords, num_images, output_dir, flickr_api_key=None, flickr_api_secret=None, fivehundredpx_consumer_key=None):
    keywords_with_angles = keywords + ['様々な角度', '複数の構図']
    
    web_images, web_failed = collect_images_from_web(keywords_with_angles, num_images, output_dir)
    print(f"Web: {len(web_images)} images downloaded, {len(web_failed)} images failed.")
    
    if flickr_api_key and flickr_api_secret:
        flickr_images, flickr_failed = collect_images_from_flickr(keywords_with_angles, num_images, output_dir, flickr_api_key, flickr_api_secret)
        print(f"Flickr: {len(flickr_images)} images downloaded, {len(flickr_failed)} images failed.")
    
    if fivehundredpx_consumer_key:
        fivehundredpx_images, fivehundredpx_failed = collect_images_from_500px(keywords_with_angles, num_images, output_dir, fivehundredpx_consumer_key)
        print(f"500px: {len(fivehundredpx_images)} images downloaded, {len(fivehundredpx_failed)} images failed.")
    
    downloaded_images = web_images + flickr_images + fivehundredpx_images
    failed_images = web_failed + flickr_failed + fivehundredpx_failed
    
    print(f"Total: {len(downloaded_images)} images downloaded, {len(failed_images)} images failed.")
    
    return downloaded_images, failed_images

if __name__ == "__main__":
    keywords = ["建物", "建築"]
    num_images = 1000
    output_dir = "collected_images"
    flickr_api_key = "your_flickr_api_key"
    flickr_api_secret = "your_flickr_api_secret"
    fivehundredpx_consumer_key = "your_500px_consumer_key"
    
    downloaded_images, failed_images = collect_images(keywords, num_images, output_dir, flickr_api_key, flickr_api_secret, fivehundredpx_consumer_key)
