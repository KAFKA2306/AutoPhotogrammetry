# image_collector.py
import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse

def collect_images(keywords, urls, output_dir):
    image_urls = []
    for url in urls:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        for img in soup.find_all('img'):
            img_url = img.get('src')
            if img_url and any(keyword in img_url for keyword in keywords):
                if not img_url.startswith('http'):
                    base_url = urlparse(url).scheme + '://' + urlparse(url).netloc
                    img_url = base_url + img_url
                image_urls.append(img_url)
    
    for i, img_url in enumerate(image_urls):
        try:
            response = requests.get(img_url)
            file_name = f"{i}.jpg"
            file_path = os.path.join(output_dir, file_name)
            with open(file_path, 'wb') as f:
                f.write(response.content)
        except Exception as e:
            print(f"Error downloading {img_url}: {e}")
    
    return image_urls

# image_clustering.py
import os
import numpy as np
from sklearn.cluster import DBSCAN
from skimage.io import imread
from skimage.feature import hog, local_binary_pattern, graycomatrix, graycoprops
from skimage.color import rgb2gray

def extract_features(image_path):
    image = imread(image_path)
    image_gray = rgb2gray(image)
    features = []
    features.extend(hog(image_gray))
    features.extend(local_binary_pattern(image_gray, P=8, R=1).flatten())
    features.extend(graycoprops(graycomatrix(image_gray), ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']).flatten())
    return features

def cluster_images(image_dir):
    image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.jpg')]
    features = [extract_features(image_path) for image_path in image_paths]
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    labels = dbscan.fit_predict(features)
    return labels, image_paths

# image_generator.py
import torch
from diffusers import StableDiffusionPipeline

def generate_images(prompt, num_images, output_dir):
    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", use_auth_token=True)
    pipe = pipe.to("cuda")
    
    generated_images = []
    for i in range(num_images):
        image = pipe(prompt).images[0]
        file_name = f"generated_{i}.jpg"
        file_path = os.path.join(output_dir, file_name)
        image.save(file_path)
        generated_images.append(file_path)
    
    return generated_images

# image_selector.py
import os
import numpy as np
from skimage.io import imread
from skimage.metrics import structural_similarity as ssim
from skimage.filters import laplace

def calculate_sharpness(image_path):
    image = imread(image_path)
    return np.mean(laplace(image))

def calculate_similarity(image_path1, image_path2):
    image1 = imread(image_path1)
    image2 = imread(image_path2)
    return ssim(image1, image2, multichannel=True)

def select_images(image_paths, sharpness_threshold, similarity_threshold, output_dir):
    selected_images = []
    for image_path in image_paths:
        if calculate_sharpness(image_path) >= sharpness_threshold:
            if not any(calculate_similarity(image_path, selected_image) >= similarity_threshold for selected_image in selected_images):
                selected_images.append(image_path)
                file_name = os.path.basename(image_path)
                output_path = os.path.join(output_dir, file_name)
                os.rename(image_path, output_path)
    return selected_images

# main.py
import os
from image_collector import collect_images
from image_clustering import cluster_images
from image_generator import generate_images
from image_selector import select_images

def main():
    keywords = ["building", "architecture"]
    urls = ["https://example.com/images", "https://example.org/photos"]
    collected_dir = "collected_images"
    clustered_dir = "clustered_images"
    generated_dir = "generated_images"
    selected_dir = "selected_images"
    
    os.makedirs(collected_dir, exist_ok=True)
    os.makedirs(clustered_dir, exist_ok=True)
    os.makedirs(generated_dir, exist_ok=True)
    os.makedirs(selected_dir, exist_ok=True)
    
    collected_images = collect_images(keywords, urls, collected_dir)
    print(f"{len(collected_images)} images collected.")
    
    clustered_labels, clustered_image_paths = cluster_images(collected_dir)
    for label, image_path in zip(clustered_labels, clustered_image_paths):
        file_name = os.path.basename(image_path)
        output_path = os.path.join(clustered_dir, f"cluster_{label}_{file_name}")
        os.rename(image_path, output_path)
    print("Images clustered.")
    
    generated_images = []
    for label in set(clustered_labels):
        prompt = f"A {keywords[0]} from a different angle"
        generated_images.extend(generate_images(prompt, num_images=5, output_dir=generated_dir))
    print(f"{len(generated_images)} images generated.")
    
    all_images = clustered_image_paths + generated_images
    selected_images = select_images(all_images, sharpness_threshold=8.0, similarity_threshold=0.8, output_dir=selected_dir)
    print(f"{len(selected_images)} images selected.")
    
    # 3Dモデルの構築と評価

if __name__ == "__main__":
    main()
