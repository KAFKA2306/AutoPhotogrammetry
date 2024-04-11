import os
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.feature_selection import mutual_info_classif
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

def optimize_feature_weights(features, labels):
    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(features, labels)
    feature_importances = rf.feature_importances_
    return feature_importances

def reduce_dimensionality(features):
    pca = PCA(n_components=0.95)
    features_reduced = pca.fit_transform(features)
    return features_reduced

def evaluate_clustering(features, labels):
    silhouette_avg = silhouette_score(features, labels)
    db_index = davies_bouldin_score(features, labels)
    return silhouette_avg, db_index

def select_representative_images(features, labels, image_paths):
    representative_images = []
    for label in set(labels):
        cluster_features = features[labels == label]
        cluster_paths = [path for path, l in zip(image_paths, labels) if l == label]
        center_index = np.argmin(np.sum(np.square(cluster_features - np.mean(cluster_features, axis=0)), axis=1))
        representative_images.append(cluster_paths[center_index])
    return representative_images

def save_features_and_labels(features, labels, image_paths, output_dir):
    data = pd.DataFrame(features, columns=[f"feature_{i}" for i in range(features.shape[1])])
    data["label"] = labels
    data["image_path"] = image_paths
    output_path = os.path.join(output_dir, "features_and_labels.csv")
    data.to_csv(output_path, index=False)
    print(f"Features and labels saved to {output_path}")

def cluster_images(image_dir, output_dir):
    image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.jpg')]
    features = np.array([extract_features(image_path) for image_path in image_paths])
    
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    labels = dbscan.fit_predict(features)
    
    feature_weights = optimize_feature_weights(features, labels)
    features_weighted = features * feature_weights
    
    features_reduced = reduce_dimensionality(features_weighted)
    
    silhouette_avg, db_index = evaluate_clustering(features_reduced, labels)
    print(f"Silhouette score: {silhouette_avg}, Davies-Bouldin index: {db_index}")
    
    representative_images = select_representative_images(features_reduced, labels, image_paths)
    print(f"Representative images: {representative_images}")
    
    save_features_and_labels(features_weighted, labels, image_paths, output_dir)
    
    return labels, image_paths, representative_images
