import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

# Function to read images from a folder
def read_images_from_folder(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        if filename.endswith(('.jpg', '.jpeg', '.png')):  
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) 
            if img is not None:
                images.append(img)
    return images

# Updated function to perform image retrieval
def retrieve_similar_images(query_image, dataset_folder):
    # Read images from the dataset folder
    dataset_images = read_images_from_folder(dataset_folder)

    
    orb = cv2.ORB_create()

    
    kp1, des1 = orb.detectAndCompute(query_image, None)

    # Brute Force Matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    matching_results = []

    for idx, dataset_image in enumerate(dataset_images):
        kp2, des2 = orb.detectAndCompute(dataset_image, None)

        matches = bf.match(des1, des2)

        
        matching_score = len(matches)

        # Append the result to the list
        matching_results.append((idx, matching_score))

    # Sort results based on matching scores
    matching_results = sorted(matching_results, key=lambda x: x[1], reverse=True)

    most_similar_indices = [result[0] for result in matching_results[:2]]

    for idx in most_similar_indices:
        plt.imshow(dataset_images[idx], cmap='gray')
        plt.title(f'Most Similar Image {most_similar_indices.index(idx) + 1}')
        plt.show()

    return most_similar_indices

# Function to calculate accuracy
def calculate_accuracy(ground_truth_indices, retrieved_indices):
    correct_matches = sum(1 for idx in retrieved_indices if idx in ground_truth_indices)
    total_queries = len(ground_truth_indices)
    accuracy = correct_matches / total_queries
    return accuracy

# Function to calculate precision at K
def calculate_precision_at_k(ground_truth_index, retrieved_indices, k):
    relevant_at_k = sum(1 for idx in retrieved_indices[:k] if idx == ground_truth_index)
    precision_at_k = relevant_at_k / k
    return precision_at_k

query_image_path = 'images/data_3.jpg'
dataset_folder_path = "images"

query_image = cv2.imread(query_image_path, cv2.IMREAD_GRAYSCALE)

ground_truth_indices = [1, 2]

# Use the retrieve_similar_images function with the dataset folder
retrieved_indices = retrieve_similar_images(query_image=query_image, dataset_folder=dataset_folder_path)

accuracy = calculate_accuracy(ground_truth_indices, retrieved_indices)
print(f'Accuracy: {accuracy}')

k = 2  # Set the value of K for precision calculation
precision_at_k = calculate_precision_at_k(ground_truth_indices[0], retrieved_indices, k)
print(f'Precision at {k}: {precision_at_k}')
