import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

# Function to read images from a folder
def read_images_from_folder(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        if filename.endswith(('.jpg', '.jpeg', '.png')):  # Add other supported image extensions
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Read images in grayscale
            if img is not None:
                images.append(img)
    return images

# Updated function to perform image retrieval using SIFT
def retrieve_similar_images_sift(query_image, dataset_folder):
    # Read images from the dataset folder
    dataset_images = read_images_from_folder(dataset_folder)

    # SIFT Detector
    sift = cv2.SIFT_create()

    # Detect and compute keypoints and descriptors for the query image
    kp1, des1 = sift.detectAndCompute(query_image, None)

    # Brute Force Matcher
    bf = cv2.BFMatcher()

    # List to store matching results
    matching_results = []

    for idx, dataset_image in enumerate(dataset_images):
        # Detect and compute keypoints and descriptors for the dataset image
        kp2, des2 = sift.detectAndCompute(dataset_image, None)

        # Match descriptors
        matches = bf.knnMatch(des1, des2, k=2)

        # Apply ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        # Calculate the matching score based on the number of good matches
        matching_score = len(good_matches)

        # Append the result to the list
        matching_results.append((idx, matching_score))

    # Sort results based on matching scores
    matching_results = sorted(matching_results, key=lambda x: x[1], reverse=True)

    # Retrieve the indices of the two most similar images
    most_similar_indices = [result[0] for result in matching_results[:2]]

    # Display the two most similar images
    for idx in most_similar_indices:
        plt.imshow(dataset_images[idx], cmap='gray')
        plt.title(f'Most Similar Image {most_similar_indices.index(idx) + 1}')
        plt.show()

    return most_similar_indices

# Function to calculate accuracy
def calculate_accuracy(ground_truth_indices, retrieved_indices):
    correct_matches = sum(1 for idx in retrieved_indices if idx in ground_truth_indices)
    total_queries = len(ground_truth_indices)
    accuracy = correct_matches / total_queries if total_queries != 0 else 0.0
    return accuracy

# Function to calculate precision at K
def calculate_precision_at_k(ground_truth_index, retrieved_indices, k):
    relevant_at_k = sum(1 for idx in retrieved_indices[:k] if idx == ground_truth_index)
    precision_at_k = relevant_at_k / k if k != 0 else 0.0
    return precision_at_k

# Example usage
query_image_path = 'images/data_3.jpg'
dataset_folder_path = 'images'

# Read the query image
query_image = cv2.imread(query_image_path, cv2.IMREAD_GRAYSCALE)

# Ground truth indices (assumed for demonstration purposes)
ground_truth_indices = [1, 3]

# Use the retrieve_similar_images_sift function with the dataset folder
retrieved_indices_sift = retrieve_similar_images_sift(query_image=query_image, dataset_folder=dataset_folder_path)

# Calculate and print accuracy
accuracy_sift = calculate_accuracy(ground_truth_indices, retrieved_indices_sift)
print(f'Accuracy using SIFT: {accuracy_sift}')

# Calculate and print precision at K
k_sift = 2  # Set the value of K for precision calculation
precision_at_k_sift = calculate_precision_at_k(ground_truth_indices[0], retrieved_indices_sift, k_sift)
print(f'Precision at {k_sift} using SIFT: {precision_at_k_sift}')
