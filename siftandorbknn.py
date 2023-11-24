import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

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

# Hybrid method combining ORB and SIFT with k-means clustering
def retrieve_similar_images_hybrid_kmeans(query_image, dataset_folder, num_clusters=50):
    # Read images from the dataset folder
    dataset_images = read_images_from_folder(dataset_folder)

    # ORB Detector
    orb = cv2.ORB_create()

    # SIFT Detector
    sift = cv2.SIFT_create()

    # Detect and compute keypoints and descriptors for the query image using ORB
    kp1_orb, des1_orb = orb.detectAndCompute(query_image, None)

    # Resize ORB descriptors to match the size of SIFT descriptors
    des1_orb_resized = cv2.resize(des1_orb, (128, des1_orb.shape[0]))

    # Detect and compute keypoints and descriptors for the query image using SIFT
    if query_image is not None:  # Add this check to ensure query_image is not empty
        kp1_sift, des1_sift = sift.detectAndCompute(query_image, None)
    else:
        des1_sift = np.empty((0, 128), dtype=np.uint8)  # Assuming SIFT descriptor size is 128

    
    descriptors_combined = np.concatenate((des1_orb_resized, des1_sift), axis=0)

    # Apply k-means clustering
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(descriptors_combined)

    # Assign cluster labels to ORB and SIFT descriptors
    labels_orb = kmeans.predict(des1_orb_resized)
    labels_sift = kmeans.predict(des1_sift)

    # List to store matching results
    matching_results = []

    for idx, dataset_image in enumerate(dataset_images):
       
        kp2_orb, des2_orb = orb.detectAndCompute(dataset_image, None)

        
        des2_orb_resized = cv2.resize(des2_orb, (128, des2_orb.shape[0]))

        
        kp2_sift, des2_sift = sift.detectAndCompute(dataset_image, None)

        # Assign cluster labels to dataset image descriptors
        labels2_orb = kmeans.predict(des2_orb_resized)
        labels2_sift = kmeans.predict(des2_sift)

        
        common_labels_orb = np.intersect1d(labels_orb, labels2_orb)

        # Count the common cluster labels between query and dataset for SIFT
        common_labels_sift = np.intersect1d(labels_sift, labels2_sift)

        # Calculate the matching score based on the number of common labels
        matching_score = len(common_labels_orb) + len(common_labels_sift)

        # Append the result to the list
        matching_results.append((idx, matching_score))

   
    matching_results = sorted(matching_results, key=lambda x: x[1], reverse=True)

    
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


query_image_path = 'images/data_3.jpg'
dataset_folder_path = 'images'

query_image = cv2.imread(query_image_path, cv2.IMREAD_GRAYSCALE)

# Ground truth indices (assumed for demonstration purposes)
ground_truth_indices = [1, 3]

retrieved_indices_hybrid_kmeans = retrieve_similar_images_hybrid_kmeans(query_image=query_image, dataset_folder=dataset_folder_path)

# Calculate and print accuracy
accuracy_hybrid_kmeans = calculate_accuracy(ground_truth_indices, retrieved_indices_hybrid_kmeans)
print(f'Accuracy using Hybrid (ORB + SIFT + k-means): {accuracy_hybrid_kmeans}')

# Calculate and print precision at K
k_hybrid_kmeans = 2 
precision_at_k_hybrid_kmeans = calculate_precision_at_k(ground_truth_indices[0], retrieved_indices_hybrid_kmeans, k_hybrid_kmeans)
print(f'Precision at {k_hybrid_kmeans} using Hybrid (ORB + SIFT + k-means): {precision_at_k_hybrid_kmeans}')
