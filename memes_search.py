# The program task is to text to image search
# used framework: Pytorch
# model: openai/clip(ViT-B/32) model for text to image findings
# Author: Mahfuz


# import the libraries
import torch
import clip
from PIL import Image
import os
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# load the open ai clip model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'The device we found: {device}')
model, preprocess = clip.load("ViT-B/32", device)

# Define the path to the folder containing images
image_folder = "/media/mahfuz/Media/ML_challanges/challange_1/image"

# Preprocess and Generate Image Embeddings
def generate_image_embeddings(folder_path):
    image_embeddings = {}
    for image_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_name)
        if image_name.endswith(('.png', '.jpg', '.jpeg','.webp')):
            try:
                # Load and preprocess the image
                image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
                
                # Generate image embedding
                with torch.no_grad():
                    embedding = model.encode_image(image)
                    embedding = embedding / embedding.norm(dim=-1, keepdim=True)
                    image_embeddings[image_name] = embedding.cpu().numpy()
            except Exception as e:
                print(f"Error processing {image_name}: {e}")
    return image_embeddings

image_embeddings = generate_image_embeddings(image_folder)

# Accept a Text Query and Generate Text Embedding
def get_text_embedding(query):
    text = clip.tokenize([query]).to(device)
    with torch.no_grad():
        embedding = model.encode_text(text)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
    return embedding.cpu().numpy()


# Compute Similarity and Retrieve Top Matches
def search_images(query, image_embeddings, top_k=5):
    # Generate text embedding
    text_embedding = get_text_embedding(query)

    # Compute cosine similarity with all image embeddings
    similarities = {}
    for image_name, img_embedding in image_embeddings.items():
        sim = cosine_similarity(text_embedding, img_embedding)
        similarities[image_name] = sim[0][0]

    # Sort images by similarity score in descending order
    sorted_images = sorted(similarities.items(), key=lambda x: x[1], reverse=True)

    return sorted_images[:top_k]

# Step 7: Test the System
if __name__ == "__main__":
    while True:
        query = input("Inter a query (or 'exit' to quit): ")
        if query.lower() =='exit':
            print('Exiting the program. Thanks')
            break
        
        top_matches = search_images(query, image_embeddings, top_k=5)

        print("Top Matches:")
        for image_name, similarity in top_matches:
            print(f"Image: {image_name}, Similarity: {similarity:.4f}")