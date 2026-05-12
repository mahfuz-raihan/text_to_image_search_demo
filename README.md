# Text-to-Image Semantic Search Demo

This project demonstrates a semantic image search engine using the **OpenAI CLIP (Contrastive Language-Image Pre-training)** model. It allows users to search through a local collection of images using natural language text prompts.

## Overview

Unlike traditional image search that relies on filenames or manual tagging, this system understands the semantic content of both text and images. By leveraging CLIP, the system can:
- Map images into a high-dimensional vector space.
- Convert text queries into the same vector space.
- Find the most relevant images by calculating the **cosine similarity** between text and image embeddings.

## Tech Stack

- **Model**: OpenAI CLIP (`ViT-B/32`)
- **Framework**: PyTorch
- **Libraries**: 
  - `clip`: For model loading and encoding.
  - `Pillow`: For image processing.
  - `scikit-learn`: For calculating similarity scores.
  - `NumPy`: For numerical operations.

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd text_to_image_search_demo
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Place your images in the `image/` directory.
2. Update the `image_folder` path in `memes_search.py` if necessary (defaults to `file_path/` - *recommend updating to `./image/`*).
3. Run the search script:
   ```bash
   python memes_search.py
   ```
4. Enter your search query when prompted (e.g., "a happy dog", "sunset on the beach").
5. The system will return the top matches with their similarity scores.

## Scope of Exploration

This project provides a foundation for exploring several advanced AI concepts:
- **Zero-Shot Learning**: See how well the model identifies objects or concepts it hasn't been explicitly trained on.
- **Multimodal Embeddings**: Understand how visual and textual information can be represented in a shared space.
- **Model Comparison**: Swap `ViT-B/32` with other variants like `RN50` or `ViT-L/14` to compare accuracy and performance.

## Future Improvements

There are several directions to take this project for production-readiness or enhanced functionality:

1. **Embedding Persistence**: Currently, embeddings are re-generated on every run. Implementing a caching mechanism (e.g., saving to `.npy` files or a SQLite database) would drastically speed up startup time.
2. **Vector Database Integration**: For larger datasets (thousands of images), use a dedicated vector database like **FAISS**, **ChromaDB**, or **Pinecone** for efficient similarity searches.
3. **Web Interface**: Develop a frontend using **Streamlit**, **Gradio**, or a standard web framework (Next.js/React) to display results visually.
4. **Batch Processing**: Optimize image encoding by processing images in batches rather than individually.
5. **API Endpoint**: Wrap the search logic in a FastAPI or Flask app to serve it as a microservice.
6. **Support for more formats**: Expand support to GIFs and video frames.

---
*Developed by Mahfuz Raihan*
