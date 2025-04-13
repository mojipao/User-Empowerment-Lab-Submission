# Reddit Topic Modeling Project

This project implements various text analysis and topic modeling techniques to analyze Reddit posts, from simple word frequency analysis to advanced methods like BERTopic and LDA.

## Project Structure

- `reddit_analysis_project.py`: Main script implementing all analysis methods
- `topic_clustering_comparative.py`: Example code with implementation of various topic clustering techniques
- `requirements.txt`: Required Python packages

## Methods Implemented

### Basic Techniques
1. **Word Frequency Analysis**: Simple counting and visualization of common words
2. **TF-IDF + K-Means Clustering**: Using term weighting and clustering
3. **Word2Vec + HDBSCAN Clustering**: Using word embeddings and density-based clustering

### Advanced Techniques
1. **BERTopic**: Using contextual embeddings with dimensionality reduction and clustering
2. **LDA (Latent Dirichlet Allocation)**: Using probabilistic topic modeling

## Setup and Installation

1. Install required packages:
   ```
   python -m pip install -r requirements.txt
   ```

2. Run the main analysis script:
   ```
   python reddit_analysis_project.py
   ```

## Key Findings

The analysis reveals several patterns in Reddit post data:
- AI is a dominant topic in various contexts
- Emotional expression is very common
- Multi-language content is present (English and French)
- Personal relationships and struggles are common themes

## Results

The analysis generates visualizations and a detailed report in the `reddit_analysis_results` directory, including:
- Word frequency bar charts
- Word clouds
- Topic lists with representative terms
- Example posts for each topic
- Comparative analysis of all methods 