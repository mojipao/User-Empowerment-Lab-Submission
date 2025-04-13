# Reddit Topic Modeling Project

This project implements various text analysis and topic modeling techniques to analyze Reddit posts. It includes methods ranging from simple word frequency analysis to advanced techniques like BERTopic and LDA.

## Project Structure

- `reddit_analysis_project.py`: Main Python script that implements all analysis methods
- `requirements.txt`: Required Python packages
- `reddit_analysis_results/`: Directory containing output files
  - `final_report.md`: Detailed report of all analysis results
  - `word_frequency.png`: Bar chart of most common words
  - `wordcloud.png`: Word cloud visualization
- `reddit_analysis_summary.md`: Executive summary of findings

## Methods Implemented

1. **Word Frequency Analysis**: Simple counting and visualization of common words
2. **TF-IDF + K-Means Clustering**: Using term weighting and clustering
3. **Word2Vec + HDBSCAN Clustering**: Using word embeddings and density-based clustering
4. **BERTopic**: Using contextual embeddings with dimensionality reduction and clustering
5. **LDA (Latent Dirichlet Allocation)**: Using probabilistic topic modeling

## Requirements

- Python 3.8+
- Required packages in `requirements.txt`

## Setup and Installation

1. Install required packages:
   ```
   python -m pip install -r requirements.txt
   ```

2. Download NLTK data (done automatically when running the script):
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   ```

## Usage

Run the main analysis script:
```
python reddit_analysis_project.py
```

The script will:
1. Load and preprocess the Reddit data
2. Apply all analysis methods
3. Generate visualizations
4. Create a detailed report in the `reddit_analysis_results` directory

## Key Findings

The analysis revealed several interesting patterns in the Reddit data:

1. AI is a dominant topic in various contexts (art, technology, ethics)
2. Emotional expression is very common
3. Multi-language content is present (English and French)
4. Personal relationships and struggles are common themes
5. Technology complaints form a distinct topic cluster

## Conclusion

The project demonstrates that combining multiple text analysis approaches provides a more comprehensive understanding of text data. For this Reddit dataset, TF-IDF + K-Means and LDA proved most effective at identifying meaningful topic clusters. 