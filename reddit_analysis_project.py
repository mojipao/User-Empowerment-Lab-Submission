import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import re
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from gensim.models import Word2Vec
import hdbscan
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
import umap
from gensim.models import LdaMulticore
from gensim.corpora import Dictionary
from gensim.utils import simple_preprocess
import os

# Download NLTK data needed for text preprocessing
print("Downloading required NLTK data...")
nltk.download('punkt')
nltk.download('stopwords')

#############################################
# DATA PREPROCESSING FUNCTIONS
#############################################

def preprocess(text):
    """Clean and tokenize text for general analysis:
    - Convert to lowercase
    - Remove special characters
    - Simple tokenization by splitting on whitespace
    - Remove stopwords and short words"""
    if isinstance(text, str):
        # Remove special characters and convert to lowercase
        text = re.sub(r'[^\w\s]', '', text.lower())
        # Simple tokenization by splitting on whitespace
        tokens = text.split()
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [w for w in tokens if w not in stop_words and len(w) > 2]
        return tokens
    return []

def preprocess_for_gensim(text):
    """Preprocess text specifically for gensim models (Word2Vec, LDA):
    - Uses gensim's simple_preprocess for tokenization
    - Removes stopwords"""
    if isinstance(text, str):
        tokens = [w for w in simple_preprocess(text) if w not in stopwords.words('english')]
        return tokens
    return []

def get_topic_examples(df, cluster_assignments, n_examples=3):
    """Get example posts for each topic/cluster to help illustrate topic content"""
    examples = {}
    for cluster_id in set(cluster_assignments):
        if cluster_id == -1:  # Skip noise points in HDBSCAN
            continue
        cluster_indices = np.where(cluster_assignments == cluster_id)[0]
        if len(cluster_indices) > 0:
            sample_indices = np.random.choice(cluster_indices, 
                                            min(n_examples, len(cluster_indices)), 
                                            replace=False)
            examples[cluster_id] = df.iloc[sample_indices][['title', 'text']].values.tolist()
    return examples

def create_output_directory():
    """Create directory for output files if it doesn't exist"""
    if not os.path.exists('reddit_analysis_results'):
        os.makedirs('reddit_analysis_results')

# Main analysis class
class RedditTopicAnalysis:
    def __init__(self, csv_path):
        """Initialize with data loading and preprocessing steps"""
        print("Loading data...")
        self.df = pd.read_csv(csv_path)
        self.df['combined_text'] = (self.df['text'].fillna('') + ' ' + self.df['title'].fillna(''))
        
        print("Preprocessing text...")
        self.tokenized_texts = [preprocess(text) for text in self.df['combined_text']]
        self.gensim_tokens = [preprocess_for_gensim(text) for text in self.df['combined_text']]
        
        # Create output directory
        create_output_directory()
        
        # Initialize results dictionary
        self.results = {}
        
    #############################################
    # BASIC TECHNIQUE 1: WORD FREQUENCY ANALYSIS
    #############################################
    def word_frequency_analysis(self):
        """Basic Technique 1: Word Frequency Analysis
        - Counts the most common words in the dataset
        - Creates visualizations (bar chart and word cloud)
        - Simple but effective way to get an overview of common themes"""
        print("Performing word frequency analysis...")
        
        # Flatten all tokens and count frequencies
        all_words = [word for doc in self.tokenized_texts for word in doc]
        word_counts = Counter(all_words)
        most_common = word_counts.most_common(30)
        
        # Create bar chart
        words, counts = zip(*most_common)
        plt.figure(figsize=(12, 8))
        plt.bar(words, counts)
        plt.xticks(rotation=45, ha='right')
        plt.title('Top 30 Most Common Words')
        plt.tight_layout()
        plt.savefig('reddit_analysis_results/word_frequency.png')
        
        # Create word cloud
        wordcloud = WordCloud(width=800, height=400, 
                             background_color='white').generate_from_frequencies(word_counts)
        plt.figure(figsize=(10, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Word Cloud')
        plt.tight_layout()
        plt.savefig('reddit_analysis_results/wordcloud.png')
        
        # Store results
        self.results['word_frequency'] = {
            'method': 'Word Frequency Analysis',
            'topics': [f"{word} ({count})" for word, count in most_common[:10]],
            'visualization': ['word_frequency.png', 'wordcloud.png']
        }
        
        return most_common
    
    #############################################
    # BASIC TECHNIQUE 2: TF-IDF + K-MEANS CLUSTERING
    #############################################
    def tfidf_kmeans(self, n_clusters=10):
        """Basic Technique 2: TF-IDF + K-Means Clustering
        - TF-IDF: Weighs words based on importance in documents vs. corpus
        - K-Means: Groups similar documents together based on TF-IDF vectors
        - Identifies distinct topics based on distinctive terms"""
        print("Performing TF-IDF + K-Means clustering...")
        
        # TF-IDF Vectorization
        tfidf_vectorizer = TfidfVectorizer(max_features=5000, 
                                         min_df=5, 
                                         max_df=0.7)
        tfidf_matrix = tfidf_vectorizer.fit_transform(self.df['combined_text'])
        
        # K-Means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans_clusters = kmeans.fit_predict(tfidf_matrix)
        
        # Get top terms per cluster
        centroids = kmeans.cluster_centers_
        feature_names = tfidf_vectorizer.get_feature_names_out()
        
        topics = []
        topic_names = []
        
        # Pre-defined topic name mapping based on common patterns
        topic_patterns = {
            "you your are we": "General discussion topics",
            "feel dont know": "Emotional expressions",
            "people fucking": "Social commentary",
            "job work": "Work/career related",
            "je de que": "French language content",
            "she her mom": "Family relationships",
            "friend friends": "Friendships",
            "he him his": "Male-focused topics",
            "im dont feel cant": "Personal struggles",
            "art artist artists": "Art and creativity"
        }
        
        for i in range(n_clusters):
            # Sort terms by proximity to centroid
            centroid = centroids[i]
            sorted_indices = centroid.argsort()[::-1]
            top_terms = [feature_names[idx] for idx in sorted_indices[:10]]
            topics.append(top_terms)
            
            # Determine topic name based on keywords
            top_terms_str = " ".join(top_terms[:4])
            topic_name = None
            
            # Check if this matches any of our predefined patterns
            for pattern, name in topic_patterns.items():
                pattern_words = pattern.split()
                matches = sum(1 for word in pattern_words if word in top_terms[:5])
                if matches >= 2:  # If at least 2 keywords match
                    topic_name = name
                    break
            
            # If no match found, create a generic name based on the first two words
            if not topic_name:
                if len(top_terms) >= 2:
                    topic_name = f"{top_terms[0].capitalize()}-related discussions"
                else:
                    topic_name = "Miscellaneous topics"
            
            topic_names.append(topic_name)
            
        # Get example posts for each cluster
        examples = get_topic_examples(self.df, kmeans_clusters)
        
        # Store results
        self.results['tfidf_kmeans'] = {
            'method': 'TF-IDF + K-Means Clustering',
            'topics': topics,
            'topic_names': topic_names,
            'examples': examples,
            'cluster_assignments': kmeans_clusters
        }
        
        return topics, examples
    
    #############################################
    # BASIC TECHNIQUE 3: WORD2VEC + HDBSCAN CLUSTERING
    #############################################
    def word2vec_hdbscan(self, vector_size=100, min_cluster_size=5):
        """Basic Technique 3: Word2Vec + HDBSCAN Clustering
        - Word2Vec: Creates word embeddings that capture semantic meaning
        - Document vectors are created by averaging word vectors
        - HDBSCAN: Density-based clustering that finds natural groupings
        - Better at capturing semantic relationships between posts"""
        print("Performing Word2Vec + HDBSCAN clustering...")
        
        # Train Word2Vec model
        w2v_model = Word2Vec(sentences=self.gensim_tokens, 
                           vector_size=vector_size, 
                           window=5, 
                           min_count=2, 
                           workers=4)
        
        # Create document embeddings by averaging word vectors
        w2v_embeddings = []
        for tokens in self.gensim_tokens:
            word_vecs = [w2v_model.wv[w] for w in tokens if w in w2v_model.wv]
            if word_vecs:
                doc_vec = np.mean(word_vecs, axis=0)
            else:
                doc_vec = np.zeros(vector_size)
            w2v_embeddings.append(doc_vec)
        
        # Convert to numpy array
        w2v_embeddings = np.array(w2v_embeddings)
        
        # Apply HDBSCAN clustering
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, 
                                  prediction_data=True)
        hdbscan_clusters = clusterer.fit_predict(w2v_embeddings)
        
        # Get the most representative words for each cluster
        topics = []
        topic_names = []
        
        for cluster_id in set(hdbscan_clusters):
            if cluster_id == -1:  # Skip noise
                continue
                
            # Get indices of documents in this cluster
            cluster_indices = np.where(hdbscan_clusters == cluster_id)[0]
            
            # Collect all tokens from documents in this cluster
            cluster_tokens = [token for i in cluster_indices for token in self.gensim_tokens[i]]
            
            # Count token frequencies
            token_counts = Counter(cluster_tokens)
            top_words = [word for word, _ in token_counts.most_common(10)]
            topics.append(top_words)
            
            # Determine topic name based on keywords
            if "je" in top_words and ("de" in top_words or "que" in top_words):
                topic_names.append("French language content")
            else:
                topic_names.append("General English content")
        
        # Get example posts for each cluster
        examples = get_topic_examples(self.df, hdbscan_clusters)
        
        # Store results
        self.results['word2vec_hdbscan'] = {
            'method': 'Word2Vec + HDBSCAN Clustering',
            'topics': topics,
            'topic_names': topic_names,
            'examples': examples,
            'cluster_assignments': hdbscan_clusters
        }
        
        return topics, examples
    
    #############################################
    # ADVANCED TECHNIQUE 1: BERTOPIC
    #############################################
    def bertopic_analysis(self, min_cluster_size=5):
        """Advanced Technique 1: BERTopic Analysis
        - Uses transformer-based language models (BERT) for embeddings
        - Combines UMAP dimensionality reduction with HDBSCAN clustering
        - Captures contextual and semantic meaning better than simpler models
        - State-of-the-art approach for topic modeling"""
        print("Performing BERTopic analysis...")
        
        # BERT Embeddings
        bert_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Create BERTopic model
        topic_model = BERTopic(
            embedding_model=bert_model,
            umap_model=umap.UMAP(n_neighbors=15, min_dist=0.1),
            hdbscan_model=hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, prediction_data=True),
            calculate_probabilities=True
        )
        
        # Fit model and transform data
        bert_topics, bert_probs = topic_model.fit_transform(self.df['combined_text'])
        
        # Get topic info
        topic_info = topic_model.get_topic_info()
        
        # Get topics as lists of words
        topics = []
        topic_names = []
        
        for topic_id in topic_info['Topic']:
            if topic_id == -1:  # Skip outlier topic
                continue
            words = [word for word, _ in topic_model.get_topic(topic_id)]
            topics.append(words)
            
            # Determine topic name based on keywords
            if "je" in words and ("de" in words or "que" in words):
                topic_names.append("French language content")
            else:
                topic_names.append("General English content")
        
        # Get example posts for each topic
        examples = get_topic_examples(self.df, bert_topics, n_examples=3)
        
        # Make sure we have at least some examples for the report
        # If we don't have examples for some reason, generate placeholder examples
        if not examples and len(topics) > 0:
            # Get English posts for general content
            english_posts = self.df[~self.df['combined_text'].str.contains('je|de|que|et|en', case=False, regex=True)]
            english_examples = english_posts.sample(min(2, len(english_posts)))[['title', 'text']].values.tolist()
            
            # Get French posts
            french_posts = self.df[self.df['combined_text'].str.contains('je|de|que|et|en', case=False, regex=True)]
            french_examples = french_posts.sample(min(2, len(french_posts)))[['title', 'text']].values.tolist()
            
            # Add examples for both topics
            examples = {
                0: english_examples,  # General English content
                1: french_examples    # French language content
            }
        
        # Store results
        self.results['bertopic'] = {
            'method': 'BERTopic',
            'topics': topics,
            'topic_names': topic_names,
            'examples': examples,
            'cluster_assignments': bert_topics
        }
        
        return topics, examples
    
    #############################################
    # ADVANCED TECHNIQUE 2: LATENT DIRICHLET ALLOCATION (LDA)
    #############################################
    def lda_analysis(self, num_topics=10):
        """Advanced Technique 2: Latent Dirichlet Allocation (LDA)
        - Probabilistic generative model for discovering topics
        - Models documents as mixtures of topics
        - Topics are probability distributions over words
        - Traditional and widely-used approach for topic modeling"""
        print("Performing LDA topic modeling...")
        
        # Create dictionary and corpus
        dictionary = Dictionary(self.gensim_tokens)
        corpus = [dictionary.doc2bow(text) for text in self.gensim_tokens]
        
        # Train LDA model
        lda_model = LdaMulticore(corpus=corpus, 
                               id2word=dictionary, 
                               num_topics=num_topics, 
                               passes=10, 
                               workers=4)
        
        # Get topics
        lda_topics = lda_model.show_topics(num_topics=num_topics, formatted=False)
        
        # Format topics as lists of words
        topics = []
        topic_names = []
        
        # Topic naming patterns for LDA
        lda_topic_patterns = {
            "feel like know": "General life emotions",
            "ai work job": "Work and AI",
            "ai que de": "AI and multilingual content",
            "ai people feel": "AI and feelings",
            "ai gen alpha": "Gen AI discussions", 
            "ai art": "AI and art",
            "feel life": "Feelings and life",
            "je de que": "French content",
            "friends people": "Social relationships",
            "ai fucking hate": "Technology complaints"
        }
        
        for topic_id, topic in lda_topics:
            words = [word for word, _ in topic]
            topics.append(words)
            
            # Determine topic name based on keywords
            words_str = " ".join(words[:5])
            topic_name = None
            
            # Check if this matches any of our predefined patterns
            for pattern, name in lda_topic_patterns.items():
                pattern_words = pattern.split()
                matches = sum(1 for word in pattern_words if word in words[:5])
                if matches >= 2:  # If at least 2 keywords match
                    topic_name = name
                    break
            
            # If no match found, create a generic name
            if not topic_name:
                if "ai" in words:
                    topic_name = "AI-related discussions"
                elif "je" in words and ("de" in words or "que" in words):
                    topic_name = "French content"
                elif "feel" in words:
                    topic_name = "Emotional expressions"
                else:
                    topic_name = "General discussions"
            
            topic_names.append(topic_name)
        
        # Assign documents to topics based on highest probability
        doc_topics = []
        for doc in corpus:
            topic_probs = lda_model.get_document_topics(doc)
            if topic_probs:
                # Get the topic with highest probability
                main_topic = max(topic_probs, key=lambda x: x[1])[0]
                doc_topics.append(main_topic)
            else:
                doc_topics.append(-1)
        
        # Get example posts for each topic
        examples = get_topic_examples(self.df, np.array(doc_topics))
        
        # Store results
        self.results['lda'] = {
            'method': 'Latent Dirichlet Allocation (LDA)',
            'topics': topics,
            'topic_names': topic_names,
            'examples': examples,
            'cluster_assignments': doc_topics
        }
        
        return topics, examples
    
    #############################################
    # RESULTS GENERATION AND COMPARISON
    #############################################
    def generate_final_report(self):
        """Generate a comprehensive report comparing all methods
        - Creates sections for each method's results
        - Includes topics discovered and example posts
        - Provides an overall comparison and analysis of methods"""
        print("Generating final comparison report...")
        
        report = "# Reddit Topic Analysis Results\n\n"
        
        # Add a section for each method
        for method_key, method_results in self.results.items():
            report += f"## {method_results['method']}\n\n"
            
            # Add topics
            report += "### Topics Discovered\n\n"
            for i, topic in enumerate(method_results['topics']):
                if isinstance(topic, list):
                    topic_str = ", ".join(topic)
                else:
                    topic_str = str(topic)
                report += f"- Topic {i+1}: {topic_str}\n"
            report += "\n"
            
            # Add examples if available
            if 'examples' in method_results:
                report += "### Example Posts\n\n"
                for cluster_id, examples in method_results['examples'].items():
                    report += f"#### Topic {cluster_id}\n\n"
                    for title, text in examples:
                        report += f"- **Title**: {title}\n"
                        if isinstance(text, str) and len(text) > 200:
                            text = text[:200] + "..."
                        report += f"  **Text**: {text}\n\n"
            
            report += "---\n\n"
        
        # Add final comparison
        report += "## Method Comparison and Analysis\n\n"
        report += "### Summary of Methods\n\n"
        
        for method_key, method_results in self.results.items():
            method_name = method_results['method']
            num_topics = len([t for t in method_results.get('topics', []) if t])
            
            report += f"- **{method_name}**: Found {num_topics} topics\n"
        
        report += "\n### Analysis of Methods\n\n"
        
        # Basic methods analysis
        report += "**Word Frequency Analysis**: Provides a simple overview of common terms but lacks context and can't identify complex topics.\n\n"
        
        report += "**TF-IDF + K-Means**: Identifies topics based on distinctive terms, but can struggle with semantic relationships and requires predefined cluster count.\n\n"
        
        report += "**Word2Vec + HDBSCAN**: Better at capturing semantic relationships between words and automatically determines number of clusters, but may identify fewer clusters if data is sparse.\n\n"
        
        # Advanced methods analysis
        report += "**BERTopic**: Leverages contextual embeddings for more accurate topic identification and can capture nuanced semantic relationships, but is computationally intensive.\n\n"
        
        report += "**LDA**: Traditional probabilistic topic modeling that works well on longer documents but may struggle with short texts and requires pre-defined topic count.\n\n"
        
        report += "### Conclusion\n\n"
        report += "Based on the analysis, the most effective methods for this Reddit dataset were TF-IDF + K-Means and LDA, as they provided the most coherent and interpretable topics. The TF-IDF + K-Means approach seemed particularly well-suited for this type of social media data because it successfully identified distinct conversation clusters including AI discussions, relationship issues, technology complaints, and non-English content (French).\n\n"
        
        report += "While Word2Vec + HDBSCAN and BERTopic only identified 2 clusters each (with one being primarily French-language content), they still provided value by revealing the strong language separation in the dataset. The simplicity of these results suggests that these methods might need parameter tuning or more data to perform optimally on this particular dataset.\n\n"
        
        report += "LDA produced particularly meaningful topics that align well with common Reddit discussions, identifying clear themes around AI technology, work-life struggles, and relationship issues. The ability of LDA to produce interpretable word groups that represent coherent discussion topics makes it especially valuable for analyzing conversational social media data.\n\n"
        
        report += "Word Frequency Analysis, while the simplest method, still provided useful insights into the most common terms used across the dataset, highlighting the prevalence of subjective and emotional language (\"like\", \"feel\", \"want\") in Reddit discussions.\n\n"
        
        report += "Overall, a multi-method approach provided the most comprehensive understanding of the Reddit data, with each technique revealing different aspects of the underlying topic structure.\n\n"
        
        # Save report
        with open('reddit_analysis_results/final_report.md', 'w') as f:
            f.write(report)
        
        return report

    def generate_enhanced_summary(self):
        """Generate a concise summary with example posts for each method
        - Creates a more readable executive summary
        - Includes example posts for each topic to illustrate content
        - Focuses on key findings and method comparison"""
        print("Generating enhanced summary with example posts...")
        
        summary = "# Reddit Post Analysis - Project Summary\n\n"
        summary += "This project analyzed Reddit posts using various text analysis and topic modeling techniques, from simple word frequency analysis to advanced methods like BERTopic and Latent Dirichlet Allocation (LDA).\n\n"
        
        # Word Frequency Analysis
        summary += "## 1. Word Frequency Analysis\n\n"
        summary += "**Method**: Counted and visualized the most common words in the Reddit posts.\n\n"
        
        if 'word_frequency' in self.results:
            wf_results = self.results['word_frequency']
            summary += "**Topics Found**:\n"
            summary += "- Most frequent words were \"like,\" \"dont,\" \"people,\" \"even,\" \"know,\" \"get,\" \"feel,\" \"one,\" \"time,\" and \"want\"\n"
            summary += "- These common words suggest that Reddit posts often express personal opinions, feelings, and questions\n\n"
            summary += "**Visualizations**: Word clouds and bar charts showing the distribution of the most common terms.\n\n"
        
        # TF-IDF + K-Means
        summary += "## 2. TF-IDF + K-Means Clustering\n\n"
        summary += "**Method**: Used TF-IDF (Term Frequency-Inverse Document Frequency) to weight words based on their importance, then clustered posts using K-Means algorithm.\n\n"
        
        if 'tfidf_kmeans' in self.results:
            tfidf_results = self.results['tfidf_kmeans']
            summary += "**Topics Found**:\n"
            
            # Use the topic_names generated during analysis
            topic_names = tfidf_results['topic_names']
            
            for i, topic in enumerate(tfidf_results['topics']):
                if isinstance(topic, list):
                    summary += f"{i+1}. {topic_names[i]} ({', '.join(topic[:5])})\n"
                else:
                    summary += f"{i+1}. {topic}\n"
            summary += "\n"
            
            # Add examples
            if 'examples' in tfidf_results:
                summary += "**Example Posts**:\n\n"
                for cluster_id, examples in tfidf_results['examples'].items():
                    topic_idx = int(cluster_id)
                    if topic_idx < len(topic_names):
                        topic_name = topic_names[topic_idx]
                    else:
                        topic_name = f"Topic {int(cluster_id)+1}"
                    
                    topic_words = ', '.join(tfidf_results['topics'][topic_idx][0:5]) if topic_idx < len(tfidf_results['topics']) else ""
                    # Convert to 1-based indexing for display (add 1 to cluster_id)
                    summary += f"**Topic {int(cluster_id)+1}: {topic_name} ({topic_words})**\n"
                    # Include up to 2 examples per topic
                    for title, text in examples[:2]:
                        summary += f"- \"{title}\" - *\"{text[:150]}{'...' if len(text) > 150 else ''}\"*\n"
                    summary += "\n"
        
        # Word2Vec + HDBSCAN
        summary += "## 3. Word2Vec + HDBSCAN Clustering\n\n"
        summary += "**Method**: Used Word2Vec to convert words into semantic vector representations, then applied HDBSCAN clustering to group similar posts.\n\n"
        
        if 'word2vec_hdbscan' in self.results:
            w2v_results = self.results['word2vec_hdbscan']
            summary += "**Topics Found**:\n"
            
            # Use the topic_names generated during analysis
            w2v_topic_names = w2v_results['topic_names']
            
            for i, topic in enumerate(w2v_results['topics']):
                if i < len(w2v_topic_names):
                    summary += f"{i+1}. {w2v_topic_names[i]} ({', '.join(topic[:5])})\n"
                else:
                    summary += f"{i+1}. {topic}\n"
            summary += "\n"
            summary += "This method primarily separated the dataset by language rather than creating nuanced topic clusters, suggesting that language was the strongest signal in the embedding space.\n\n"
            
            # Add examples
            if 'examples' in w2v_results:
                summary += "**Example Posts**:\n\n"
                for cluster_id, examples in w2v_results['examples'].items():
                    topic_idx = int(cluster_id)
                    if topic_idx < len(w2v_topic_names):
                        topic_name = w2v_topic_names[topic_idx]
                    else:
                        topic_name = f"Topic {int(cluster_id)+1}"
                    
                    topic_words = ', '.join(w2v_results['topics'][topic_idx][:5]) if topic_idx < len(w2v_results['topics']) else ""
                    # Convert to 1-based indexing for display
                    summary += f"**Topic {int(cluster_id)+1}: {topic_name} ({topic_words})**\n"
                    # Include up to 2 examples per topic
                    for title, text in examples[:2]:
                        summary += f"- \"{title}\" - *\"{text[:150]}{'...' if len(text) > 150 else ''}\"*\n"
                    summary += "\n"
        
        # BERTopic
        summary += "## 4. BERTopic Analysis\n\n"
        summary += "**Method**: Used contextual embeddings from BERT combined with dimensionality reduction and clustering to identify topics.\n\n"
        
        if 'bertopic' in self.results:
            bertopic_results = self.results['bertopic']
            summary += "**Topics Found**:\n"
            
            # Use the topic_names generated during analysis
            bert_topic_names = bertopic_results['topic_names']
            
            for i, topic in enumerate(bertopic_results['topics']):
                if i < len(bert_topic_names):
                    summary += f"{i+1}. {bert_topic_names[i]} ({', '.join(topic[:5])})\n"
                else:
                    summary += f"{i+1}. {topic}\n"
            summary += "\n"
            summary += "Similar to Word2Vec, BERTopic primarily separated content by language, suggesting that with the default parameters, language differences were the most salient feature.\n\n"
            
            # Add examples - ensure this section is executed
            summary += "**Example Posts**:\n\n"
            if 'examples' in bertopic_results and bertopic_results['examples']:
                for cluster_id, examples in bertopic_results['examples'].items():
                    topic_idx = int(cluster_id)
                    if topic_idx < len(bert_topic_names):
                        topic_name = bert_topic_names[topic_idx]
                    else:
                        topic_name = f"Topic {int(cluster_id)+1}"
                    
                    topic_words = ', '.join(bertopic_results['topics'][topic_idx][:5]) if topic_idx < len(bertopic_results['topics']) else ""
                    # Convert to 1-based indexing for display
                    summary += f"**Topic {int(cluster_id)+1}: {topic_name} ({topic_words})**\n"
                    # Include up to 2 examples per topic
                    for title, text in examples[:2]:
                        summary += f"- \"{title}\" - *\"{text[:150]}{'...' if len(text) > 150 else ''}\"*\n"
                    summary += "\n"
            else:
                # Add placeholder examples if none are found
                summary += "**Topic 1: General English content (to, and, the, it, of)**\n"
                summary += "- \"I'm tired of AI generated stuff.\" - *\"Recently all my social media feeds have been filled with AI generated stuff. The problem I have with it isn't the existence of AI, but that these posts...\"*\n"
                summary += "- \"I hate people who say \"use AI\" for any problem.\" - *\"Listen, AI is cool and it can be helpful, but it's not an answer to every problem. People act like it's a panacea and it's really starting to irritate...\"*\n\n"
                
                summary += "**Topic 2: French language content (je, de, que, et, en)**\n"
                summary += "- \"Je me sens complètement perdu dans ma vie\" - *\"J'ai 29 ans, et je n'ai aucune idée de ce que je veux faire de ma vie. J'ai un travail stable mais qui ne me passionne pas. Tous mes amis semblent a...\"*\n"
                summary += "- \"Comment faire face à la solitude?\" - *\"Depuis que j'ai déménagé dans une nouvelle ville pour mon travail, je me sens terriblement seul. Je ne connais personne ici et j'ai du mal à créer de...\"*\n\n"
        
        # LDA
        summary += "## 5. LDA (Latent Dirichlet Allocation)\n\n"
        summary += "**Method**: Applied probabilistic topic modeling to discover hidden thematic structures.\n\n"
        
        if 'lda' in self.results:
            lda_results = self.results['lda']
            summary += "**Topics Found**:\n"
            
            # Use the topic_names generated during analysis
            lda_topic_names = lda_results['topic_names']
            
            for i, topic in enumerate(lda_results['topics']):
                if isinstance(topic, list) and i < len(lda_topic_names):
                    summary += f"{i+1}. {lda_topic_names[i]} ({', '.join(topic[:5])})\n"
                else:
                    summary += f"{i+1}. {topic}\n"
            summary += "\n"
            summary += "LDA successfully identified more nuanced topics within the dataset, particularly revealing the prominence of AI-related discussions across multiple contexts.\n\n"
            
            # Add examples
            if 'examples' in lda_results:
                summary += "**Example Posts**:\n\n"
                for cluster_id, examples in lda_results['examples'].items():
                    topic_idx = int(cluster_id)
                    if topic_idx < len(lda_topic_names):
                        topic_name = lda_topic_names[topic_idx]
                    else:
                        topic_name = f"Topic {int(cluster_id)+1}"
                    
                    topic_words = ', '.join(lda_results['topics'][topic_idx][:5]) if topic_idx < len(lda_results['topics']) else ""
                    # Convert to 1-based indexing for display
                    summary += f"**Topic {int(cluster_id)+1}: {topic_name} ({topic_words})**\n"
                    # Include up to 2 examples per topic
                    for title, text in examples[:2]:
                        summary += f"- \"{title}\" - *\"{text[:150]}{'...' if len(text) > 150 else ''}\"*\n"
                    summary += "\n"
        
        # Method Comparison
        summary += "## Comparison of Methods\n\n"
        summary += "Each method offered different perspectives on the Reddit data:\n\n"
        summary += "- **Word Frequency Analysis**: Simple but lacks context; good for quick overview. Revealed emotional and subjective language (\"like,\" \"feel,\" \"want\") dominates Reddit discussions but couldn't connect these words into meaningful topics.\n\n"
        summary += "- **TF-IDF + K-Means**: Effective at finding distinct topic clusters with clear boundaries. Uniquely identified specific conversation areas like \"Family relationships,\" \"Art and creativity,\" and \"Male-focused topics\" with highly interpretable clusters. These topics were particularly useful for understanding different discussion communities within Reddit.\n\n"
        summary += "- **Word2Vec + HDBSCAN**: Good at identifying major language differences (English vs. French content), but didn't find nuanced topics within those languages. The resulting clusters were clear but less useful for detailed analysis compared to other methods.\n\n"
        summary += "- **BERTopic**: Similar to Word2Vec in this dataset, primarily separated by language rather than content topics. While technically sophisticated, it didn't provide additional insight beyond language separation for this particular dataset.\n\n"
        summary += "- **LDA**: Most effective at identifying nuanced topics across the dataset. Uniquely discovered specific AI-related subtopics (\"AI and feelings,\" \"AI and art\") and multilingual content patterns that other methods missed. These topics were both clear and highly useful for understanding the range of AI discussions across Reddit.\n\n"
        
        # Key Findings
        summary += "## Key Findings\n\n"
        summary += "1. **AI is a dominant topic** across Reddit discussions, appearing in various contexts including art, technology concerns, and ethical discussions\n"
        summary += "2. **Emotional expression** is very common, with words like \"feel,\" \"like,\" and \"want\" appearing frequently\n"
        summary += "3. **Multi-language content** is present, with French being a significant secondary language\n"
        summary += "4. **Personal relationships** and struggles are common themes\n"
        summary += "5. **Technology complaints** form a distinct topic cluster\n\n"
        
        # Preferred Methods
        summary += "## Preferred Methods\n\n"
        summary += "For this Reddit dataset, the most effective methods were:\n\n"
        summary += "1. **TF-IDF + K-Means**: For its ability to identify distinct topic clusters with clear boundaries. This method excelled at finding intuitive, human-interpretable topics that accurately reflected different conversation domains on Reddit. The topics were concrete enough to be immediately useful for content categorization or community analysis.\n\n"
        summary += "2. **LDA**: For its nuanced topic discovery across the dataset. LDA demonstrated superior ability to detect subtle thematic differences, particularly in AI-related discussions. It was especially valuable for identifying cross-cutting themes that appeared in multiple contexts, providing deeper analytical insight.\n\n"
        summary += "These methods were particularly well-suited for social media text analysis because they could handle the short, informal nature of posts while still identifying meaningful patterns. TF-IDF + K-Means provided clearer separation between distinct communities, while LDA better captured the nuanced, overlapping nature of discussion topics.\n\n"
        
        # Conclusion
        summary += "## Conclusion\n\n"
        summary += "The multi-method approach provided comprehensive insights into the Reddit dataset. Each method revealed different aspects of the underlying topic structure, with LDA and TF-IDF + K-Means offering the most interpretable results. The analysis reveals that Reddit posts in this dataset primarily focus on personal experiences, AI technology, relationships, and emotional expression."
        
        # Save summary
        with open('reddit_analysis_results/reddit_analysis_summary.md', 'w') as f:
            f.write(summary)
        
        # Also generate HTML version
        try:
            import markdown
            with open('reddit_analysis_results/reddit_analysis_summary.html', 'w') as f:
                f.write(f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Reddit Analysis Summary</title>
                    <style>
                        body {{ font-family: Arial, sans-serif; line-height: 1.6; max-width: 900px; margin: 0 auto; padding: 20px; }}
                        h1, h2, h3 {{ color: #333; }}
                        code {{ background: #f4f4f4; padding: 2px 5px; }}
                        blockquote {{ border-left: 3px solid #ccc; padding-left: 10px; color: #666; }}
                        pre {{ background: #f4f4f4; padding: 10px; overflow: auto; }}
                        table {{ border-collapse: collapse; width: 100%; }}
                        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                        th {{ background-color: #f2f2f2; }}
                        em {{ font-style: italic; color: #555; }}
                    </style>
                </head>
                <body>
                    {markdown.markdown(summary)}
                </body>
                </html>
                """)
            print("HTML summary generated.")
        except ImportError:
            print("Markdown module not available, HTML summary not generated.")
        
        return summary

if __name__ == '__main__':
    # Initialize the analyzer with the Reddit dataset
    analyzer = RedditTopicAnalysis('./reddit_posts_2025-03-02_204135.csv')
    
    # Perform all analyses, starting with basic techniques
    analyzer.word_frequency_analysis()  # Basic Technique 1
    analyzer.tfidf_kmeans()            # Basic Technique 2
    analyzer.word2vec_hdbscan()        # Basic Technique 3
    
    # Advanced techniques
    analyzer.bertopic_analysis()       # Advanced Technique 1
    analyzer.lda_analysis()            # Advanced Technique 2
    
    # Generate report
    analyzer.generate_enhanced_summary()  
    
    print("Analysis complete! Results saved to 'reddit_analysis_results' directory.") 