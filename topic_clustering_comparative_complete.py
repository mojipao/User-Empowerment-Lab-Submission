
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation
from gensim.models import Word2Vec
from gensim.corpora import Dictionary
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from bertopic import BERTopic
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from collections import Counter

nltk.download('stopwords')
nltk.download('punkt')

# Load data
df = pd.read_csv("reddit_posts_2025-03-02_204135.csv")
df["text"] = df["text"].astype(str)

# Preprocess function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\\S+|[^a-z\\s]", "", text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in stopwords.words("english")]
    return " ".join(tokens)


df["clean_text"] = df["text"].apply(clean_text)

## Count word frequencies
word_counts = Counter(" ".join(df["clean_text"]).split())

if word_counts:
    wordcloud = WordCloud(width=800, height=400).generate_from_frequencies(word_counts)
    wordcloud.to_file("wordcloud.png")
    print("✅ Word cloud saved as wordcloud.png")
else:
    print("⚠️ Word cloud not generated: no words to draw.")



# TF-IDF + KMeans
tfidf = TfidfVectorizer(max_features=1000)
X_tfidf = tfidf.fit_transform(df["clean_text"])
kmeans = KMeans(n_clusters=5, random_state=42)
df["kmeans_cluster"] = kmeans.fit_predict(X_tfidf)

# Word2Vec + HDBSCAN
df["tokens"] = df["clean_text"].apply(lambda x: x.split())
w2v_model = Word2Vec(sentences=df["tokens"], vector_size=100, window=5, min_count=2, workers=4)

def get_mean_vector(tokens):
    vectors = [w2v_model.wv[token] for token in tokens if token in w2v_model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(100)

df["vector"] = df["tokens"].apply(get_mean_vector)
vectors = np.stack(df["vector"].values)
hdbscan = HDBSCAN(min_cluster_size=10)
df["hdbscan_cluster"] = hdbscan.fit_predict(vectors)

# BERTopic
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
bertopic_model = BERTopic(embedding_model=embedding_model)
topics, _ = bertopic_model.fit_transform(df["clean_text"])
df["bertopic_cluster"] = topics

# LDA
count_vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words="english")
X_count = count_vectorizer.fit_transform(df["clean_text"])
lda = LatentDirichletAllocation(n_components=5, random_state=42)
lda.fit(X_count)

print("LDA Topics:")
terms = count_vectorizer.get_feature_names_out()
for idx, topic in enumerate(lda.components_):
    print(f"Topic {idx}: ", [terms[i] for i in topic.argsort()[:-6:-1]])

# Save clustered data
df.to_csv("clustered_reddit_posts.csv", index=False)
print("✅ All methods complete. Outputs saved.")
