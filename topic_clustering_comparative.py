import pandas as pd
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
import umap
import hdbscan
from gensim.models import Word2Vec, LdaMulticore
from gensim.corpora import Dictionary
from gensim.utils import simple_preprocess
import nltk
from nltk.tokenize import word_tokenize
import numpy as np
from transformers import pipeline

# nltk.download('punkt')
# nltk.download('stopwords')

# Data preprocessing function
def preprocess(text):
    from nltk.corpus import stopwords
    tokens = [w for w in simple_preprocess(text) if w not in stopwords.words('english')]
    return tokens

# Local LLM-based summarization function using Hugging Face Transformers
def summarize_topics_hf(topic_texts):
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    combined_text = " ".join(topic_texts)
    summary = summarizer(combined_text, max_length=150, min_length=50, do_sample=False)
    return summary[0]['summary_text']

# Local LLM-based summarization function using DeepSeek
def summarize_topics_deepseek(topic_texts):
    summarizer = pipeline("text-generation", model="deepseek-ai/deepseek-coder-6.7b-instruct")
    combined_text = " ".join(topic_texts)
    prompt = f"Summarize the following topics:\n{combined_text}\nSummary:"
    summary = summarizer(prompt, max_new_tokens=150, do_sample=False)
    return summary[0]['generated_text'].split('Summary:')[-1].strip()

if __name__ == '__main__':
    output_log = []

    df = pd.read_csv('/Users/marxw/Documents/MW-001/Downloads/AI-Teen/data/reddit/reddit_posts_2025-03-02_204135.csv')
    texts = (df['text'].fillna('') + ' ' + df['title'].fillna('')).tolist()
    tokenized_texts = [preprocess(text) for text in texts]

    # BERT Embeddings and clustering
    bert_model = SentenceTransformer('all-MiniLM-L6-v2')
    bert_embeddings = bert_model.encode(texts)

    topic_model_bert = BERTopic(
        embedding_model=bert_model,
        umap_model=umap.UMAP(n_neighbors=15, min_dist=0.1),
        hdbscan_model=hdbscan.HDBSCAN(min_cluster_size=5, prediction_data=True),
        calculate_probabilities=True
    )
    bert_topics, bert_probs = topic_model_bert.fit_transform(texts)
    bert_topic_list = topic_model_bert.get_topic_info()
    output_log.append(f"BERT topics:\n{bert_topic_list}\n")

    # Word2Vec and clustering
    w2v_model = Word2Vec(sentences=tokenized_texts, vector_size=100, window=5, min_count=2, workers=4)
    w2v_embeddings = [
        np.mean([w2v_model.wv[w] for w in tokens if w in w2v_model.wv] or [np.zeros(100)], axis=0)
        for tokens in tokenized_texts
    ]

    clusterer_w2v = hdbscan.HDBSCAN(min_cluster_size=5, prediction_data=True)
    w2v_clusters = clusterer_w2v.fit_predict(w2v_embeddings)
    output_log.append(f"Word2Vec clusters:\n{set(w2v_clusters)}\n")

    # LDA clustering
    dictionary = Dictionary(tokenized_texts)
    corpus = [dictionary.doc2bow(text) for text in tokenized_texts]

    lda_model = LdaMulticore(corpus=corpus, id2word=dictionary, num_topics=10, passes=10, workers=4)
    lda_topics = lda_model.print_topics(num_words=10)
    output_log.append(f"LDA topics:\n{lda_topics}\n")

    # Local LLM thematic summarization using Hugging Face
    combined_topics = [topic for _, topic in lda_topics]
    thematic_summary_hf = summarize_topics_hf(combined_topics)
    output_log.append(f"Hugging Face LLM-based thematic summary:\n{thematic_summary_hf}\n")

    # Local LLM thematic summarization using DeepSeek
    thematic_summary_deepseek = summarize_topics_deepseek(combined_topics)
    output_log.append(f"DeepSeek LLM-based thematic summary:\n{thematic_summary_deepseek}\n")

    # Save output log to document
    with open('clustering_summary_log.txt', 'w') as file:
        file.write("\n".join(output_log))

    # # Combined results
    # results_df = pd.DataFrame({
    #     'text': texts,
    #     'BERT_topic': bert_topics,
    #     'Word2Vec_topic': clusterer_w2v.labels_,
    #     'LDA_topic': [max(lda_model[doc], key=lambda x: x[1])[0] for doc in corpus]
    # })
    # results_df.to_csv('topic_comparison_results.csv', index=False)