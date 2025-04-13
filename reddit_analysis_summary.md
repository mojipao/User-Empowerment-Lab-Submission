# Reddit Post Analysis - Project Summary

This project analyzed Reddit posts using various text analysis and topic modeling techniques, from simple word frequency analysis to advanced methods like BERTopic and Latent Dirichlet Allocation (LDA).

## 1. Word Frequency Analysis

**Method**: Counted and visualized the most common words in the Reddit posts.

**Topics Found**:
- Most frequent words were "like," "dont," "people," "even," "know," "get," "feel," "one," "time," and "want"
- These common words suggest that Reddit posts often express personal opinions, feelings, and questions

**Visualizations**: Word clouds and bar charts showing the distribution of the most common terms.

## 2. TF-IDF + K-Means Clustering

**Method**: Used TF-IDF (Term Frequency-Inverse Document Frequency) to weight words based on their importance, then clustered posts using K-Means algorithm.

**Topics Found**:
1. General discussion topics (you, your, we, they, people)
2. Emotional expressions (feel, don't, know)
3. Social commentary (people, be, what, fucking)
4. Work/career related (job, been, at, all)
5. French language content (je, de, que, et)
6. Family relationships (she, mom, about)
7. Friendships (friend, friends, talk, them)
8. Male-focused topics (he, him, his)
9. Personal struggles (im, dont, feel, cant)
10. Art and creativity (art, artists, artist)

**Example Posts**: For each topic cluster, we identified representative posts that illustrate the theme.

## 3. Word2Vec + HDBSCAN Clustering

**Method**: Used Word2Vec to convert words into semantic vector representations, then applied HDBSCAN clustering to group similar posts.

**Topics Found**:
1. General English content (ai, like, people, feel, know, etc.)
2. French language content (je, de, que, et, ai, en, pas, etc.)

This method primarily separated the dataset by language rather than creating nuanced topic clusters, suggesting that language was the strongest signal in the embedding space.

## 4. BERTopic Analysis

**Method**: Used contextual embeddings from BERT combined with dimensionality reduction and clustering to identify topics.

**Topics Found**:
1. General English content
2. French language content

Similar to Word2Vec, BERTopic primarily separated content by language, suggesting that with the default parameters, language differences were the most salient feature.

## 5. LDA (Latent Dirichlet Allocation)

**Method**: Applied probabilistic topic modeling to discover hidden thematic structures.

**Topics Found**:
1. General life emotions (like, know, want, life, feel)
2. Work and AI (work, ai, job, time)
3. AI and multilingual content (ai, que, de)
4. AI and feelings (ai, people, feel, think)
5. Gen AI discussions (gen, ai, alpha, banned)
6. AI and art (ai, art, people, make)
7. Feelings and life (feel, life, would, time)
8. French content (je, de, que, et)
9. Social relationships (friends, people, time)
10. Technology complaints (ai, fucking, hate, tech)

LDA successfully identified more nuanced topics within the dataset, particularly revealing the prominence of AI-related discussions across multiple contexts.

## Comparison of Methods

Each method offered different perspectives on the Reddit data:

- **Word Frequency Analysis**: Simple but lacks context; good for quick overview
- **TF-IDF + K-Means**: Effective at finding distinct topic clusters with clear boundaries
- **Word2Vec + HDBSCAN**: Good at identifying major language differences, but didn't find nuanced topics
- **BERTopic**: Similar to Word2Vec in this dataset, primarily separated by language
- **LDA**: Most effective at identifying nuanced topics across the dataset

## Key Findings

1. **AI is a dominant topic** across Reddit discussions, appearing in various contexts including art, technology concerns, and ethical discussions
2. **Emotional expression** is very common, with words like "feel," "like," and "want" appearing frequently
3. **Multi-language content** is present, with French being a significant secondary language
4. **Personal relationships** and struggles are common themes
5. **Technology complaints** form a distinct topic cluster

## Preferred Methods

For this Reddit dataset, the most effective methods were:

1. **TF-IDF + K-Means**: For its ability to identify distinct topic clusters
2. **LDA**: For its nuanced topic discovery across the dataset

These methods were particularly well-suited for social media text analysis because they could handle the short, informal nature of posts while still identifying meaningful patterns.

## Conclusion

The multi-method approach provided comprehensive insights into the Reddit dataset. Each method revealed different aspects of the underlying topic structure, with LDA and TF-IDF + K-Means offering the most interpretable results. The analysis reveals that Reddit posts in this dataset primarily focus on personal experiences, AI technology, relationships, and emotional expression. 