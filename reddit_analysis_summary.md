# Reddit Topic Analysis Summary

## Method Summaries

### 1. Word Frequency Analysis

**Method Description**: Simple counting and visualization of word occurrences across all Reddit posts.

**Topics Found**:
- Common emotional expressions: "like", "feel", "want"
- Personal references: "people", "know", "get"
- Time and quantity references: "one", "time"

**Example Posts**:
- "I feel a disconnect with the society. People dont want to listen to me, Is it wrong if I turn to tech to help me out?"
- "I hate my phone's 'suggestions.'"
- "After the pandemic. What habit that still affects you 'til now?"

### 2. TF-IDF + K-Means Clustering

**Method Description**: Uses TF-IDF to weight words based on importance in documents versus the corpus, then applies K-Means to group similar posts.

**Topics Found**:
- General discussion (Keywords: you, your, are, we, they)
- Emotional expression (Keywords: feel, don't, this, know)
- Social commentary (Keywords: are, they, people, this, be)
- Work/career related (Keywords: was, had, this, job, been)
- French language content (Keywords: je, de, que, et, en)
- Family relationships (Keywords: she, her, was, we, mom)
- Friendships (Keywords: him, friend, was, talk, them)
- Male-focused topics (Keywords: he, him, his, was, we)
- Personal struggles (Keywords: im, dont, feel, cant, its)
- Art and creativity (Keywords: art, you, artists, people, artist)

**Example Posts**:
- "I'm sick of everyone's advice on my resume" (General discussion)
- "I am completely exhausted by life and society." (Emotional expression)
- "What's the deal with AI 'enhanced' photos?" (Art/Technology topic)

### 3. Word2Vec + HDBSCAN Clustering

**Method Description**: Uses word embeddings to capture semantic relationships, then applies density-based clustering to find natural groupings.

**Topics Found**:
- English language content (Keywords: ai, like, people, even, know)
- French language content (Keywords: je, de, que, et, ai)

**Example Posts**:
- "I'm sick of digital currency making people rich by playing on computers while The poor get poorer working everyday."
- "UI/UX Design Decline"
- "Mon copain n'est plus du tout sûr de ses sentiments et j'en souffre énormément !!"

### 4. BERTopic Analysis

**Method Description**: Advanced method using contextual embeddings from BERT with dimensionality reduction and clustering.

**Topics Found**:
- English language content (Keywords: to, and, the, it, of)
- French language content (Keywords: je, de, que, et, en)

**Example Posts**:
- Various English posts covering technology, relationships, and personal reflections
- French language posts with similar diverse topics

### 5. Latent Dirichlet Allocation (LDA)

**Method Description**: Probabilistic topic modeling that represents documents as mixtures of topics and topics as distributions over words.

**Topics Found**:
- General emotions (Keywords: like, know, want, even, get)
- Work and AI (Keywords: work, like, time, ai, even)
- AI and multilingual content (Keywords: ai, like, que, even, people)
- AI discussions (Keywords: ai, like, people, get, feel)
- AI governance (Keywords: gen, ai, banned, get, like)
- AI and art (Keywords: ai, people, art, like, get)
- Life reflections (Keywords: like, even, feel, would, ai)
- French content (Keywords: je, de, que, et, ai)
- Social relationships (Keywords: like, ai, feel, know, friends)
- Technology complaints (Keywords: ai, like, fucking, hate, get)

**Example Posts**:
- "Electronics/gadgets have reached an average consumer standstill"
- "No, I don't bleeping want Siri…"
- "AI Text Detectors are gonna be the death of me."

## Final Comparison and Analysis

### Topics Discovered by Each Method

- **Word Frequency Analysis**: Identified common terms but not topics; showed prevalence of emotional language and personal references
- **TF-IDF + K-Means**: Discovered 10 distinct topics including general discussions, emotional expression, social commentary, work-related issues, personal relationships, and French content
- **Word2Vec + HDBSCAN**: Found only 2 major clusters primarily differentiated by language (English vs. French)
- **BERTopic**: Similar to Word2Vec, detected 2 clusters mainly separated by language
- **LDA**: Discovered 10 nuanced topics, particularly effective at finding AI-related discussions across different contexts (art, technology, ethics, governance)

### Clarity and Usefulness of Topics

- **Word Frequency Analysis**: Clear but limited usefulness - gives basic vocabulary overview without context
- **TF-IDF + K-Means**: High clarity with distinct boundaries between topics; useful for identifying major themes including language differences
- **Word2Vec + HDBSCAN**: Low thematic clarity but useful for language identification; didn't capture nuanced topics with default parameters
- **BERTopic**: Similar to Word2Vec in this dataset; didn't provide expected granularity despite being an advanced method
- **LDA**: High clarity and interpretability; very useful for identifying relationships between topics and understanding AI discussions across different contexts

### Preferred Methods

1. **LDA (Latent Dirichlet Allocation)** is preferred for its ability to:
   - Discover nuanced topics with clear semantic coherence
   - Show how AI discussions span multiple contexts (art, technology, ethics)
   - Provide interpretable word groups that make intuitive sense
   - Reveal relationships between topics

2. **TF-IDF + K-Means** is also highly effective for:
   - Creating clear topic boundaries with distinct separation
   - Computational efficiency compared to more advanced methods
   - Identifying major thematic clusters including language differences
   - Finding distinctive terms for each topic

The simplest method (Word Frequency Analysis) provided a useful foundation, while the neural embedding methods (Word2Vec and BERTopic) would likely require parameter tuning to reach their full potential with this dataset.

Overall, the multi-method approach was most valuable, with each technique revealing different aspects of the Reddit discussions. 