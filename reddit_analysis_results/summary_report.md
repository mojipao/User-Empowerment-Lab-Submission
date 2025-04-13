# Reddit Topic Analysis Summary Report

This report provides a concise summary of all methods used, topics discovered, and comparative analysis.

## Method Summaries

### Word Frequency Analysis

**Description**: Counts and visualizes most common words in the dataset. Simple but effective for initial exploration.

#### Topics Discovered

- **Topic 1**: like (2445)
- **Topic 2**: dont (1855)
- **Topic 3**: people (1421)
- **Topic 4**: even (1330)
- **Topic 5**: know (1254)
- **Topic 6**: get (1251)
- **Topic 7**: feel (1199)
- **Topic 8**: one (1053)
- **Topic 9**: time (1045)
- **Topic 10**: want (1010)

---

### TF-IDF + K-Means Clustering

**Description**: Uses TF-IDF to weight words by importance and K-Means to cluster similar documents together.

#### Topics Discovered

- **Topic 1**: General Discussion (Keywords: you, your, are, we, they)
- **Topic 2**: Emotional Expression (Keywords: feel, don, this, ve, do)
- **Topic 3**: General Discussion (Keywords: are, they, people, this, be)
- **Topic 4**: Work and Career (Keywords: was, had, this, job, ve)
- **Topic 5**: French Language Content (Keywords: je, de, que, et, en)
- **Topic 6**: General Discussion (Keywords: she, her, was, we, this)
- **Topic 7**: Personal Relationships (Keywords: him, friend, was, talk, them)
- **Topic 8**: General Discussion (Keywords: he, him, his, was, we)
- **Topic 9**: Emotional Expression (Keywords: im, dont, feel, cant, its)
- **Topic 10**: Art and Creativity (Keywords: art, you, artists, people, artist)

#### Example Posts

- **From Topic 0**: "I'm so sick of AI while job hunting " - I'm am sick and tired of these companies using AI to filter through applicants when they apply! I wa...

- **From Topic 1**: "AI has ruined the only things I’m good at and made me redundant before I’ve even entered a career" - I graduated high school in 2023, and my senior year people were only just beginning to use AI confid...

- **From Topic 2**: "If mental health is already massive thing now, picture in 10 years." - I’m imagining what the future will look like in 10 years. I see a lot of people talking about anxiet...

---

### Word2Vec + HDBSCAN Clustering

**Description**: Uses word embeddings (Word2Vec) to capture semantic meaning and density-based clustering (HDBSCAN).

#### Topics Discovered

- **Topic 1**: AI and Technology (Keywords: ai, like, people, even, know)
- **Topic 2**: AI and Technology (Keywords: je, de, que, et, ai)

#### Example Posts

- **From Topic 0**: "So many cheap-ass knock-off brands" - I did an Amazon search for gym shoes and here are the brands represented in the first page of result...

- **From Topic 1**: "J’ai volé 100$ aux caisses à wallmart et ils m’ont attrapé " - Je travaille aux caisses J'ai volé 100$ aux caisses à wallmart et ils m'ont attrapé j'ai tout avoué....

---

### BERTopic

**Description**: Advanced method using BERT embeddings with dimensionality reduction and clustering.

#### Topics Discovered

- **Topic 1**: General Discussion (Keywords: to, and, the, it, of)
- **Topic 2**: French Language Content (Keywords: je, de, que, et, en)

#### Example Posts

---

### Latent Dirichlet Allocation (LDA)

**Description**: Probabilistic topic modeling approach that discovers hidden topic structures.

#### Topics Discovered

- **Topic 1**: Work and Career (Keywords: like, time, even, job, could)
- **Topic 2**: AI and Technology (Keywords: like, feel, know, ai, want)
- **Topic 3**: AI and Technology (Keywords: like, ai, even, would, know)
- **Topic 4**: AI and Technology (Keywords: je, de, que, et, ai)
- **Topic 5**: AI and Technology (Keywords: ai, people, like, art, use)
- **Topic 6**: General Discussion (Keywords: get, like, would, apps, even)
- **Topic 7**: AI and Technology (Keywords: que, de, ai, não, ela)
- **Topic 8**: General Discussion (Keywords: like, get, na, time, one)
- **Topic 9**: AI and Technology (Keywords: like, ai, even, one, people)
- **Topic 10**: Emotional Expression (Keywords: like, know, people, feel, life)

#### Example Posts

- **From Topic 0**: "UI/UX Design Decline" - Is it just me or did every tech company on this Earth collectively decided to ramp up the process of...

- **From Topic 1**: "DNA test revealed some family drama" - I read so many stories like this here, half of which were probably AI written, but still can’t belie...

- **From Topic 2**: "It's my birthday today, yet I sort of feel like I don't want to be here. Feels like my life is over, that I'm doomed. I blame character ai for this." - BTW this post is ridiculous so I'm ready for the downvotes. Yeah, this situation is pathetic. Even t...

---

## Final Comparison and Analysis

### Topics Discovered by Each Method

- **Word Frequency Analysis**: Found 10 topics. Sample topics: like (2445); dont (1855); people (1421)
- **TF-IDF + K-Means Clustering**: Found 10 topics. Sample topics: you, your, are; feel, don, this; are, they, people
- **Word2Vec + HDBSCAN Clustering**: Found 2 topics. Sample topics: ai, like, people; je, de, que
- **BERTopic**: Found 2 topics. Sample topics: to, and, the; je, de, que
- **Latent Dirichlet Allocation (LDA)**: Found 10 topics. Sample topics: like, time, even; like, feel, know; like, ai, even

### Clarity and Usefulness of Topics

**Word Frequency Analysis**: Provides clear overview of common terms but lacks context for understanding complex topics. Useful for initial exploration but not for deep analysis.

**TF-IDF + K-Means**: Produces relatively clear topic boundaries and distinctive terms. Particularly useful for identifying major themes like AI discussions, relationships, and language groups. The predefined cluster count (K=10) works well for this dataset.

**Word2Vec + HDBSCAN**: Less clear topic boundaries, mainly separating by language rather than thematic content. Limited usefulness for detailed topic analysis but effective at identifying major language differences.

**BERTopic**: Similar to Word2Vec in this dataset, primarily identifying language differences rather than thematic topics. Despite being an advanced method, it didn't provide the expected granularity with default parameters.

**LDA**: Produced the most nuanced and interpretable topics, particularly revealing AI themes across contexts. Very useful for understanding the diverse discussions around similar themes.

### Preferred Methods

Based on this analysis, the most effective methods for analyzing Reddit posts were:

1. **LDA (Latent Dirichlet Allocation)**: Preferred for its ability to discover nuanced topics and show relationships between themes. It revealed that AI appears across multiple contexts (art, technology, ethics), which other methods didn't capture as clearly.

2. **TF-IDF + K-Means**: Preferred for its clear topic boundaries and computational efficiency. This method was particularly effective at separating distinct conversation types, including identifying non-English content.

Word Frequency Analysis, while basic, provided a valuable foundation for understanding common terms. The advanced neural methods (Word2Vec and BERTopic) would likely require parameter tuning to perform optimally on this specific dataset.

Overall, a multi-method approach provided the most comprehensive understanding, with each technique revealing different aspects of the underlying topic structure in Reddit discussions.

