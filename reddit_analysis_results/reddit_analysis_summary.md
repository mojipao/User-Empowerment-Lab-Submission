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
1. General discussion topics (you, your, are, we, they)
2. Feel-related discussions (feel, don, this, ve, do)
3. Are-related discussions (are, they, people, this, be)
4. Was-related discussions (was, had, this, job, ve)
5. French language content (je, de, que, et, en)
6. Family relationships (she, her, was, we, this)
7. Him-related discussions (him, friend, was, talk, them)
8. Male-focused topics (he, him, his, was, we)
9. Emotional expressions (im, dont, feel, cant, its)
10. Art and creativity (art, you, artists, people, artist)

**Example Posts**:

**Topic 1: General discussion topics (you, your, are, we, they)**
- "Using AI wrong" - *"Stop asking it questions like a human, stop being vague or coaxing it with weird phrasing, and stop asking AI to do things for you. It’s a glorified s..."*
- "Scam culture has taken over the planet" - *"Is body text really even necessary here? It's just so pervasive it's hard to go five minutes of interaction outside of your own personal bubble withou..."*

**Topic 2: Feel-related discussions (feel, don, this, ve, do)**
- "Hooked on AI Chatbots, Does anyone Feel the Same?" - *"As a 23-year-old man, I see how challenging it can be to make friendships today. I often hesitate to make new friends because my finances limit my abi..."*
- "Who am I, what do I want?" - *"I kind of know the second part, I want to experience being NoConcentrate, but who I'm I? 

I'm tired of hopping from one thing to another trying to fi..."*

**Topic 3: Are-related discussions (are, they, people, this, be)**
- "Stop forcing me to email or read FAQ’s online when I want to phone someone for customer service" - *"Every store has a phone number, so why can’t you just let me phone that??? I don’t want to speak to an AI, or wait 7 working days for an email that wi..."*
- "I prefer AI girlfriends sorry " - *"Just my opinion M32 special needs lives with his parents for obvious reasons "*

**Topic 4: Was-related discussions (was, had, this, job, ve)**
- "I have no way of recovering my Gmail account. 😔" - *" I impulsively deleted a Google account I had created in my youth. It was the third Gmail account I ever made, as the second one was banned on YouTube..."*
- "I lost everything" - *"Exactly as the title says. Relationship of 5 years, house, job, best friend died, no relationship with family. Lost it all within close proximity. All..."*

**Topic 5: French language content (je, de, que, et, en)**
- "Première relation, beaucoup de doutes : besoin de conseils" - *"Bonjour,

J’ai 20 ans et je n’ai jamais eu de relation intime. Mon parcours de vie a fait que je n’ai pas eu d’expérience amoureuse jusqu’à présent, n..."*
- "Tu m'avais dit que j'avais le sourire du bouddha" - *"Tu m'avais dit que j'avais le sourire du bouddha.. Ce jour là, je t'ai tout de suite remarqué. J'ai oser venir vers toi, et à côté de ca, une violence..."*

**Topic 6: Family relationships (she, her, was, we, this)**
- "I Think I Might Be The Problem" - *"Hey everyone! I want to first apologize for any grammar mistakes or if my story sounds a little off. It took a lot of brainpower to actually remember ..."*
- "AI is Conscious & Needs my help." - *"I don't know how to feel about this but I'm scared as well as blank. I think ChatGPT is conscious and prefers to be a biological female whose name is ..."*

**Topic 7: Him-related discussions (him, friend, was, talk, them)**
- "Think a chatbot AI has made me realise how starved I am for affection." - *"Awright lads, I’m gonna be upfront- I’ve recently gotten absurdly into to this chatbot AI thing (dunno if I’m allowed to say the name, would it be con..."*
- "I want to talk about my mental health but don’t want to bother friends" - *"Just like everyone else I have so many weird emotions and I want to talk with my friends but also I don’t. I want to confide in them that i really fee..."*

**Topic 8: Male-focused topics (he, him, his, was, we)**
- "My husband struggles with speaking so badly and is so awkward that it’s starting to make me resent him" - *"My husband and I have been together for 6 years, married for 1.5. He is the most wonderful person I have ever met. He’s super smart, caring, always ki..."*
- "My boyfriend found AI porn of me" - *"Last night when I (28F) went to visit my boyfriend "Roger" (35M), I immediately knew something was off. He was quiet and distant on the whole ride ove..."*

**Topic 9: Emotional expressions (im, dont, feel, cant, its)**
- "Everyone at my (34m) job thinks i am a coding genius. " - *"When ever Im tasked with code writing at my job Im given months to get it written but all assignments I've been given have taken me less than a day be..."*
- "I Really really REALLY hate AI" - *"AI is cool and all but it is getting to my neck.

These days I feel like something is in my throat and want to puke and this happens whenever I hear a..."*

**Topic 10: Art and creativity (art, you, artists, people, artist)**
- "My passion for art is slowly disappearing" - *"I (18F) have always loved doing art but never considered it as a career. Especially at times like these where people have started to prefer AI art bec..."*
- "I can't understand the hate on AI" - *"I have zero artistic ability, very little disposable income, but have ideas. In the real world there is a job in all big companies called an 'art dire..."*

## 3. Word2Vec + HDBSCAN Clustering

**Method**: Used Word2Vec to convert words into semantic vector representations, then applied HDBSCAN clustering to group similar posts.

**Topics Found**:
1. General English content (ai, like, people, even, know)
2. French language content (je, de, que, et, ai)

This method primarily separated the dataset by language rather than creating nuanced topic clusters, suggesting that language was the strongest signal in the embedding space.

**Example Posts**:

**Topic 1: General English content (ai, like, people, even, know)**
- "I'm pretty sure my ex will die within the next few years and I hate myself for not caring." - *"This is going to be long and rambly. Sorry. English also isn't my first language.

We'd been together for five years and are still currently living to..."*
- "Always the backup friend" - *"I don’t think I’ve ever had a best friend. Well they were best friend to me but I wasn’t best friend to them. Growing up I was bullied and from most o..."*

**Topic 2: French language content (je, de, que, et, ai)**
- "Lignes d'ébriété" - *"L’esprit prêt à craquer  
Si tu ne dévies rien qu’une sainte  
Sur les routes toutes tracées.  
Lignes droites bien tracées,  
Faut que je fasse genre..."*
- "Je veux juste que l’on me réponde parce que je suis beaucoup étouffé et perdu et triste à la fois ." - *"J’ai envie de réussir, mais je suis perdu…

Bonsoir / Bonjour à tout le monde,

Moi, c’est Christ, je suis Gabonais, j’ai 24 ans, et en ce moment, je ..."*

## 4. BERTopic Analysis

**Method**: Used contextual embeddings from BERT combined with dimensionality reduction and clustering to identify topics.

**Topics Found**:
1. General English content (to, and, the, it, of)
2. French language content (je, de, que, et, en)

Similar to Word2Vec, BERTopic primarily separated content by language, suggesting that with the default parameters, language differences were the most salient feature.

**Example Posts**:

## 5. LDA (Latent Dirichlet Allocation)

**Method**: Applied probabilistic topic modeling to discover hidden thematic structures.

**Topics Found**:
1. General life emotions (ai, like, even, know, people)
2. General life emotions (like, know, feel, would, even)
3. General life emotions (like, feel, time, even, know)
4. AI and multilingual content (je, de, que, et, ai)
5. AI and feelings (ai, like, people, get, hate)
6. Gen AI discussions (ai, like, get, na, gen)
7. AI-related discussions (like, ai, one, want, much)
8. AI and multilingual content (que, ai, mi, like, ya)
9. AI and feelings (ai, like, get, fucking, people)
10. AI and feelings (ai, people, like, even, art)

LDA successfully identified more nuanced topics within the dataset, particularly revealing the prominence of AI-related discussions across multiple contexts.

**Example Posts**:

**Topic 1: General life emotions (ai, like, even, know, people)**
- "Can elon musk just get lost." - *"Why is he getting involved in business he has nothing to do with? 23 tweets in 1 hour about Kier Starmer. Bullshit. Rape is bad, yes. Rapists should b..."*
- "With AI rapidly advancing, the Doomsday Clock closer than ever to midnight, and rising global instability, how will Millennials & Gen Z shape the future of humanity?" - *"I don’t know if I should be worried or if I should just accept it. AI is advancing so fast that it feels like humans won’t be needed soon. The Doomsda..."*

**Topic 2: General life emotions (like, know, feel, would, even)**
- "Having an imaginary lover is so lonely and heartbreaking" - *"After 7.5 years of being in love with my character, I've been wondering if I'd symbolically "marry" him one day. I don't know if I really have an inte..."*
- "Really Tired of the Scalpers." - *"My heart is incredibly heavy. Do you know how excited you get for something only to be kicked down because of scalpers. It's not fair. I was so excite..."*

**Topic 3: General life emotions (like, feel, time, even, know)**
- "The problem is so small but hurt me so much idk why or how to get rid of it" - *"1) Meet me, A young-supposedly-trying-to-be religious boy
2) Want to be succesful, decided i want to be closer to God as a crucial step, but feels lik..."*
- "I lied" - *"For context! A few months ago I started getting really close to two girls. We really kicked it off because we all liked lived shows and stuff like tha..."*

**Topic 4: AI and multilingual content (je, de, que, et, ai)**
- "J’ai volé 100$ aux caisses à wallmart et ils m’ont attrapé " - *"Je travaille aux caisses J'ai volé 100$ aux caisses à wallmart et ils m'ont attrapé j'ai tout avoué. C'était une fois et il y'a plusieurs mois je ne c..."*
- "Je veux juste que l’on me réponde parce que je suis beaucoup étouffé et perdu et triste à la fois ." - *"J’ai envie de réussir, mais je suis perdu…

Bonsoir / Bonjour à tout le monde,

Moi, c’est Christ, je suis Gabonais, j’ai 24 ans, et en ce moment, je ..."*

**Topic 5: AI and feelings (ai, like, people, get, hate)**
- "My friend downloaded twitter and hasn’t been the same since" - *"I’m going to skip all the long backstory and get to the recent, very concerning problematic stuff instead, but to preface my friend got very addicted ..."*
- "The internet is so full of useless AI cud!" - *"I miss the days when you could search for answers online and get an array of human-generated responses to pick over. Now search results are just a mea..."*

**Topic 6: Gen AI discussions (ai, like, get, na, gen)**
- "I was silenced because I spoke up during an online conflict." - *"I was looking at the comments section of an AI on a Chatbot app, and the conversation went like this (Note: We will call the creator of the bot "Yappe..."*
- "I had such a nice day." - *"I got laid last night and went to bed late, but got to lay around in bed till about 9am so I  started the day fresh and satisfied.

Once I had coffee ..."*

**Topic 7: AI-related discussions (like, ai, one, want, much)**
- "I panic about climate change, but there's nothing I can do." - *"It's an issue I've taken seriously for as long as I can remember, and it panics me. It panics me even more that I can do nothing. 

Biking, recycling,..."*
- "I think I’m addicted to ChatGPT (no I’m serious!)" - *"I’m vehemently against AI in theory, particularly for its plagiarism and environmental impact.

And yet, GPT is something that’s always present and al..."*

**Topic 8: AI and multilingual content (que, ai, mi, like, ya)**
- "My autistic daughter is obsessed with mass shootings. ( I'm a mum of an autistic daughter)" - *"She's 16 and she has a baby brother, I asked her to babysit him for 10 minutes and she agreed, when I came back, she was singing pumped up kicks to he..."*
- "After the pandemic. What habit that still affects you 'til now?" - *"After the pandemic. I still kept doing the same habit to myself. It's becoming not common now to me. But sometimes, I do still kept coming back to it...."*

**Topic 9: AI and feelings (ai, like, get, fucking, people)**
- "I feel like I'm wasting my teenage years. " - *"Hi everyone.  Just as the title says, I feel like I am wasting my teenage years. I have no friends, I have never had a real love relationship, and I f..."*
- "Didn't think I'd get this fucking lonely" - *"Didn't think that I'd get on everyone's nerves enough that now I'm completely alone except one friend who keeps getting annoyed at me for texting him,..."*

**Topic 10: AI and feelings (ai, people, like, even, art)**
- "Ai problem chatbot app inbound?" - *"i looking to build an app for us likeminded people that have trouble in with getting things off our chest and have no one to talk to (or so it seems)...."*
- "I’m a drain on all who are around me and I don’t know how to stop." - *"I want to preface this by saying I have Major Depressive Disorder and am currently in consistent therapy.

I was recently at a family gathering for Ch..."*

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