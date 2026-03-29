from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Step 1: Sample documents
documents = [
    "I love cricket and football",
    "Politics is about government and elections",
    "Football and cricket are popular sports",
    "Government policies affect the economy"
]

# Step 2: Convert text to numbers
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(documents)

# Step 3: Apply LDA
lda = LatentDirichletAllocation(n_components=2, random_state=0)
lda.fit(X)

# Step 4: Print topics
words = vectorizer.get_feature_names_out()

for i, topic in enumerate(lda.components_):
    print(f"Topic {i+1}:")
    print([words[j] for j in topic.argsort()[-5:]])
