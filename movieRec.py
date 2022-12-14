from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

text = ["James Kofi Kofi", "Kofi James James"]
cv = CountVectorizer()

countMatrix = cv.fit_transform(text)

print(countMatrix.toarray())
print("Hello World Bro!!!")

similarityScores = cosine_similarity(countMatrix)

print(similarityScores)
