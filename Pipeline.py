import nltk 
import string
from nltk.tokenize import word_tokenize
from nltk import FreqDist
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt


def quitarStopWords_eng(tokens):

    stop_words = set(stopwords.words('english'))

    tokens_limpios = [
        palabra.lower()
        for palabra in tokens
        if palabra.lower() not in stop_words and palabra.isalpha()
    ]

    return tokens_limpios

def lematizar(tokens):

    lemmatizer = WordNetLemmatizer()

    tokens_lematizados = [
        lemmatizer.lemmatize(palabra)
        for palabra in tokens
    ]

    return tokens_lematizados

corpus = [

lematizar(quitarStopWords_eng(word_tokenize(
"Python is an interpreted and high-level language, while CPlus is a compiled and low-level language."
))),

lematizar(quitarStopWords_eng(word_tokenize(
"JavaScript runs in web browsers, while Python is used in various applications, including data science and artificial intelligence."
))),

lematizar(quitarStopWords_eng(word_tokenize(
"JavaScript is dynamically and weakly typed, while Rust is statically typed and ensures greater data security."
))),

lematizar(quitarStopWords_eng(word_tokenize(
"Python and JavaScript are interpreted languages, while Java, CPlus, and Rust require compilation before execution."
))),

lematizar(quitarStopWords_eng(word_tokenize(
"JavaScript is widely used in web development, while Go is ideal for servers and cloud applications."
))),

lematizar(quitarStopWords_eng(word_tokenize(
"Python is slower than CPlus and Rust due to its interpreted nature."
))),

lematizar(quitarStopWords_eng(word_tokenize(
"JavaScript has a strong ecosystem with Node.js for backend development, while Python is widely used in data science."
))),

lematizar(quitarStopWords_eng(word_tokenize(
"JavaScript does not require compilation, while CPlus and Rust require code compilation before execution."
))),

lematizar(quitarStopWords_eng(word_tokenize(
"Python and JavaScript have large communities and an extensive number of available libraries."
))),

lematizar(quitarStopWords_eng(word_tokenize(
"Python is ideal for beginners, while Rust and CPlus are more suitable for experienced programmers."
)))

]


from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

#Corpus preparado
corpus_preparado = [" ".join(doc) for doc in corpus]

print("Corpus preparado:")
for doc in corpus_preparado:
    print(doc)

#Crear TF-IDF
vectorizer = TfidfVectorizer()

tfidf = vectorizer.fit_transform(corpus_preparado)

#Vocabulario
vocabulario = vectorizer.get_feature_names_out()

print("\n Vocabulario:")
print(vocabulario)

#Matriz TF-IDF
matriz_tfidf = tfidf.toarray()

print("\n Matriz TF-IDF:")
print(matriz_tfidf)

print("\n Matriz TF-IDF en formato tabla:")
df = pd.DataFrame(matriz_tfidf, columns=vocabulario)
print(df)

from collections import Counter
import matplotlib.pyplot as plt

#Unimos todo el corpus en una sola lista de palabras
todas_palabras = [palabra for doc in corpus for palabra in doc]

#Frecuencia de palabras
frecuencia = Counter(todas_palabras)

#6 palabras más usadas
top_6 = frecuencia.most_common(6)

print("Top 6 palabras más usadas:")
for palabra, freq in top_6:
    print(palabra, ":", freq)

#Palabra menos utilizada
menos_usada = min(frecuencia, key=frecuencia.get)
print("\n Palabra menos utilizada:")
print(menos_usada, ":", frecuencia[menos_usada])

#Palabras más repetidas en una misma oración
print("\n Palabras más repetidas por oración:")
for i, doc in enumerate(corpus):
    conteo = Counter(doc)
    mas_repetida = conteo.most_common(1)
    print(f"Oración {i+1}: {mas_repetida}")
    
