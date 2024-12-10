import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

def download_nltk_resource(resource_name):
    try:
        nltk.data.find(resource_name)
    except LookupError:
        print(f"Downloading {resource_name}...")
        nltk.download(resource_name)


download_nltk_resource('tokenizers/punkt')
download_nltk_resource('corpora/stopwords')
download_nltk_resource('corpora/wordnet')
download_nltk_resource('corpora/omw-1.4')
download_nltk_resource('corpora/punkt_tab')


lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))


def text_tokenizer(text):
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalnum()]
    filtered_tokens = [word for word in tokens if word not in stop_words]
    stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in stemmed_tokens]

    return lemmatized_tokens


def generate_inverted_index(preprocessed_documents):
    inverted_index = {}

    for doc_id, tokens in preprocessed_documents.items():
        for token in tokens:
            if token not in inverted_index:
                inverted_index[token] = set()
            inverted_index[token].add(doc_id)

    return inverted_index
