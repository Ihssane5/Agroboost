
import nltk
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
import numpy as np

nltk.download('punkt')
"""
class Sentence_Processing:

    Providing Essential Sentence Processing Techniques : Tokenization, Stemming, Bag of words
    Attributes:
    sentence: str
        the sentence to be processed
    tokenized_sentence: list
        list of tokens in the sentence
    stemmed_sentence: list
        list of stemmed tokens
    
    
    Methods:
    Tokenize(sentence: str) -> list:
        Applies Tokenization to the Sentence.
        Input : String
        Output : List of tokens
    Stem(sentence: list) -> list
        Applies Stemming to the tokenized sentence 
        Input : list of tokens
        Output : list of stemmed tokens
    Bag_of_Words(tokenized_sentence: list) -> list:
        Applies bag of words 
        Input : list of tokens
        Output : list
    """



def Tokenize(sentence):
    tokenized_sentence = word_tokenize(sentence)
    return tokenized_sentence

def Stem(word):
    #Instantiating the stemmer
    stemmer = PorterStemmer()
    stemmed_word = stemmer.stem(word.lower())
    return stemmed_word
    
def Bag_of_words(tokenized_sentence,  all_words):
    stemmed_sentence = [Stem(w) for w in tokenized_sentence]
    bag = np.zeros (len(all_words), dtype=np.float32)
    for index, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[index] = 1.0
    return bag
    



if __name__ == "__main__":
    sentence = 'I\'m working on building an fertilizer prediction app'
    print(sentence)
    tokenized_sentence = Tokenize(sentence)
    print(tokenized_sentence)
    stemmed_sentence = Stem(sentence)
    print(stemmed_sentence)
    all_words = ["I", 'app','ferilizer']
    bag = Bag_of_words(sentence,all_words)
    print(bag)

    