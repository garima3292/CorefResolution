from __future__ import print_function
import nltk
import time
import nltk
from nltk.corpus import stopwords
from nltk.corpus import brown
import wikipedia, string
import os
from nltk.stem.wordnet import WordNetLemmatizer
import operator
from gensim.models import Word2Vec
import numpy as np
import h5py

class word2vec():
    def __init__(self,Person,Query):
        self.lmtzr = WordNetLemmatizer()
        self.lmtzr = WordNetLemmatizer()
        self.cachedStopWords = stopwords.words("english")
        self.cachedPunctuations = set(string.punctuation)
        self.cachedPunctuations.remove('-')
        self.cachedPunctuations.remove('.')
        self.vector(Person, Query)

    def getWikiPage(self, query):
        print('Processing wikipedia page for', query)
        wikiPage = wikipedia.page(query)
        return wikiPage

    def removeStopWords(self, content):
        removed = ' '.join([word for word in content.split() if word not in self.cachedStopWords])
        return removed

    def removePunctuation(self, text):
        removed = ''.join(ch for ch in text if ch not in self.cachedPunctuations)
        return removed

    def getTransactionsMovingWindow(self, content, window=20, overlap=5):
        # Gets a moving window of transactions
        # print ' the ' in content
        # content = self.removePunctuation(self.removeStopWords(content)).lower().encode('utf-8')
        content = content.strip()
        content = self.removePunctuation(content)
        content = self.removeStopWords(content).encode('utf-8')
        print(content)
        content = content.decode().split(".")
        words = []
        for sentences in content:
            sentences = sentences.strip()
            contentLemmatized = []
            if sentences != (" " or ""):
                WordList = sentences.split(" ")
                for word in WordList:
                    try:
                        lemout = self.lmtzr.lemmatize(word)
                        if(lemout != ''):
                            contentLemmatized.append(lemout)
                    except:
                        pass
                words.append(contentLemmatized)
        self.NamedEntityReco(words)
        return words

    def NamedEntityReco(self, tokens):
        sentences = [nltk.pos_tag(sent) for sent in tokens]
        print(sentences)
        return

    def vector(self, Person, Query):
        min_count = 2
        size = 50
        window = 4
        page = self.getWikiPage(Person)
        #print(brown.sents())
        if page != None:
            transactions = self.getTransactionsMovingWindow(page.content)
            model = Word2Vec(transactions)

word2vec("Barack Obama","Obama")