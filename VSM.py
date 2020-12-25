# %%
"""
# importing libraries
"""

# %%
from collections import defaultdict
import math
import sys
import re
import glob
import numpy as np 
import pandas as pd
from nltk.stem import PorterStemmer
from functools import reduce
from nltk.corpus import stopwords

# %%
"""
# Getting data and tokenization and removing stop words
"""

# %%
docNames= []
document_filenames = {}
for name in glob.glob(r'Files'+'\*'):
        docNames.append(name)
for i in range(0,10):
    document_filenames[i] = docNames[i]
N = len(document_filenames)
N = len(document_filenames)
dictionary = set()
docsLen=[]
postings = defaultdict(dict)
documents_content= defaultdict(dict)
document_frequency = defaultdict(int)
length = defaultdict(float)


# %%
'''
def main():
    initialize_terms_and_postings()
    initialize_document_frequencies()
    norm()
    while True:
        query=input("please enter query>>")
        do_search(query)
'''
# %%
def initialize_terms_and_postings():
    global dictionary, postings, docsLen
    for id in document_filenames:
        f = open(document_filenames[id],'r')
        document = f.read().replace('\n'," ")
        documents_content[id]=document
        f.close()
        terms = tokenize(document)
        unique_terms = set(terms)
        NormTF= set()
        docsLen.append(len(unique_terms))
        dictionary = dictionary.union(unique_terms)
        for term in unique_terms:
            postings[term][id] = terms.count(term)

# %%
def tokenize(document):
        stop_words = stopwords.words('english')
        document = re.sub(" "," ", document)
        document = re.sub(r'won’t', 'will not', document)
        document = re.sub(r'can\’t', 'can not', document)
        document = re.sub(r'n\’t', 'not', document)
        document = re.sub(r'\’re', 'are', document)
        document = re.sub(r'\’s', 'is', document)
        document = re.sub(r'\’d', 'would', document)
        document = re.sub(r'\’ll', 'will', document)
        document = re.sub(r'\’t', 'not', document)
        document = re.sub(r'\’ve', ' have', document)
        document = re.sub(r'\’m', 'am', document)
        document = re.sub(r'[0–9]+', "", document)
        document = re.sub(r'[^\w\s]'," ", document)
        document = document.lower()
        terms=[]
        for word in document.split(" "):
            if word!="":
                if word not in stop_words:
                    terms.append(word)
        doc = list(filter(None, terms)) 
        return doc
    

# %%
"""
# Document Frequency
"""

# %%
#term frequncy in doc > postings[term][id]
'''lazm n normalize l term frequency 3shan kol document l size beta3ha mo5talef 3n l tanya
fa ha2sem l tf/total num of term in document'''
# number of documents that include the term >  len(postings[term])
def initialize_document_frequencies():
    global document_frequency
    for term in dictionary:
        document_frequency[term] = len(postings[term])

# %%
"""
# Normalized Term Frequency
"""

# %%
def NormTf():
    NormTF= defaultdict(dict)
    for id in document_filenames:
        for i in dictionary:
            NormTF[i][id] = i.count(i)/docsLen[id]
    return NormTF

"""
# Norm of query and documents
"""

# %%
def norm():
    global length
    for id in document_filenames:
        l = 0
        for term in dictionary:
            l += tf_idf(term,id)**2
        length[id] = math.sqrt(l)
def qnorm(query):
    l2 = 0
    for i in range(len(query)):
        for term in query:
            for id in document_filenames:
                l2 += tf_idf(term,id)**2
        qlength = math.sqrt(l2)
    return qlength
        

# %%
"""
# TF-IDF
"""

# %%
def tf_idf(term,id):
    tf = NormTf()
    if id in tf[term]:
        return tf[term][id]*inverse_document_frequency(term)
    else:
        return 0.0

# %%
"""
# Inverse Document Frequency
"""

# %%
def inverse_document_frequency(term):
    if term in dictionary:
        return math.log(N/document_frequency[term],2)
    else:
        return 0.0

# %%
def do_search(x):
    filenames = {}
    for i in range(10):
        s = document_filenames[i].split('\\')
        filenames[i] = s[-1]
    query = tokenize(x)
    if query == []:
        sys.exit()
    relevant_document_ids = intersection(
            [set(postings[term].keys()) for term in query])
    if not relevant_document_ids:
        scores=[]
    else:
        scores = sorted([(id,similarity(query,id)) for id in relevant_document_ids],
                        key=lambda x: x[1],
                        reverse=True)
    return scores, filenames, documents_content


# %%
def intersection(sets):
    return reduce(set.intersection, [s for s in sets])

# %%
def similarity(query,id):
    similarity = 0.0
    qlength= qnorm(query)
    for term in query:
        if term in dictionary:
            similarity += inverse_document_frequency(term)*tf_idf(term,id)
    similarity = similarity / length[id]*qlength
    return similarity

