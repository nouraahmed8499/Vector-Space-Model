# %%
"""
# importing libraries
"""

# %%
from collections import defaultdict
import math
import sys
import numpy as np 
import re
import glob
import pandas as pd
from nltk.stem import PorterStemmer
from functools import reduce
from nltk.corpus import stopwords
import nltk

#uncomment to download nltk files
#nltk.download()
# %%
"""
# Getting data and tokenization and removing stop words

"""

# %%
docNames= []
document_filenames = {}
for name in glob.glob(r'Files'+'\*'):
        docNames.append(name)
N = len(docNames)
for i in range(N):
    document_filenames[i] = docNames[i]
dictionary = set()
docsLen=[]
tfIdfMatrix=defaultdict(dict)
idf = defaultdict(dict)
postings = defaultdict(dict)
posIndex= defaultdict(dict)
documents_content= defaultdict(dict)
document_frequency = defaultdict(int)
length = defaultdict(float)



# %%
def initialize_terms_and_postings():
    global dictionary, postings, docsLen, posIndex
    for id in document_filenames:
        f = open(document_filenames[id],'r')
        document = f.read().replace('\n'," ")
        documents_content[document_filenames[id].split('\\')[1]]=document
        f.close()
        terms = tokenize(document)
        unique_terms = set(terms)
        docsLen.append(len(unique_terms))
        dictionary = dictionary.union(unique_terms)
        #here we get the log tf weighting not just term frequency
        for term in unique_terms:
            postings[term][id] = 1+np.log10(int(terms.count(term)))
        #now we initialize postitional index
        poslst=[] #this is a position list for a term
        for term in terms:
            for i in range(0,len(terms)):
                if term == terms[i]:
                     poslst.append(i)
            posIndex[term][id] = poslst #we save positions in the positional index for each term in each doc 
            poslst=[] 


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
        terms2=[]
        ps= PorterStemmer()
        for term in terms :
            term = ps.stem(term)
            terms2.append(term)            
        doc = list(filter(None, terms2)) 
        return doc
   

# %%
 # Document Frequency

# number of documents that include the term >  len(postings[term])
def initialize_document_frequencies():
    global document_frequency
    for term in dictionary:
        document_frequency[term] = len(postings[term])
        

# %%
initialize_terms_and_postings()
initialize_document_frequencies()


# %%
documents_content


# %%
"""
# Positional Index
"""

"""
# Log Term Frequency Weighting 
"""

"""
# Inverse Document Frequency
"""

# %%
def inverse_document_frequency(term):
    if term in dictionary:
        return np.log10(N/document_frequency[term])
    else:
        return 0.0

# %%
for term in dictionary: 
      idf[term]=float(inverse_document_frequency(term))

# %%
"""
# TF-IDF Matrix for Collection

"""

# %%
for i in range(N):
    for term in postings:
            if i in postings[term]:
                tfIdfMatrix[term][i]= postings[term][i]* idf[term]
            else:  tfIdfMatrix[term][i]=0


"""
# Norm of query and documents
"""

# %%
def norm():
    for id in document_filenames:
        l = 0
        for term in dictionary:
            l += tfIdfMatrix[term][id]**2
        length[id] = math.sqrt(l)
        
def qnorm(query):
    QueryTfIdf = queryPreperation(query)
    l2 = 0
    for term in QueryTfIdf.values():
        if type(term)== float: l2 += term**2
        qlength = math.sqrt(l2)
    return qlength
norm()
length

# %%
"""
# Query Preperation
"""

# %%
#this is to make a tfidf for the query
QueryFreqList = defaultdict(dict)
QueryTfIdfMatrix = defaultdict(dict)
queryLen= None
def queryPreperation(query):
    terms = tokenize(query)
    unique_terms = set(terms)
    queryList=[]
    for term in unique_terms: 
        if term in dictionary: 
            queryList.append(term)
    dictionary1 = dictionary.union(queryList)
    for term in dictionary1:
        if int(terms.count(term)) != 0:
             QueryFreqList[term]= 1+np.log10(int(terms.count(term)))
        else: QueryFreqList[term]=0
    for i in range(N):
        for term in QueryFreqList.keys():
                if QueryFreqList[term]!=0: 
                    QueryTfIdfMatrix[term]= float(QueryFreqList[term]* idf[term])
                else:  QueryTfIdfMatrix[term]=0
    return QueryTfIdfMatrix


# %%
#using the positional index to check if the terms in the right order in documents
def phraseQuery(query):
    positionsDict = {}
    listOflsts = []
    listOfRelDocId = []
    terms = tokenize(query)
    for id in document_filenames:
        for term in terms:
            if term in dictionary:
                if id in posIndex[term]:
                    listOflsts.append(posIndex[term][id])
        positionsDict[id] = listOflsts
        listOflsts = []
    empty_keys = [k for k,v in positionsDict.items() if v==[]]
    for k in empty_keys:
        del positionsDict[k]
    #getting the relevant docs
    for docId,listOfLists in positionsDict.items():
        listOfLists2 = [] #this is list of position lists after subtraction from i 
        i = 0
        for lst in listOfLists:
            lst2 = [] #this is positions list after subtraction from i 
            for item in lst:
                lst2.append(item - i)
            listOfLists2.append(lst2)
            i += 1
        # now we get intersection of lists
        # if not empty, they're in order
        # if empty they're not in order
        result = list(reduce(set.intersection, [set(item) for item in listOfLists2]))
        if result: 
            listOfRelDocId.append(docId)
    return listOfRelDocId

# %%
"""
# Cosine Similarity 
"""

# %%
def similarity(query,id):
    similarity = 0.0
    qlen = qnorm(query)
    QueryTfIdf = queryPreperation(query)
    query = tokenize(query)
    for term in query:
        if term in dictionary:
                similarity += (QueryTfIdf[term]/qlen)*(tfIdfMatrix[term][id]/length[id])
    if similarity > 1: similarity = 1.00000
    return similarity


# %%
"""
# Search
"""

# %%
def do_search(query):
    filenames = {}
    relDoc= []
    tPositions={}
    listOfPositions=[]
    scores = {}
    dataframe = defaultdict(dict)
    relDocs = phraseQuery(query)
    for i in relDocs:
        scores[i] = similarity(query,i)
    for i in relDocs: 
        if scores[i]!= 0:
             dataframe[document_filenames[i].split('\\')[1]] = scores[i]
    df = pd.DataFrame.from_dict(dataframe,orient='index',columns = ['Cosine Similarity(Query,Document)'])
    df.sort_values(by=['Cosine Similarity(Query,Document)'], inplace=True, ascending = False)
    if dataframe == {}: return 0 
    else: return dataframe
    

# %%
"""
# similarity between query and each document
"""
