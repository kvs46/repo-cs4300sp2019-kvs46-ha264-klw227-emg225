
# coding: utf-8

# In[1]:

import csv
import re
import numpy as np
import json
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse.linalg import svds
from sklearn.preprocessing import normalize
from sklearn.manifold import TSNE


# In[2]:

def to_dict(data):
    with open(data, mode='r') as file:
        reader = csv.reader(file)

        all_data={}
        for rows in reader:
            if len(rows)!=0:
                name = rows[0].replace('"', '')
                all_data[name]={}
                all_data[name]['address']=rows[6].replace('"', '')
                all_data[name]['summary']=rows[1].replace('"', '')
                all_data[name]['long']=rows[8]
                all_data[name]['num_ratings']=rows[3]
                all_data[name]['neighborhood']=rows[4]
                all_data[name]['zip']=rows[15]
                all_data[name]['photo']=rows[13]
                all_data[name]['lat']=rows[2]
                all_data[name]['reviews']=rows[12].replace('"', '')
                all_data[name]['boro']=rows[14].lower().replace("+", " ")
                all_data[name]['rating']=rows[7]
                all_data[name]['amenities']=rows[9].replace('"', '')
                all_data[name]['type']=rows[10]
                all_data[name]['hours']=rows[5]
                all_data[name]['score']=rows[11]    
                
               
    return all_data


# In[3]:
def names_array():
    names = []
    with open('final_data_updated.csv', mode='r') as file:
        reader = csv.reader(file)
        for rows in reader:
            if len(rows)!=0:
                name = rows[0].replace('\\', '')
                name = name.replace('"', '')
                names.append(name)
    return names

def review_to_array(all_reviews):
    tuples_array = []
    first_split = all_reviews.split('/AUTHOR')
    for count in range(1, len(first_split)):
        second_split = first_split[count].split('/RATING:')
        third_split=second_split[1].split('/TEXT: ')
        rating = third_split[0]
        review = third_split[1]
        review = review.replace('"', '')
        tuples_array.append((review, rating))
    return tuples_array


# In[4]:

def clean_data(data):
    for facility in data:
        clean = data[facility]['amenities']
        if(clean != '' and clean != 'No required amenities'):
            before = clean
            after = re.sub(r"(\w)([A-Z])", r"\1 \2", before)
            after2 = after.replace(".", ". ")
            data[facility]['amenities']=after2
        if(data[facility]['reviews']!=''):
            array = review_to_array(data[facility]['reviews'])
            first = [i[0] for i in array]
            data[facility]['reviews']= array
            data[facility]['text'] = "  ".join(first)
        else:
            data[facility]['text'] = data[facility]['summary']
        if data[facility]['boro'] == 'staten is':
            boro = 'staten island'
            data[facility]['boro'] = boro
        
    return data

# In[5]:

def make_document_list(data_dic): 
    documents = []
    for facility in data_dic:
        amenities = data_dic[facility]['amenities']
        summary = data_dic[facility]['summary']
        reviews = data_dic[facility]['reviews']
        first = [i[0] for i in reviews]
        reviews = "  ".join(first)
        combined = amenities + summary + reviews
        documents.append((combined, facility))
    return documents


# In[6]:

def create_vectorizer(max_df, min_df):
    return TfidfVectorizer(stop_words = 'english', max_df = max_df, 
                            min_df = min_df, lowercase=True)


# In[7]:

def create_term_doc(vectorizer, documents):
    my_matrix = vectorizer.fit_transform([x[0] for x in documents]).transpose()
    return my_matrix


# In[8]:

def closest_words(words_compressed, word_to_index, index_to_word, word_in, k = 10):
    if word_in not in word_to_index: return "Not in vocab."
    sims = words_compressed.dot(words_compressed[word_to_index[word_in],:])
    asort = np.argsort(-sims)[:k+1]
    return [(index_to_word[i],sims[i]/sims[asort[0]]) for i in asort[1:]]


# In[9]:

def dict_to_list(data):
    data_list = []
    for park, posting in data.items():
        d = {}
        d[park] = {}
        for key, value in posting.items():
            d[park][key] = value
        data_list.append(d)
    return data_list


# In[10]:

def tokenize(text):
    return re.findall(r'[a-zA-Z]+', text.lower())


# In[11]:

def build_inverted_index(msgs, boro):
    result = {}
    i = -1
    for park, postings in msgs.items():
        i += 1
        seen = set()
        text = tokenize(postings['text'])
        for word in text:
            if word not in seen:
                count = text.count(word)
                seen.add(word)

                if word not in result:
                    result[word] = [(i,count)]
                else:
                    result[word].append((i,count))
      
    return result


# In[12]:

def compute_idf(inv_idx, n_docs, min_df=10, max_df_ratio=0.95):
    result = {}
    for word in inv_idx:
        w_docs = len(inv_idx[word])
        df_ratio = float(w_docs) / float(n_docs)
        #if w_docs > min_df and df_ratio < max_df_ratio: 
        idf = math.log(float(n_docs)/ (1 + float(w_docs)) , 2)
        result[word] = idf
        
    return result


# In[13]:

def compute_doc_norms(index, idf, n_docs):
    result = np.zeros(n_docs)
    sums = {}
    for term in index:
        for (doc, count) in index[term]:
            if term in idf.keys():
                element = math.pow(np.dot(count, idf[term]),2)
                result[doc] = result[doc] + element
    
    
    result = np.sqrt(result)
    return result


# In[14]:

def index_search(query, index, idf, doc_norms):
    #query = list of words 
    #returns results, list of tuples (score, doc_id)
#         Sorted list of results such that the first element has
#         the highest score, and `doc_id` points to the document
#         with the highest score.

    q_norm = 0
    q_norm_seen = set()
    
    """initialize return variables/dictionary to keep track of numerators for documents"""
    result = []
    numerators = {}
    
    """loop through words in query"""
    for term in query:
        if term in index:
            """update q_norm"""
            q_term = np.dot(query.count(term), idf[term])
            if term not in q_norm_seen:
                q_norm = q_norm + math.pow(q_term,2)
                q_norm_seen.add(term)
                
            """start calculating numerators and add to dictionary""" 
            for (doc, count) in index[term]:
                doc_ij = idf[term]*count
                if doc in numerators:
                    numerators[doc] = numerators[doc] + doc_ij*q_term
                else:
                    numerators[doc] = doc_ij*q_term
                    
    """finish calculating q_norm"""
    q_norm = math.sqrt(q_norm) 
    
    """append all the documents and scores to result"""
    for doc in range(len(doc_norms)):
        if doc in numerators.keys():
            denominator = q_norm * doc_norms[doc]
            result.append((float(numerators[doc])/float(denominator), doc))
        else:
            result.append((0, doc))
    
    """sort"""
    #result.sort(key=lambda tup: tup[0])
    result.sort()
    result = result[::-1]
    return result


# In[15]:

def get_rankings(words_compressed, word_to_index, index_to_word,termlist, boros, data):
    final = []
    data_list = dict_to_list(data)
    #COSINE SIM
    n_docs = len(data)
    index = build_inverted_index(data, boros)
    idf = compute_idf(index, n_docs)
    doc_norms = compute_doc_norms(index, idf, n_docs)
    query = termlist
    #expand query based on ML
    for word in termlist:
        close_words = closest_words(words_compressed,word_to_index, index_to_word,word.lower(), k=10)
        terms, closeness = zip(*close_words)
        query = query + list(terms)
                
    output = index_search(query, index, idf, doc_norms) #(score, doc index list)
    
    #only want boro, don't want repeats
    seen = []
    for score, doc in output:
        for park, posting in data_list[doc].items():
            for boro in boros:
                if posting['boro'] == boro and posting['text'] not in seen:  
                    final.append(data_list[doc])
                seen.append(posting['text'])

    return final


# In[16]:
def good_types():
    all_data = to_dict('final_data_updated.csv')
    all_data = clean_data(all_data)
    documents = make_document_list(all_data)
    vectorizer = create_vectorizer(0.6, 5)
    term_doc = create_term_doc(vectorizer, documents)
    words_compressed, _, docs_compressed = svds(term_doc, k=40)
    docs_compressed = docs_compressed.transpose()
    word_to_index = vectorizer.vocabulary_
    index_to_word = {i:t for t,i in word_to_index.items()}

    good_types = []
    for item in index_to_word.items():
        good_types.append(item[1])
    return good_types

def main(boro, termlist):
    all_data = to_dict('final_data_updated.csv')
    all_data = clean_data(all_data)
    documents = make_document_list(all_data)
    vectorizer = create_vectorizer(0.6, 5)
    term_doc = create_term_doc(vectorizer, documents)
    words_compressed, _, docs_compressed = svds(term_doc, k=40)
    docs_compressed = docs_compressed.transpose()
    word_to_index = vectorizer.vocabulary_
    index_to_word = {i:t for t,i in word_to_index.items()}
    words_compressed = normalize(words_compressed, axis = 1)
    
    results = get_rankings(words_compressed, word_to_index, index_to_word,termlist, boro, all_data)
   
   
    return results


# In[ ]:

# def similar_parks():
#     with open('sim_dict.csv', mode='r') as file:
#         similar={}
#         reader = csv.reader(file)
#         for row in reader:
#             if len(row)!=0:
#                 similar[row[0]]=[row[1], row[2], row[3], row[4], row[5]]
#     return similar

# In[ ]:
def similar_parks(park):
    all_data = to_dict('final_data_updated.csv')
    all_data = clean_data(all_data)
    data_list = dict_to_list(all_data)
    final = []
    with open('sim_dict.csv', mode='r') as file:
        similar={}
        reader = csv.reader(file)
        for row in reader:
            if len(row)!=0:
                similar[row[0]]=[row[1], row[2], row[3], row[4], row[5]]
    
    for p in similar[park]:
        entry = {}
        entry[p] = all_data[p]
        final.append(entry)
    return final