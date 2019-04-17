import csv
import re

def read_csv(file):
    with open(file, mode = 'r') as file:
        reader = csv.reader(file)
        data = {}
        first = True
        for rows in reader:
            if first:
                first = False
            else:
                data[rows[0]] = {}
                data[rows[0]]['rating'] = rows[1]
                data[rows[0]]['num_ratings'] = rows[2]
                data[rows[0]]['zip'] = rows[3]
                data[rows[0]]['uid'] = rows[4]
                data[rows[0]]['address'] = rows[5]
                boro = rows[6].lower()
                if boro == 'staten+is':
                    boro = 'staten+island'
                data[rows[0]]['boro'] = boro
                data[rows[0]]['reviews'] = rows[7]
                data[rows[0]]['lat'] = rows[8]
                data[rows[0]]['website'] = rows[9]
                data[rows[0]]['long'] = rows[10]
                data[rows[0]]['photo'] = rows[11]
                data[rows[0]]['hours'] = rows[12]
                data[rows[0]]['type'] = rows[13]
    return data

#get good types of reviews:
#get string of all reviews:
def get_good_types(data):
    all_reviews = ""
    for park, posting in data.items():
        reviews = posting['reviews']
        all_reviews = all_reviews + " " + reviews
    return reviews

#tokenize:
def tokenize(text):
    return re.findall(r'[a-zA-Z]+', text.lower())

def distinct_word(review_words):
    num_distinct_words = set()
    for word in review_words:
        num_distinct_words.add(word)
    return num_distinct_words
    
def dict_word_count(data, num_distinct_words):
    #dictionary of word counts 
    word_counts = {}
    for word in num_distinct_words:
        for park, posting in data.items():
            if word in posting['reviews']:
                #already in dictionary
                if word in word_counts.keys():
                    word_counts[word] = word_counts[word] + 1

                #not in dictionary yet
                else:
                    word_counts[word] = 1
    return word_counts

def output_good_types(input_word_counts):
    """Returns a list of good types in alphabeitcally sorted order
        Params: {input_word_counts: Dict}
        Returns: List
    """
    # YOUR CODE HERE
    result = []
    for word in input_word_counts:
        if input_word_counts[word] > 1 :
            result.append(word)
    result.sort()
    return result

#make a tuple list of (word, count)
def tuple_word_counts(good_types, review_words):
    tuple_counts = []
    for word in good_types:
        count = review_words.count(word)
        if count < 220:
            tuple_counts.append((word, count))

    tuple_counts.sort(key= lambda tup: tup[1])
    tuple_counts.reverse()
    return tuple_counts

#return list of {park:posting} dictionaries such that each park contains
#the desired keyword in the reviews and is order so that index 0 has the highest rating. 

def get_rankings(terms, boros, data):
    final = []
    rankings = []
    for park, posting in data.items():
        current_boro = posting['boro']
        #only want specified boros
        for boro in boros:
            if boro == current_boro:
                #bool search of desired keyword
                reviews = posting['reviews']
                for term in terms:
                    if term in reviews:
                        rating = float(posting['rating'])
                        num_reviews = float(posting['num_ratings'])
                        overall_ranking = rating*num_reviews
                        
                        if len(rankings) > 0:
                            name, ranks = zip(*rankings)
                            #skip if already in ranking
                            if overall_ranking in ranks:
                                continue
                                
                            #not already in ranking- sort once new park added: 
                            else:
                                if len(rankings) < 11:
                                    rankings.append((park, overall_ranking))
                                    #
                                    rankings = sorted(rankings, key=lambda tup: tup[1])[::-1]
                                    
                                else:
                                    if overall_ranking > rankings[4][1]:
                                        rankings.pop(10)
                                        rankings.append((park, overall_ranking))
                                        #sort
                                        rankings = sorted(rankings, key=lambda tup: tup[1])[::-1]
                        else:
                            rankings.append((park, overall_ranking))
                            #sort
                            rankings = sorted(rankings, key=lambda tup: tup[1])[::-1]
            
    #add to dicitionary:
    for (p, rank) in rankings:
        dictionary = {}
        dictionary[p] = data[p]
        final.append(dictionary)
        
    return final

def main(boro, feature):
    parks_and_gardens = read_csv('parks_and_gardens.csv')
    reviews = get_good_types(parks_and_gardens)
    review_words = tokenize(reviews)
    distinct = distinct_word(review_words)
    word_counts = dict_word_count(parks_and_gardens, distinct)
    good_types = output_good_types(word_counts)
    tuple_counts = tuple_word_counts(good_types, review_words)
    
    results = get_rankings(feature, boro, parks_and_gardens)
    
    return results

