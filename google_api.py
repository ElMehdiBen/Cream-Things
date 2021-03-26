import json
import re
from googleapiclient.discovery import build
# pip install google-api-python-client
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

def getService():
    service = build("customsearch", "v1", developerKey="AIzaSyDxpJfGwM9KzeTFlKa-_Z-wEgI1sKMcKKo")

    return service

def main():

    pageLimit = 4
    service = getService()
    startIndex = 1
    response = []

    for nPage in range(0, pageLimit):
        print("Reading page number:",nPage+1)

        response.append(service.cse().list(
            q = 'Digital Marketing', #Search words
            cx = '001132580745589424302:jbscnf14_dw',  #CSE Key
            lr = 'lang_en', #Search language
            start = startIndex
        ).execute())

        startIndex = response[nPage].get("queries").get("nextPage")[0].get("startIndex")

    with open('data.json', 'w') as outfile:
        json.dump(response, outfile)


def exploreResult(results):
	with open(results) as json_file:
		data = json.load(json_file)
	snippets = ""
	for page in data:
		for result in page["items"]:
			snippets += result["snippet"] + " "
	return snippets


def pre_process(text):
    
    # lowercase
    text=text.lower()
    
    #remove tags
    text=re.sub("","",text)
    
    # remove special characters and digits
    text=re.sub("(\\d|\\W)+"," ",text)
    
    return text

def get_stop_words(stop_file_path):
    """load stop words """
    
    with open(stop_file_path, 'r', encoding="utf-8") as f:
        stopwords = f.readlines()
        stop_set = set(m.strip() for m in stopwords)
        return frozenset(stop_set)

def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    """get the feature names and tf-idf score of top n items"""
    
    #use only topn items from vector
    sorted_items = sorted_items[:topn]

    score_vals = []
    feature_vals = []
    
    # word index and corresponding tf-idf score
    for idx, score in sorted_items:
        
        #keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])

    #create a tuples of feature,score
    #results = zip(feature_vals,score_vals)
    results= {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]]=score_vals[idx]
    
    return results

def tfidf(text, stopwords):
	cv = CountVectorizer(stop_words=stopwords)
	word_count_vector = cv.fit_transform([text])
	tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
	tfidf_transformer.fit(word_count_vector)
	feature_names = cv.get_feature_names()
	tf_idf_vector = tfidf_transformer.transform(cv.transform([text]))
	sorted_items = sort_coo(tf_idf_vector.tocoo())
	keywords = extract_topn_from_vector(feature_names,sorted_items,10)
	return keywords

main()
agg_text = exploreResult("data.json")
pre_text = pre_process(agg_text)
stopwords = get_stop_words("stopwords.txt")
keywords = tfidf(pre_text, stopwords)
print("\n===Keywords===")
for k in keywords:
    print(k,keywords[k])