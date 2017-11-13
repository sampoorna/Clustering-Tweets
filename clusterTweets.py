# -*- coding: utf-8 -*-
'''
Reads tweets from text (CSV) file storing scraped tweets from news providers on Twitter. Clusters them into different topics according to keywords using mini batch k-means. The clusters are grown over time as new tweets are added to the text file.
'''

import nltk, re, io
import numpy as np
import csv, time
from nltk.stem.snowball import SnowballStemmer
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from scipy.sparse import coo_matrix, hstack

reload(sys)  
sys.setdefaultencoding('utf8')

def tokenize_and_stem(text):
	stemmer = SnowballStemmer("english")
	# first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
	tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
	filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
	for token in tokens:
		if re.search('[a-zA-Z]', token):
			filtered_tokens.append(token)
	stems = [stemmer.stem(t) for t in filtered_tokens]
	final_stems = []
	for stem in stems:
		if stem not in stop_words:
			final_stems.append(stem)
	return final_stems

def load_stopwords():
	stop_words = nltk.corpus.stopwords.words('english')
	stop_words.extend(['this','that','the','might','have','been','from', 'but','they','will','has','having','had','how','went', 'were','why','and','still','his','her','was','its','per','cent', 'a','able','about','across','after','all','almost','also','am','among', 'an','and','any','are','as','at','be','because','been','but','by','can', 'cannot','could','dear','did','do','does','either','else','ever','every', 'for','from','get','got','had','has','have','he','her','hers','him','his', 'how','however','i','if','in','into','is','it','its','just','least','let', 'like','likely','may','me','might','most','must','my','neither','nor', 'not','of','off','often','on','only','or','other','our','own','rather','said', 'say','says','she','should','since','so','some','than','that','the','their', 'them','then','there','these','they','this','tis','to','too','twas','us', 'wants','was','we','were','what','when','where','which','while','who', 'whom','why','will','with','would','yet','you','your','ve','re','rt', 'retweet', '#fuckem', '#fuck', 'fuck', 'ya', 'yall', 'yay', 'youre', 'youve', 'ass','factbox', 'com', '&lt', 'th', 'retweeting', 'dick', 'fuckin', 'shit', 'via', 'fucking', 'shocker', 'wtf', 'hey', 'ooh', 'rt&amp', '&amp', '#retweet', 'retweet', 'goooooooooo', 'hellooo', 'gooo', 'fucks', 'fucka', 'bitch', 'wey', 'sooo', 'helloooooo', 'lol', 'smfh'])
	#turn list into set for faster search
	stop_words = set(stop_words)
	return stop_words

def normalize_text(text):
	try:
		text = text.encode('utf-8')
		#print text
	except: 
		print "error: ", text
		print sys.exc_info()[0]
		pass
	text = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(pic\.twitter\.com/[^\s]+))','', text)
	text = re.sub('@[^\s]+','', text)
	text = re.sub('#([^\s]+)', '', text)
	text = re.sub('[:;>?<=*+()/,\-#!$%\{|\}\[^_\\@\]1234567890’‘]',' ', text)
	text = re.sub('[\d]','', text)
	text = text.replace(".", '')
	text = text.replace("'", ' ')
	text = text.replace("\"", ' ')
	text = text.replace("\x9c", ' ').replace("\xc2", ' ').replace("\xa6", ' ')
	#text = text.replace("-", " ")
	#normalize some utf8 encoding
	text = text.replace("\x9d",' ').replace("\x8c",' ').replace("\x94",' ').replace("\x97",' ')
	text = text.replace("\xa0",' ')
	text = text.replace("\x9d\x92", ' ').replace("\x9a\xaa\xf0\x9f\x94\xb5", ' ').replace("\xf0\x9f\x91\x8d\xcb\x87\xba\xf0\x9f\x87\xb8", ' ').replace("\x9f",' ').replace("\x91\x8d",' ')
	text = text.replace("\xf0\x9f\x87\xba\xf0\x9f\x87\xb8",' ').replace("\xf0",' ').replace('\xf0x9f','').replace("\x9f\x91\x8d",' ').replace("\x87\xba\x87\xb8",' ')	
	text = text.replace("\xe2\x80\x94",' ').replace("\x9d\xa4",' ').replace("\x96\x91",' ').replace("\xe1\x91\xac\xc9\x8c\xce\x90\xc8\xbb\xef\xbb\x89\xd4\xbc\xef\xbb\x89\xc5\xa0\xc5\xa0\xc2\xb8",' ')
	text = text.replace("\xe2\x80\x99s", " ").replace("\xe2\x80\x98", ' ').replace("\xe2\x80\x99", ' ').replace("\xe2\x80\x9c", " ").replace("\xe2\x80\x9d", " ")
	text = text.replace("\xe2\x82\xac", " ").replace("\xc2\xa3", " ").replace("\xc2\xa0", " ").replace("\xc2\xab", " ").replace("\xf0\x9f\x94\xb4", " ").replace("\xf0\x9f\x87\xba\xf0\x9f\x87\xb8\xf0\x9f", "")
	return text
		
'''Prepare features, where doc has terms separated by comma'''
def custom_tokenize_text(text):
	REGEX = re.compile(r",\s*")
	tokens = []
	for tok in REGEX.split(text):
		#if "@" not in tok and "#" not in tok:
		if "@" not in tok:
			#tokens.append(stem(tok.strip().lower()))
			tokens.append(tok.strip().lower())
	return tokens
# --------------------------------------------------------------------- #
	
tweet_filename = 'tweets3.csv' # Input file consisting of tweets scraped from Twitter
cluster_dist = 0.2 # Max distance between tweet and cluster centroid
hot_tweets = []
hot_tweets_urls = []
avg_retweet_count = 0
num_clusters = 7
cluster_dist_dict = {}
	
# Read all tweets from file
print "Reading tweets from memory ..."
with io.open(tweet_filename, 'r', encoding='utf-8', errors='replace') as f:
    tweet_reader = csv.reader(f)
    tweets = list(tweet_reader)
print "Reading tweets from memory ... COMPLETE"

stop_words = load_stopwords()

all_stems = []
all_tweet_texts = []
all_tweet_ids = []

# Process tweets
retweets = []
for tw in tweets:
	#print tw
	tweet_text = normalize_text(tw[1].decode('utf-8', 'replace'))
	tweet_text = tweet_text.decode('utf-8', 'replace')
	retweets.append(int(tw[3]))
	#print type(tweet_text)
	#stems = tokenize_and_stem(tweet_text)
	#all_stems.append(stems)
	all_tweet_texts.append(tweet_text)
	all_tweet_ids.append(tw[0])
	#print stems
#print retweets
avg_retweet_count = np.mean(retweets)

tfidf_vectorizer = TfidfVectorizer(max_df=0.6, max_features=10000, min_df=0.01, stop_words='english', use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1, 1))

tfidf_matrix = tfidf_vectorizer.fit_transform(all_tweet_texts) # fit the vectorizer to stemmed tweets
print tfidf_matrix.shape

km = MiniBatchKMeans(n_clusters=num_clusters, init='k-means++')
km.fit(tfidf_matrix)

# Cluster tweets using Mini batch k-means
	
# Cluster tweets using DBSCAN
#min_samples = 10
#db = DBSCAN(eps = 0.3, min_samples = min_samples).fit(tfidf_matrix)
#clusters = db.labels_.tolist()
# Number of clusters in labels, ignoring noise if present
#num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
#print num_clusters

print("Top terms per cluster: ")
print
#sort cluster centers by proximity to centroid
order_centroids = km.cluster_centers_.argsort()[:, ::-1] 
terms = tfidf_vectorizer.get_feature_names()

#print km.cluster_centers_

#print terms

for i in range(num_clusters):
    print "Cluster ", i, " words:"
    
    for ind in order_centroids[i, :10]: #replace 6 with n words per cluster
		#   print(' %s' % terms[ind].split(' ').encode('utf-8', 'ignore'))
		print terms[ind],
    print #add whitespace
 
# Finding the tweet that is closest to the centroid
clusters = km.fit_predict(tfidf_matrix)

doc_ids_per_cluster = []
min_dist_tweet_ind = [0 for i in range(num_clusters)]
print "Getting documents in each cluster ..."
for cl in range(num_clusters):
	print "Tweet for cluster ", cl,
	idx = np.where(clusters==cl)[0] # List of indices of tweets that belong to cluster == cl
	
	D = euclidean_distances(tfidf_matrix[idx], km.cluster_centers_[cl].reshape(1, -1)) # Distance matrix of all tweets in one cluster, from the centroid of the cluster
	#print D.shape
	D = D.tolist()
	min_dist = min(D) # Select tweet id that is closest to center
	cluster_dist_dict[cl] = max(D)
	#print all_tweet_texts[idx[D.index(min_dist)]]
	
#print clusters

#print type(clusters)

recluster_count = 0
while True:
	# Reading more text from file
	print 'Checking for new tweets ...'
	new_tweets = []
	new_tweet_texts = []
	with io.open(tweet_filename, 'r', encoding='utf-8', errors='replace') as f:
		tweet_reader = csv.reader(f)
		for i, row in enumerate(tweet_reader):
			if i > len(tweets):
				#tweets.append(list(tweet_reader[row]))
				#print list(row)
				new_tweets.append(list(row))
				new_tweet_texts.append(row[1])
				retweets.append(int(row[3]))
				#retweets.append()
        #break
	#print new_tweets
	tweets.extend(new_tweets)
	all_tweet_texts.extend(new_tweet_texts)
	avg_retweet_count = np.mean(retweets)
	print 'Checking for new tweets ... COMPLETE'
	
	if len(new_tweets) > 0:
		# If new tweet is far from existing clusters, inc num of clusters
		tfidf_tweet = tfidf_vectorizer.fit_transform(all_tweet_texts)
	
		## Add to matrix
		#tfidf_matrix = tfidf_matrix.append(tfidf_tweet)
		#hstack(tfidf_matrix, tfidf_tweet)
	
		for tweet_ind in range(len(new_tweets)):
			#D = euclidean_distances(tfidf_tweet, km.cluster_centers_[cl].reshape(1, -1))
			clustered_flag = 0 # Keeps track of whether this tweet is assigned to a cluster
			for cl in range(num_clusters):
				distance = euclidean_distances(tfidf_tweet[tweet_ind], km.cluster_centers_[cl].reshape(1, -1))
				if distance <= cluster_dist_dict[cl]:
					## Assign label of this cluster
					np.append(clusters, cl)
					clustered_flag = 1
			# If tweet was not assigned to any cluster, add it to new cluster
			if clustered_flag == 0:
				np.append(clusters, num_clusters + 1)
				
				# If new tweet is in new cluster and has higher than average retweet count, it is a hot tweet
				if new_tweets[tweet_ind][3] > avg_retweet_count:
					if len(hot_tweets < 2):
						hot_tweets.append(new_tweets[tweet_ind])
						hot_tweets_urls.append('https://twitter.com' + '/' + new_tweets[tweet_ind][2] + '/status/' + str(new_tweets[tweet_ind][0]))
					else:
						for t in hot_tweets:
							if t[3] < new_tweets[tweet_ind][3]:
								ind = hot_tweets.index(t)
								hot_tweets[ind] = new_tweets[tweet_ind]
								hot_tweets_urls[ind] = 'https://twitter.com' + '/' + new_tweets[tweet_ind][2] + '/status/' + str(new_tweets[tweet_ind][0])
								break
	
	# After 1 hour, re-cluster everything
	recluster_count = recluster_count + 600
	if recluster_count > 60*60:
		print 'Reclustering ...'
		## Reset counter
		recluster_count = 0
		
		## Recluster
		num_clusters = num_clusters + 1
		km = MiniBatchKMeans(n_clusters=num_clusters, init='k-means++')
		tfidf_vectorizer = TfidfVectorizer(max_df=0.6, max_features=10000, min_df=0.01, stop_words='english', use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1, 1))
		tfidf_matrix = tfidf_vectorizer.fit_transform(all_tweet_texts) # fit the vectorizer to stemmed tweets
		km.fit(tfidf_matrix)
		clusters = km.fit_predict(tfidf_matrix)
		print 'Reclustering ... COMPLETE'
		
	print "Hot tweets identified so far...", 
	print hot_tweets_urls
	# Sleep for 10 mins before checking for new tweets again
	time.sleep(600)