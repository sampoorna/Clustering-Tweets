'''
Scrapes tweets from news providers on Twitter and stores them in a text file. Currently only stores the following attributes: tweet id, tweet text, news provider and number of retweets, but these attributes can be modified according to requirement.
'''

import tweepy
from tweepy import OAuthHandler
import sys
import codecs
import csv
import time
 
consumer_key = ''
consumer_secret = ''
access_token = ''
access_secret = ''
 
auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)
 
api = tweepy.API(auth)

news_urls = ['https://twitter.com/nytimes', 'https://twitter.com/thesunnewspaper', 'https://twitter.com/thetimes', 'https://twitter.com/ap', 'https://twitter.com/cnn', 'https://twitter.com/bbcnews', 'https://twitter.com/cnet', 'https://twitter.com/msnuk', 'https://twitter.com/telegraph', 'https://twitter.com/usatoday', 'https://twitter.com/wsj', 'https://twitter.com/washingtonpost', 'https://twitter.com/bostonglobe', 'https://twitter.com/newscomauhq', 'https://twitter.com/skynews', 'https://twitter.com/sfgate', 'https://twitter.com/ajenglish', 'https://twitter.com/independent', 'https://twitter.com/guardian', 'https://twitter.com/latimes', 'https://twitter.com/reutersagency', 'https://twitter.com/abc', 'https://twitter.com/bloombergnews', 'https://twitter.com/bw', 'https://twitter.com/time']

screen_names = ['nytimes', 'thesunnewspaper', 'thetimes', 'ap', 'cnn', 'bbcnews', 'cnet', 'msnuk', 'telegraph', 'usatoday', 'wsj', 'washingtonpost', 'bostonglobe', 'newscomauhq', 'skynews', 'sfgate', 'ajenglish', 'independent', 'guardian', 'latimes', 'reutersagency', 'abc', 'bloombergnews', 'bw', 'time']

run = 0 # Indicates if it is running for the first time
sys.stdout = codecs.getwriter(sys.stdout.encoding)(sys.stdout, errors='replace')

latest_dict = {}
tweet_filename = 'tweets3.csv'

# Run service
while True:
	for sn in screen_names:
		tweets = []
		if run <= len(screen_names):
			# Download max allowed tweets
			print 'Downloading for ...', sn
			try:
				print 'Trying to download'
				new_tweets = api.user_timeline(screen_name = sn, count = 200, include_rts = True)
			except:
				run = run + 1
				#print 'Error, couldnt get the Twitter user'
				continue
			#print 'Crossed this point'
			while len(new_tweets) > 0:
				# Save most recent tweets
				tweets.extend(new_tweets)
				
				# Update the id of the oldest tweet less one
				oldest = tweets[-1].id - 1
				print "getting tweets before %s" % (oldest)
				
				#all subsequent requests use the max_id param to prevent duplicates
				new_tweets = api.user_timeline(screen_name = sn, count = 200, max_id = oldest)
				
				print "...%s tweets downloaded so far" % (len(tweets))
				latest_dict[sn] = tweets[0].id
			print len(tweets)
				
		else:
			# Only get the new tweets that we don't yet have
			print 'Downloading for ...', sn
			try:
				new_tweets = api.user_timeline(screen_name = sn, count = 200, include_rts = True, since_id = latest_dict[sn])
			except:
				continue
			
			while len(new_tweets) > 0:
				latest_dict[sn] = new_tweets[0].id
			
				# Add the newest tweets to our running set
				tweets.extend(new_tweets)
				
				print "Getting tweets since %s" % (latest_dict[sn])

				new_tweets = api.user_timeline(screen_name = sn, count = 200, since_id = latest_dict[sn])
				
				print "...%s tweets downloaded so far" % (len(tweets))

		# Flatten into list for CSV

		with open(tweet_filename,'ab') as out_file:
			writing = csv.writer(out_file, delimiter=',', quotechar='"')
			for tw in tweets:
				tw_list = [tw.id, tw.text, sn, tw.retweet_count]
				writing.writerow([unicode(s).encode("utf-8") for s in tw_list])
		# Update run counter
		run = run + 1
		
	# Sleep for 10 mins before cycling through all news providers again
	time.sleep(600)