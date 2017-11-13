import numpy as np
import re

f = open('doc-topics', 'r')
num_topics = 400
medicine_count = 0
hockey_count = 0
medicine_topics = np.zeros((num_topics))
hockey_topics = np.zeros((num_topics))
for line in f:
	items = re.split('\s+', line.strip())
	m = re.split('/',items[1])
	newsgroup = m[-2]
	if newsgroup == 'sci.med':
		this_topic = np.array(items[2:], dtype='float64')
		medicine_topics += this_topic
		medicine_count += 1
	if newsgroup == 'rec.sport.hockey':
		this_topic = np.array(items[2:], dtype='float64')
		hockey_topics += this_topic
		hockey_count += 1

top_n = 3
avg_medicine_topics = medicine_topics / medicine_count
top_medicine_inds = avg_medicine_topics.argsort()[-top_n:][::-1]
avg_hockey_topics = hockey_topics / hockey_count
top_hockey_inds = avg_hockey_topics.argsort()[-top_n:][::-1]

f_topics = open('topic-keys', 'r')
lines = f_topics.readlines()

print "Top Medicine Topics:"
for i, ind in enumerate(top_medicine_inds):
	print "#%i: %s" %(i+1, lines[ind])

print "Top Hockey Topics:"
for i, ind in enumerate(top_hockey_inds):
	print "#%i: %s" %(i+1, lines[ind])