#Let's import required library
import graphlab

#Load data into SFrame
products = graphlab.SFrame('amazon_baby.gl/')

#Let's count words in reviews
products['word_count'] = graphlab.text_analytics.count_words(products['review'])

# Let's consider ratings 4 and 5 as postive rating and 1 and 2 as negative.
# this variable will be our target variable
products['sentiment'] = products['rating'] >= 4

#divide dataset into training and testing dataset
train_data, test_data = products.random_split(.8, seed=0)

