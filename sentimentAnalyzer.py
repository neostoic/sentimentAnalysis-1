# coding: utf-8

# # Predicting sentiment from product reviews
# 
# # Fire up GraphLab Create
import graphlab

# # Read some product review data
# 
# ### Loading reviews for a set of baby products.
products = graphlab.SFrame('amazon_baby.gl/')

# # Let's explore this data together
# 
# Data includes the product name, the review text and the rating of the review. 
products.head()

# #Build the word count vector for each review
products['word_count'] = graphlab.text_analytics.count_words(products['review'])
products.head()

# # Examining the reviews for most-sold product:  'Vulli Sophie the Giraffe Teether'
giraffe_reviews = products[products['name'] == 'Vulli Sophie the Giraffe Teether']
len(giraffe_reviews)

# # Build a sentiment classifier
products['rating'].show(view='Categorical')

# ## Define what's a positive and a negative sentiment
# 
# We will ignore all reviews with rating = 3, since they tend to have a neutral sentiment.  Reviews with a rating of 4 or higher will be considered positive, while the ones with rating of 2 or lower will have a negative sentiment.   
# ignore all 3* reviews
products = products[products['rating'] != 3]

# positive sentiment = 4* or 5* reviews
products['sentiment'] = products['rating'] >= 4

# ## Let's train the sentiment classifier
train_data, test_data = products.random_split(.8, seed=0)
sentiment_model = graphlab.logistic_classifier.create(train_data,
                                                      target='sentiment',
                                                      features=['word_count'],
                                                      validation_set=test_data)

# # Evaluate the sentiment model
sentiment_model.evaluate(test_data, metric='roc_curve')

# #Applying the learned model to understand sentiment for Giraffe
giraffe_reviews['predicted_sentiment'] = sentiment_model.predict(giraffe_reviews, output_type='probability')
giraffe_reviews.head()

# ##Sort the reviews based on the predicted sentiment and explore
giraffe_reviews = giraffe_reviews.sort('predicted_sentiment', ascending=False)
giraffe_reviews.head()

# ##Most positive reviews for the giraffe
giraffe_reviews[0]['review']
giraffe_reviews[1]['review']

# ##Show most negative reviews for giraffe
giraffe_reviews[-1]['review']
giraffe_reviews[-2]['review']


# Define function to count specific words in review
def awesome_count(review):
    count = 0
    for words in review:
        if words == 'awesome':
            count += 1
        else:
            continue
    return count


def great_count(review):
    count = 0
    for word in review:
        if word == 'great':
            count += 1
        else:
            continue
    return count


def fantastic_count(review):
    count = 0
    for word in review:
        if word == 'fantastic':
            count += 1
        else:
            continue
    return count


def amazing_count(review):
    count = 0
    for word in review:
        if word == 'amazing':
            count += 1
        else:
            continue
    return count


def love_count(review):
    count = 0
    for word in review:
        if word == 'love':
            count += 1
        else:
            continue
    return count


def horrible_count(review):
    count = 0
    for word in review:
        if word == 'horrible':
            count += 1
        else:
            continue
    return count


def bad_count(review):
    count = 0
    for word in review:
        if word == 'bad':
            count += 1
        else:
            continue
    return count


def terrible_count(review):
    count = 0
    for word in review:
        if word == 'terrible':
            count += 1
        else:
            continue
    return count


def awful_count(review):
    count = 0
    for word in review:
        if word == 'awful':
            count += 1
        else:
            continue
    return count


def wow_count(review):
    count = 0
    for word in review:
        if word == 'wow':
            count += 1
        else:
            continue
    return count


def hate_count(review):
    count = 0
    for word in review:
        if word == 'hate':
            count += 1
        else:
            continue
    return count


# Define list of selected words
selected_words = ['awesome', 'great', 'fantastic', 'amazing', 'love', 'horrible', 'bad', 'terrible', 'awful', 'wow',
                  'hate']
# Define list of functions to count number of selected words
selected_functions = [awesome_count, great_count, fantastic_count, amazing_count, love_count, horrible_count, bad_count,
                      terrible_count, awful_count, wow_count, hate_count]
# Iterate to find count
for i in range(0, 11):
    word = selected_words[i]
    func_name = selected_functions[i]
    products[word] = products['word_count'].apply(func_name)


# View products to see new columns with count for selected words
products.head()

# View total number of times each word is used in different reviews
for i in range(0, 11):
    x = products[selected_words[i]].sum()
    print " %s -> %s" % (selected_words[i], x)

# Create sentiment classifier with selected word count as features
selected_words_model = graphlab.logistic_classifier.create(train_data,
                                                           target='sentiment',
                                                           features=selected_words,
                                                           validation_set=test_data)
# View the coefficients assigned to different words selected
selected_words_model['coefficients']

# Sort the coefficients
selected_words_model['coefficients'].sort('value', ascending=False)

# Evaluate selected words model with test dataset
selected_words_model.evaluate(test_data)
# Evaluate all words model with test dataset
sentiment_model.evaluate(test_data)

# Create subset of products reviews
diaper_champ_reviews = products[products['name'] == 'Baby Trend Diaper Champ']

# Predict the sentiment using sentiment_model
diaper_champ_reviews['predicted_sentiment'] = sentiment_model.predict(diaper_champ_reviews, output_type='probability')

# Find difference in prediction of two models
diaper_champ_reviews = diaper_champ_reviews.sort('predicted_sentiment', ascending=False)
selected_words_model.predict(diaper_champ_reviews[0:1], output_type='probability')
