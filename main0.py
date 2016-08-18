from __future__ import division
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
import string
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD, NMF
from numpy.random import rand, RandomState
from numpy import array, matrix, linalg
from sklearn.cluster import KMeans, MiniBatchKMeans
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import graphlab
from graphlab import SFrame
from sklearn.metrics.pairwise import cosine_similarity



def read_data(filename):
    '''
    Gets the input file and do some preprocessing and output it as a dataframe

    INPUT: The name and address of the csv file with the post informtion
    OUTPUT: dataframe, forum post text, and data frame of the extra features
    '''
    data = pd.read_csv(filename, sep='\t', encoding='utf-16', header=3)
    df = data[['Message Subject','Message Body No HTML','Parent Node',\
    'Replies/Comments','Number of Views']]
    # converting the unicode to string
    df['Message Body No HTML'] = df['Message Body No HTML'].fillna('').apply(lambda x: x.encode('ascii','ignore'))
    message_body = df['Message Body No HTML']
    df_side_feat = df[['Replies/Comments','Number of Views']]
    return df, message_body, df_side_feat

def read_data_masked(filename):
    '''
    Only gets the orginal posts and exclude the Replies.

    INPUT: The name and address of the csv file with the post informtion
    OUTPUT: dataframe, forum post text, and data frame of the extra features, and the URLs
    '''

    # Index([u'Message Subject', u'Message Body', u'Message Body No HTML',
    #    u'Message URL', u'Parent Node', u'Node Type', u'Author',
    #    u'Time of Post', u'Replies/Comments', u'Tags', u'Labels', u'Statuses',
    #    u'Status Names', u'Me Too', u'Kudos', u'5-star Rating',
    #    u'Number of Views', u'Parent Message URL', u'Root Message URL',
    #    u'Accepted As Solution', u'Images', u'Videos', u'Attachments',
    #    u'Read by Moderator', u'Message Teaser', u'Marked Read Only',
    #    u'Last Edit Author', u'Last Edit Time', u'TKB Helpful',
    #    u'TKB Not Helpful'],
    #   dtype='object')
    data = pd.read_csv(filename, sep='\t', encoding='utf-16', header=3)
    mask = data['Message Subject'].apply(lambda x:x[0:2] != 'Re')
    df = data[['Message Subject','Message Body No HTML','Message URL','Parent Node',\
    'Replies/Comments','Number of Views']][mask]
    # converting the unicode to string
    df['Message Body No HTML'] = df['Message Body No HTML'].fillna('').apply(lambda x: x.encode('ascii','ignore'))
    message_body = df['Message Body No HTML']
    message_URL = df['Message URL']
    df_side_feat = df[['Replies/Comments','Number of Views']]
    return df, message_body, df_side_feat, message_URL

def reconst_mse(X, W, H):
    return (array(X - W.dot(H))**2).mean()

def describe_nmf_results(document_term_mat, W, H, n_top_words=15):
    '''
    Showing the topics and the most important words associated with those topics
    '''
    print("Reconstruction error: %f") %(reconst_mse(document_term_mat, W, H))
    # each row of H matrix belongs to a topic, find the highest coeff and type associated words.
    for topic_num, topic in enumerate(H):
        print("Topic number %d:" % topic_num)
        print(" ".join([feature_words[i] \
                for i in topic.argsort()[:-n_top_words - 1:-1]]))
    return


def stop_list():
    stop_list = set(stopwords.words('english') + \
                ["n't", "'s", "'m", "ca", "'re", "quot",'thanks','thank','hi','hello','autodesk'\
                ,'adsk', '&quot;']
                + list(ENGLISH_STOP_WORDS))

    # List of symbls we don't care about
    symbls = " ".join(string.punctuation).split(" ")
    symbls += [""]

    # Full set of stops
    stops = stop_list ^ set(symbls)
    return stops

def tokenize(text):
    PUNCTUATION = set(string.punctuation)
    STOPWORDS = set(stopwords.words('english') + ["n't", "'s", "'m" , "'re"] + \
                ["quot","thanks","thank","hi","hello"] + list(ENGLISH_STOP_WORDS))
    STEMMER = PorterStemmer()
    tokens = word_tokenize(text)
    lowercased = [t.lower() for t in tokens]
    no_punctuation = []
    for word in lowercased:
        punct_removed = ''.join([letter for letter in word if not letter in PUNCTUATION])
        no_punctuation.append(punct_removed)
    no_stopwords = [w for w in no_punctuation if not w in STOPWORDS]
    stemmed = [STEMMER.stem(w) for w in no_stopwords]
    return [w for w in stemmed if w]

def tf_idf(stop_lists,message_body):
    # vectorizer = TfidfVectorizer( stop_words='english')
    # vectorizer = TfidfVectorizer( stop_words='english',tokenizer=tokenize)
    # vectorizer = TfidfVectorizer(tokenizer=tokenize)
    vectorizer = TfidfVectorizer( stop_words=stop_lists)
    document_term_mat = vectorizer.fit_transform(message_body)
    feature_words = vectorizer.get_feature_names()
    return vectorizer, document_term_mat, feature_words

def nmf_model(n_topics,document_term_mat):
    print("\n\n---------\n decomposition")
    nmf = NMF(n_components=n_topics, l1_ratio=0.0)
    W_sklearn = nmf.fit_transform(document_term_mat)
    H_sklearn = nmf.components_
    describe_nmf_results(document_term_mat, W_sklearn, H_sklearn)
    return W_sklearn, H_sklearn

def nmf_model2(n_topics,document_term_mat):
    # print("\n\n---------\n decomposition")
    nmf = NMF(n_components=n_topics, l1_ratio=0.0)
    W_sklearn = nmf.fit_transform(document_term_mat)
    H_sklearn = nmf.components_
    # describe_nmf_results(document_term_mat, W_sklearn, H_sklearn)
    return W_sklearn, H_sklearn

def df_sf(df):
    sf = graphlab.SFrame(df[['user_id', 'joke_id', 'rating']])
    return df, sf


def rec_factorization(document_term_mat,side_data,n_topics):
    data = document_term_mat.toarray()
    rows = []
    cols = []
    values = []
    for row in xrange(len(data)):
        for col, value in enumerate(data[row]):
            rows.append(row)
            cols.append(col)
            values.append(value)
    d = {'doc':rows, 'word': cols, 'rating':values}
    new_df = pd.DataFrame(d)
    new_df[['doc','word']] = new_df[['doc','word']] + 1
    side_data = SFrame(data=side_data)
    sf = SFrame(data=new_df)
    rec = graphlab.recommender.factorization_recommender.create(
                sf,
                user_id='doc',
                item_id='word',
                user_data = side_data,
                target='rating',
                solver='als',
                num_factors =n_topics,
                side_data_factorization=True,
                # regularization=0.001,
                nmf=True);

    word_sf = rec['coefficients']['word'] #rec.get('coefficients') or rec.list_fields()
    doc_sf = rec['coefficients']['doc']

    H_graph = word_sf['factors'].to_numpy()
    H_graph = H_graph.T
    #H_graph.shape

    W_graph = doc_sf['factors'].to_numpy()
    describe_nmf_results(document_term_mat, W_graph, H_graph)
    return H_graph, W_graph


def post_recommender(n_related_posts=2):

    '''
    Recommends posts that may already have your answer

    INPUT: str, the question that you want to post in the forum, number of similar questions
    OUTPUT: dataframe, forum post text, and data frame of the extra features
    '''

    question = input("Please enter your question: ")
    vec_question = TfidfVectorizer(stop_words=stop_list(), vocabulary=feature_words)
    tfidf_matrix_question = vec_question.fit_transform([question]).toarray()
    cos_sim = np.empty((document_term_mat.shape[0],))
    for i in xrange(document_term_mat.shape[0]):
        temp_sim = cosine_similarity(tfidf_matrix_question[0].reshape(1, -1), \
        document_term_mat.toarray()[i].reshape(1, -1))
        cos_sim[i] = temp_sim
    print [message_body.iloc[i] for i in np.argsort(cos_sim)[:-n_related_posts - 1:-1]]


def issue_detection(matrix, W, H, num_posts=2):

    '''
    Shows the most important posts that contain the selected keyword from the selected topic

    INPUT: tfidf_matrix, W, H
    OUTPUT: Most relavant post to the selected topic with the selcted keyword
    '''

    issue_question = input("Please enter the key word/words you are intrested in: ")
    # issue_question = list(issue_question)
    topic_num = input("Please enter the topic number: ")
    ind = feature_words.index(issue_question)
    test_m = document_term_mat.toarray()
    contained_docs = test_m[:,ind]
    docs_ind = np.nonzero(contained_docs)  # indices are from the masked data frame
    docs_list = message_body.iloc[docs_ind]
    issue_number = docs_list.shape[0]

    ind_doc_max_s = np.argsort(W_sklearn[docs_ind[0],topic_num])[:-num_posts-1:-1]
    # message_body.iloc[docs_ind[0][ind_doc_max_s]] # other way to find the questions
    # top_posts = docs_list.iloc[ind_doc_max_s]
    top_posts = zip(message_URL.iloc[docs_ind[0][ind_doc_max_s]],docs_list.iloc[ind_doc_max_s])

    print top_posts





if __name__ == '__main__':

    filename = 'data/csv_search-3644074-1470267303460.csv'


    # df, message_body,df_side_feat = read_data(filename)
    df, message_body,df_side_feat, message_URL = read_data_masked(filename)

    vectorizer, document_term_mat, feature_words = tf_idf(stop_list(),message_body)

    n_features = 15
    n_topics = 5

    # W_sklearn, H_sklearn = nmf_model(n_topics,document_term_mat)

    # using graphlab recommender
    H_graph, W_graph = rec_factorization(document_term_mat,df_side_feat,n_topics)
    W_sklearn, H_sklearn = nmf_model(n_topics,document_term_mat)
    top_posts = issue_detection(document_term_mat, W_sklearn, H_sklearn)
    post_recommender()




    # try:
    #      ...:     y = float(x)
    #      ...: except:
    #      ...:     y = None
