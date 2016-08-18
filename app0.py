from flask import Flask, request, render_template
import random
import requests
import ipdb
import pickle
from main import tf_idf, stop_list, describe_nmf_results, nmf_model2,reconst_mse
import numpy as np
import main
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# import ipdb; ipdb.set_trace()
global message_body
global message_URL
global feature_words
global document_term_mat

message_body = None
message_URL = None
feature_words = None
document_term_mat = None

# Initialize the app and load the pickled data.
#================================================
# init flask app
app = Flask(__name__)
data_dict = pickle.load(open('data/data_dict_df.pkl', 'r'))
products = data_dict.keys()


@app.context_processor
def inject_enumerate():
    return dict(enumerate=enumerate)


# Homepage with form on it.
#================================================
@app.route('/')
def index():
    return render_template('jumbotron.html', title='Welcome!', products=products)


#================================================
@app.route('/topics', methods=['POST'])
def topics():

    # get data from request form
    data = request.form['productname']
    #keys = '\n'.join([k for k in request.form])
    # convert data from unicode to string
    data = str(data)
    df = data_dict[data]

    global message_body
    message_body = df['Message Body No HTML']
    global message_URL
    message_URL = df['Message URL']
    df_side_feat = df[['Replies/Comments','Number of Views']]
    global feature_words
    global document_term_mat
    vectorizer, document_term_mat, feature_words = tf_idf(stop_list(),message_body)
    n_features = 15
    n_topics = 5
    n_top_words = 10
    W_sklearn, H_sklearn = nmf_model2(n_topics,document_term_mat)
    topics_num = []
    words = []
    for topic_num, topic in enumerate(H_sklearn):
        topics_num.append("Topic number %d:" % topic_num)
        words.append(" ".join([feature_words[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))

    return render_template('jumbotron0.html', product_name=data, topics=zip(topics_num,words),\
    feature_words=feature_words)

@app.route('/keyword', methods=['POST'])
def keyword():

    # get data from request form
    issue_question = request.form['keyword']
    # keys = '\n'.join([k for k in request.form])
    # convert data from unicode to string
    issue_question = str(issue_question)
    topic_num = request.form['topic']
    num_posts = 2
    global message_body
    global message_URL
    vectorizer, document_term_mat, feature_words = tf_idf(stop_list(),message_body)
    n_features = 15
    n_topics = 5
    n_top_words = 10
    W_sklearn, H_sklearn = nmf_model2(n_topics,document_term_mat)
    ind = feature_words.index(issue_question)
    test_m = document_term_mat.toarray()
    contained_docs = test_m[:,ind]
    docs_ind = np.nonzero(contained_docs)[0]  # indices are from the masked data frame
    docs_list = message_body.iloc[docs_ind]
    # docs_list = message_body[docs_ind]
    issue_number = docs_list.shape[0]
    # docs_list = [message_body[i] for i in docs_ind]


    ind_doc_max_s = np.argsort(W_sklearn[docs_ind,topic_num])[:-num_posts-1:-1]
    x = message_URL.iloc[docs_ind[ind_doc_max_s]]
    y = docs_list.iloc[ind_doc_max_s]
    top_posts = zip(x,y)
    # top_posts = docs_list.iloc[ind_doc_max_s]

    return render_template('newpage.html', top_posts=top_posts)

@app.route('/question', methods=['POST'])
def question():

    # get data from request form
    question = request.form['question']
    #keys = '\n'.join([k for k in request.form])
    # convert data from unicode to string
    question = str(question)
    ##
    n_related_posts=3
    global feature_words
    vec_question = TfidfVectorizer(stop_words=stop_list(), vocabulary=feature_words)
    tfidf_matrix_question = vec_question.fit_transform([question]).toarray()
    global document_term_mat
    cos_sim = np.empty((document_term_mat.shape[0],))
    for i in xrange(document_term_mat.shape[0]):
        temp_sim = cosine_similarity(tfidf_matrix_question[0].reshape(1, -1), \
        document_term_mat.toarray()[i].reshape(1, -1))
        cos_sim[i] = temp_sim
    similar_posts = [message_body.iloc[i] for i in np.argsort(cos_sim)[:-n_related_posts - 1:-1]]

    return render_template('newpage0.html', similar_posts=similar_posts)

if __name__ == '__main__':

    app.run(host='0.0.0.0', port=8080, debug=True)
