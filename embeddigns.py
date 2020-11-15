import os
import sys

import pandas as pd
import numpy as np

import nltk

import re
import spacy

from spacy_lefff import LefffLemmatizer, POSTagger

from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec

import tensorflow
from tensorflow.keras import layers

from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

COLUMN_NAME = 'body'

def normalize(tokens):
    """
    Takes an input string and set it to lower characters,
    removes any nonalphanumerical character and links.

    Parameters
    ----------
        tokens : list of str
            The string to be processed

    Returns
    -------
        tokens : list of str
            The processed string
    """

    # tokens = nltk.word_tokenize(string)
    tokens = [w for w in tokens if w.isalpha()]
    return tokens

def remove_stopwords(tokens):
    """
    Removes all the french stopwords in the input array

    Parameters
    ----------
        tokens : list of str
            The token list

    Returns
    -------
        clean : list of str
            The token list without stopwords
    """
    stopwords = nltk.corpus.stopwords.words('french')
    clean = [x for x in tokens if x not in stopwords]
    return clean

def remove_hyperlink(string):
    return re.sub(r"http\S+", "", string)

def lemmatize(serie):
    """
    Takes the panda series and lemmatizes each word using
    the spacylefff lemmatizer

    Parameters
    ----------
        serie : pandas.series
            The column that is processes

    Returns
    -------
        lemmatized : pandas.series
            The lemmatized column
    """
    pos = POSTagger()
    french_lemmatizer = LefffLemmatizer(after_melt = True)
    
    nlp = spacy.load('fr_core_news_sm')
    nlp.add_pipe(pos, name = 'pos', after = 'parser')
    nlp.add_pipe(french_lemmatizer, name = 'lefff', after = 'pos')


    lemmatized = serie.map(
        lambda post : post.lower()
    ).map(
        remove_hyperlink
    ).map(
        lambda post : [doc.lemma_ for doc in nlp(post)]
    )
    return lemmatized

class EpochLogger(CallbackAny2Vec):
    """
    Callback wrapper printing loss at each epoch
    """

    def __init__(self):
        self.epoch = 0
        self.previous_loss = 0

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        if(self.epoch == 0):
             pass
        else:
            loss = loss - self.previous_loss

        self.previous_loss = loss
        print('Loss at EPOCH {} is {}'.format(self.epoch, loss))
        self.epoch = self.epoch + 1

def embeddings_gensim(serie):
    """
    Builds and trains a gensim word2vec model using the given list of tweets
    It then saves the words and the embeddings. Use gensim.models.KeyedVectors.load
    to load them back.
    
    Parameters
    ----------
        serie : pandas.Serie
            The list of all tweets

    """
    
    #size = 100          # dimensionality of the embedded vectors
    window = 5          # max distance between current and predicted word
    min_count = 0       # minimum frequency of a word for it to be considered
    sg = 1              # trainig alg. 0: CBOW, 1: skip-gram
    negative = 5        # if >0, tells the number of negative samples
    ns_exponent = 0.75  # determines the distribution of the negative sampling. Between 0 and 1.
                        # The closer to 0, the closer to a uniform distribution, the closer to 1, the closer to the frequency distribution (0.75 is from original paper on Word2Vec)
    alpha = 0.0001      # initial learning rate
    min_alpha = 0.00001 # final learning rate, droping linearily
    sorted_vocab = 1    # if 1 sorts the vocab in descending frequency before assigning word indexes
    epochLogger = EpochLogger()
    callbacks = [epochLogger]

    model = Word2Vec(
        sentences = serie,
        #size = size,
        window = window,
        min_count = min_count,
        sg = sg,
        negative = negative,
        ns_exponent = ns_exponent,
        alpha = alpha,
        min_alpha = min_alpha,
        sorted_vocab = sorted_vocab,
        compute_loss = True,
        callbacks = callbacks
    )
    word_vectors = model.wv
    word_vectors.save('word2vec.wordvectors')

    from nltk.cluster import KMeansClusterer

    NUM_CLUSTERS = 20

    kclusterer = KMeansClusterer(
        NUM_CLUSTERS,
        distance = nltk.cluster.util.cosine_distance,
        repeats = 5
    )

    assigned_clusters = kclusterer.cluster(
        model[model.wv.vocab],
        assign_clusters = True
    )

    V = np.array(list(model.wv.vocab.keys()))
    K = np.array(assigned_clusters)

    for i in range(NUM_CLUSTERS):
        print('-----------CLUSTER {}------------'.format(i))
        for v in V[K == i]:
            print(v)

def embeddings_keras(posts):
    posts = posts.map(
        lambda w : ' '.join(w)
    )
    training_data = posts.to_numpy(dtype = str)

    vectorizer = TextVectorization(output_mode = 'int')
    # Creates the vocabulary and the word to int mapping
    vectorizer.adapt(training_data)
    # list of tweets in integer representation
    integer_data = vectorizer(training_data)

    vocab = vectorizer.get_vocabulary()
    vocab_size = len(vocab)

    # created the reverse mapping int to word
    reverse_mapping = {}
    for i in range(vocab_size):
        reverse_mapping[i] = vocab[i]
    
    # to sample wrt word frequencies
    sampling_table = tensorflow.keras.preprocessing.sequence.make_sampling_table(vocab_size)
    
    tuples = []
    labels_ = []

    for integer_tweet in integer_data:
        (data_tuples, labels) = tensorflow.keras.preprocessing.sequence.skipgrams(
            integer_tweet.numpy(),
            vocab_size,
            window_size = 5,
            shuffle = True,
            sampling_table = sampling_table
        )
        if(len(data_tuples) == 0):
            pass
        else:
            tuples.append(data_tuples)
            labels_.append(labels)

    tuples = np.concatenate(tuples)
    labels_ = np.concatenate(labels_)
    
    # makes sure of proper casting
    d = [[int(x[0]), int(x[1])] for x in tuples]

    # separate in target and context words
    target_words = np.array(d)[:, 0]
    context_words = np.array(d)[:, 1]
    
    assert(len(target_words) == len(context_words))
    assert(len(target_words) == len(labels_))

    embed_dim = 100

    # takes 2 numbers as input, integers of the target and context word
    in_1 = tensorflow.keras.Input(1)
    in_2 = tensorflow.keras.Input(1)
    # add an embedding layer
    E = layers.Embedding(
        vocab_size,
        embed_dim,
        input_length = 1,
        name = 'embeddings'
    )
    # givest the inputs to the embedding layer
    E_1 = E(in_1)
    E_2 = E(in_2)
    # computes the cosine similarity
    D = tensorflow.keras.backend.sum(E_1 * E_2, axis = -1)
    # sigmoid activation function for non-linearity
    S = layers.Activation(activation = 'sigmoid')(D)

    model = tensorflow.keras.Model(
        inputs = [in_1, in_2],
        outputs = [S]
    )

    model.compile(loss = 'binary_crossentropy', optimizer = 'rmsprop')
    NUM_VALIDATION = 10
    NUM_TOPN = 10
    M = np.sum(sampling_table)
    validation_set = np.random.choice(vocab, NUM_VALIDATION, replace = False, p = sampling_table / M)
    validation_set = vectorizer(validation_set).numpy().ravel()
    
    print(labels)
    print('------------------')
    print(target_words)
    print('----------------------')
    print(context_words)
    model.fit(
        [target_words, context_words],
        labels_,
        batch_size = 64,
        epochs = 30,
        callbacks = [ValidationCallback(validation_set, model, reverse_mapping)],
        shuffle = True
    )

class ValidationCallback(tensorflow.keras.callbacks.Callback):
    
    def __init__(self, validation_set, model, reverse_mapping, NUM_TOPN = 10):
        self.validation_set = validation_set
        self.NUM_TOPN = NUM_TOPN
        self.model = model
        self.reverse_mapping = reverse_mapping
        super().__init__()
    
    def on_epoch_end(self, epoch, logs = None):
        W = self.model.get_layer('embeddings').weights[0].numpy()
        W = W / np.linalg.norm(W, axis = 1)[:, np.newaxis]
        val_vects = W[self.validation_set]

        closest = np.argsort(np.cos(W @ val_vects.T), axis = 0)

        top = closest[:self.NUM_TOPN, :]
        for (i, e) in enumerate(self.validation_set):
            print('validation word: {}'.format(self.reverse_mapping[e]))
            print('closest words:')
            string = ''
            for e in top[:, i]:
                string = string + ' ' + self.reverse_mapping[e]
            print(string)


if __name__ == '__main__':

    assert(len(sys.argv) == 2)
    
    file_name = sys.argv[1]

    df = pd.read_csv(file_name)

    posts = df[COLUMN_NAME]

    posts = lemmatize(posts)

    posts = posts.map(
        normalize
    ).map(
        remove_stopwords
    )

    embeddings_keras(posts)

