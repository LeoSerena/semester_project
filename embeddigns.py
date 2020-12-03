import os
import sys

import pandas as pd
import numpy as np

import nltk
import pickle

import matplotlib.pyplot as plt

import re
import spacy

from spacy_lefff import LefffLemmatizer, POSTagger

from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec

from tensorboard.plugins import projector

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
    
    sizes = [25, 50, 100, 200, 300] # dimensionality of the embedded vectors
    window = 5          # max distance between current and predicted word
    min_count = 1       # minimum frequency of a word for it to be considered
    sg = 1              # trainig alg. 0: CBOW, 1: skip-gram
    negative = 5        # if >0, tells the number of negative samples
    ns_exponent = 0.75  # determines the distribution of the negative sampling. Between 0 and 1.
                        # The closer to 0, the closer to a uniform distribution, the closer to 1, the closer to the frequency distribution (0.75 is from original paper on Word2Vec)
    alpha = 0.0001      # initial learning rate
    min_alpha = 0.00001 # final learning rate, droping linearily
    sorted_vocab = 1    # if 1 sorts the vocab in descending frequency before assigning word indexes
    epochLogger = EpochLogger()
    callbacks = [epochLogger]

    i = 0

    for size in sizes:

        model = Word2Vec(
            sentences = serie,
            size = size,
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
        word_vectors.save(os.path.join('embeddings', 'word2vec_{}_{}.wordvectors'.format(size, sys.argv[1].split('/')[1])))
        
        vocab = model.wv.vocab.keys()
        embeddings = model[vocab]

        log_dir = os.path.join('logs', 'gensim_{}'.format(i))
        i = i + 1
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)


        with open(os.path.join(log_dir, 'metadata.tsv'), 'w') as f:
            for word in vocab:
                try:
                    f.write("{}\n".format(word))
                except UnicodeEncodeError:
                    f.write("{}\n".format('unknown'))

        weights = tensorflow.Variable(embeddings)
        checkpoint = tensorflow.train.Checkpoint(embeddings = weights)
        checkpoint.save(os.path.join(log_dir, 'embedding.ckpt'))

        config = projector.ProjectorConfig()
        embedding = config.embeddings.add()

        embedding.tensor_name = "embedding/.ATTRIBUTES/VARIABLE_VALUE"
        embedding.metadata_path = 'metadata.tsv'


def lambda_remove_helper(tokens, to_rmv):
    for w in to_rmv:
        if w in tokens:
            tokens.remove(w)
    return tokens

def remove_singletons(tokens_list, N = 1):
    """
    Removes all words that appear only once in the given array

    Parameters
    ----------
        tokens_list : pandas.Series of list of str
            the array to be processed
        
        N : int
            If the number of a word is smaller or equal to N
            it is removed
    
    Returns
    -------
        tokens : pandas.Series of list of str
            the cleaned array
    """
    assert(N > 0)
    T = {}
    for tokens in tokens_list:
        for token in tokens:
            if token in T.keys():
                T[token] = T[token] + 1
            else:
                T[token] = 1

    to_rmv = []
    for (w,n) in T.items():
        if(n > N+1):
            to_rmv.append(w)

    tokens_list = tokens_list.map(
        lambda tokens : lambda_remove_helper(tokens, to_rmv)
    )
    return tokens_list


def embeddings_keras(posts):
    posts = remove_singletons(posts)
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

    NUM_TOPN = 10
    
    EMBEDDINGS_DIM = [50, 100, 300]
    LEARNING_RATES = [0.2, 0.1, 0.05, 0.001]
    EPOCHS = 30

    for dim in EMBEDDINGS_DIM:
        for lr in LEARNING_RATES:
            EM = Embedding_model()

            EM.build(vocab_size, embeddings_dim = dim)
            EM.compile_(learning_rate = lr)

            validation_set = ['masque']
            validation_set = [vectorizer.get_vocabulary().index(w) for w in validation_set]

            history = EM.train(
                target_words,
                context_words,
                labels_,
                epochs = EPOCHS,
                callbacks = [
                    # ValidationCallback(validation_set, EM.model, reverse_mapping, NUM_TOPN = NUM_TOPN),
                    tensorflow.keras.callbacks.ReduceLROnPlateau(monitor = 'loss', factor = 0.1, min_delta = 1, cooldown = 1, min_lr = 0, patience = 5)
                    # tensorflow.keras.callbacks.EarlyStopping(monitor = 'loss', patience = 10, restore_best_weights = True)
                ]
            )
            path = os.path.join('embeddings', 'embeddings_{}_{}_{:.2f}_{}.txt'.format(dim, str(lr), history[-1], sys.argv[1][:-4]))
            
            weights_and_voc = {}
            weights_and_voc['embeddings'] = EM.model.get_layer('embeddings').weights[0].numpy()
            weights_and_voc['vocab'] = vocab
            weights_and_voc['history'] = history

            pickle.dump(weights_and_voc, open(path, 'wb'))
    

class Embedding_model():

    def __init__(self):
        pass

    def build(self, vocab_size, embeddings_dim = 100):
        
        # takes 2 numbers as input, integers of the target and context word
        self.in_1 = tensorflow.keras.Input(1)
        self.in_2 = tensorflow.keras.Input(1)
        # add an embedding layer
        E = layers.Embedding(
            vocab_size,
            embeddings_dim,
            # embeddings_regularizer = tensorflow.keras.regularizers.l1(l1 = 0.01),
            # embeddings_constraint = tensorflow.keras.constraints.UnitNorm(axis= 1),
            input_length = 1,
            name = 'embeddings'
        )
        # gives the inputs to the embedding layer
        E_1 = E(self.in_1)
        E_2 = E(self.in_2)
        
        N_1 = tensorflow.keras.backend.l2_normalize(E_1, axis = 1)
        N_2 = tensorflow.keras.backend.l2_normalize(E_2, axis = 1)

        # computes the cosine similarity
        D = tensorflow.keras.backend.sum(E_1 * E_2, axis = -1)
        # sigmoid activation function for non-linearity
        self.S = layers.Activation(activation = 'sigmoid')(D)

    def compile_(self, learning_rate = 0.1):
        self.model = tensorflow.keras.Model(
            inputs = [self.in_1, self.in_2],
            outputs = [self.S]
        )

        self.model.compile(
           loss = 'binary_crossentropy',
            optimizer = tensorflow.keras.optimizers.RMSprop(learning_rate = learning_rate)
        )

    def train(self, target_words, context_words, labels, batch_size = 32, epochs = 20, callbacks = [], plot = False):
        history = self.model.fit(
            [target_words, context_words],
            labels,
            batch_size = batch_size,
            epochs = epochs,
            callbacks = callbacks,
            shuffle = True
        )
        if plot:
            train_loss = history.history['loss']
            plt.figure()
            plt.xlabel('epochs')
            plt.ylabel('loss')
            plt.plot(train_loss)
            plt.show()

        return history.history['loss']

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
    
    df = df[df['lang'] == 'fr']
    df = df[~df['body'].isnull()]
    serie = df['body'].astype(str)
    l = []
    for body in serie:
        l.append(nltk.word_tokenize(body))
    embeddings_gensim(np.array(l))

