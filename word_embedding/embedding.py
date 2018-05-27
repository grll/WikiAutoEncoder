import dill
import numpy as np
import pickle
from sklearn.decomposition import PCA

from .prob_array import ProbArray
from .tokenizer import Tokenizer
from .autoencoder import AutoEncoder

class Embedding:
    """Manage the embedding of articles with 3 different compression level.
    
    `Embedding` perform on a given `text_generator` representing an article the whole datapipeline.
    It then compute the co-occurance probability for this given article and generate the compressed
    data wanted either as a generator or an array.

    Attributes:
        text_generator (Func): Function that return a generator which represents each article.
        prob_array (Obj[ProbArray]): Instance of the ProbArray class representing the article.
        pca (ndarray)
    """
    def __init__(self, text_generator):
        """Initialize the attributes.
        
        Args:
            text_generator (Func): Return a generator which generate each article as text.
        """
        self.text_generator = text_generator
        self.prob_array = ProbArray()

    def create_prob_array(self, prep_ctx_size=3, app_ctx_size=3):
        """ Generate the co-occurance probability for each token in each article.

        Args:
            prep_ctx_size (int): Context size prepending each token (default is 3).
            app_ctx_size (int): Context size appending each token (default is 3).
        """
        for _, article in self.text_generator():
            if len(article) > 0:
                tokenizer = Tokenizer(article)
                tokenizer.remove_stop_words()
                tokenizer.prepare_ctx(prep_ctx_size, app_ctx_size)
                for i in range(prep_ctx_size, (prep_ctx_size + tokenizer.N)):
                    for j in range(i-prep_ctx_size, i+app_ctx_size+1):
                        if j != i and tokenizer.tokens[j] != ' ':
                            self.prob_array.addContext(tokenizer.tokens[i], tokenizer.tokens[j])

    def save_prob_array(self, filename=None):
        """Save a previously generated co-occurance probability array.
        
        Args:
            filename (str): filename under which to save the prob_array.
        """
        fname = filename or "probArray.pickle"
        with open('./saved_states/pickles/' + fname, 'wb') as f:
            dill.dump(self.prob_array, f)

    def load_prob_array(self, filename=None):
        """Load a previously generated co-occurance probability array.
        
        Args:
            filename (str): filename under which to load the prob_array.
        """
        fname = filename or "probArray.pickle"
        self.prob_array = dill.load(open('./saved_states/pickles/' + fname, "rb"))

    def init_auto_encoder(self, skip_training=True):
        """Initialize the `AutoEncoder` class and variables (required to use it).

        Args:
            skip_training (bool): Wether to train the NN from scratch or load it from file.
        """
        self.auto_encoder = AutoEncoder(self.prob_array.tokenNumRelation.maxnum)
        if skip_training is False:
            self.auto_encoder.trainNN(self.prob_array.generate_random_batch, num_steps=5)

    def generate_auto_encoded_articles(self):
        """Create a generator of auto_encoded articles from a trained `AutoEncoder`."""
        for category, article in self.text_generator():
            if len(article) > 0:
                tokenizer = Tokenizer(article)
                tokenizer.remove_stop_words()

                vectors = np.empty(
                    [len(tokenizer.tokens), self.prob_array.tokenNumRelation.maxnum], float)
                for i, token in enumerate(tokenizer.tokens):
                    vectors[i] = np.transpose(self.prob_array.makeVector(token))

                embeddings = self.auto_encoder.create_embedding(vectors)
                yield (category, np.sum(embeddings, axis=0))

    def create_encoded_article_array(self):
        """Return an array witch each probability of co-occurance for each articles auto-encoded."""
        self.auto_encoder.restore_state()
        encoded_articles = []
        for cat, encoded_article in self.generate_auto_encoded_articles():
            encoded_articles.append({ 'category': cat, 'data': encoded_article })
        return encoded_articles

    def init_pca(self, n_components=128, from_file=True):
        """Init the PCA by loading it from file or computing it.
        
        Args:
            n_components: Number of dimension to keep when computing the PCA.
            from_file (bool): Weather to load it from_file or computing it again.
        """
        if from_file:
            self.pca = pickle.load(open('./saved_states/pickles/PCA.pickle', "rb"))
            print("PCA matrix loaded from file.")
        else:
            pca = PCA(n_components)
            self.pca = pca.fit_transform(self.prob_array.get_all_tokens_matrix())
            with open('./saved_states/pickles/PCA.pickle', 'wb') as f:
                pickle.dump(self.pca, f)
            print("PCA matrix saved to file.")

    def generate_pca_articles(self):
        """Create a generator of pca reduced articles from the previously stored pca."""
        for category, article in self.text_generator():
            if len(article) > 0:
                tokenizer = Tokenizer(article)
                tokenizer.remove_stop_words()

                vectors = np.empty([len(tokenizer.tokens), self.pca.shape[1]], float)
                for i, token in enumerate(tokenizer.tokens):
                    num = self.prob_array.tokenNumRelation.getNum(token)
                    vectors[i] = self.pca[num]
                
                yield (category, np.sum(vectors, axis=0))
    
    def create_pca_article_array(self):
        """Return an array witch each probability of co-occurance for each articles pca reduced."""
        pca_articles = []
        for cat, pca_article in self.generate_pca_articles():
            pca_articles.append({ 'category': cat, 'data': pca_article })
        return pca_articles

    def generate_raw_articles(self):
        """Create a generator of articles represented as a sum of probability of co-occurance."""
        for category, article in self.text_generator():
            if len(article) > 0:
                tokenizer = Tokenizer(article)
                tokenizer.remove_stop_words()

                vectors = np.empty(
                    [len(tokenizer.tokens), self.prob_array.tokenNumRelation.maxnum], float)
                for i, token in enumerate(tokenizer.tokens):
                    vectors[i] = np.transpose(self.prob_array.makeVector(token))

                yield (category, np.sum(vectors, axis=0))

    def create_raw_articles_array(self):
        """Return an array witch each probability of co-occurance for each articles."""
        raw_articles = []
        for cat, raw_article in self.generate_raw_articles():
            raw_articles.append({ 'category': cat, 'data': raw_article })
        return raw_articles
