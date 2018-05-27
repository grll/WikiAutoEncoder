from collections import defaultdict
import numpy as np
from .token_num_relation import TokenNumRelation

class ProbArray:
    """Compute and manage the co-occurance probability matrix.

    Attributes:
        tokencooccurance (Dict[Dict[int]]): Number of occurance of each context token for each token
        sum_ctx_token (Dict[int]): Total number of context token for each token.
        token_num_relation (class(TokenNumRelation)): Manage the relation between token and number.
    """
    def __init__(self):
        """Initialize all the attributes."""
        self.tokencooccurance = defaultdict(lambda: defaultdict(int))
        self.sum_ctx_token = defaultdict(int)
        self.token_num_relation = TokenNumRelation()

    def add_ctx(self, token, ctx_token):
        """Add a `ctx_token` to a given `token`.
        
        It first transforms or returns a number using the `tokenNumRelation` class instance before
        adding the given number to the co-occurance dictionary.

        Args:
            token (str): Token to which we are adding a context token.
            ctx_token (str): Context token to add as context token.
        """
        token_num = self.token_num_relation.get_num(token, training=True)
        ctx_token_num = self.token_num_relation.get_num(ctx_token, training=True)
        self.tokencooccurance[token_num][ctx_token_num]+=1
        self.sum_ctx_token[token_num]+=1

    def make_vector_for_num(self, token_num):
        """Generate a vector of probability of co-occurance for a given token number.
        
        Args:
            token_num (int): Token number for which we want to compute the probability vector.
        """
        count = float(self.sum_ctx_token[token_num])
        prob_array = np.zeros((self.token_num_relation.maxnum,1))
        for (ctx_token_num, times) in self.tokencooccurance[token_num].items():
            prob_array[ctx_token_num] = times/count
        return np.sqrt(prob_array)

    def make_vector(self, token):
        """Generate a vector of probability of co-occurance for a given token str.
        
        Args:
            token (str): Token str for which we want to compute the probability vector.
        """
        token_num = self.token_num_relation.get_num(token) # not in training anymore.
        if token_num is None:
            return None
        return self.make_vector_for_num(token_num)

    def get_all_token_vecs(self):
        """Generate vectors of probability of co-occurance for each token number stored."""
        for num in self.tokencooccurance.keys():
            yield (num, self.make_vector_for_num(num))

    def get_all_tokens_matrix(self):
        """Return the matrix of co-ocurance for each token."""
        maxnum = self.token_num_relation.maxnum
        matrix = np.empty([maxnum, maxnum], float)
        for num in range(maxnum):
            matrix[num] = np.transpose(self.make_vector_for_num(num))
        return matrix

    def generate_random_batch(self, batch_size):
        """Generate random batches of size batch size

        Return a Generator which yield batches of size batch_size. Last bactch might be smaller.

        Args:
            batch_size (int): Size of the batches to return.
        """
        maxnum = self.token_num_relation.maxnum
        random_shuffle_indices = np.arange(maxnum)
        np.random.shuffle(random_shuffle_indices)
        for i in range(0, maxnum, batch_size):
            true_batch_size = min(batch_size, maxnum-i)
            batch = np.empty([true_batch_size, maxnum], float)
            
            for j in range(len(batch)):
                random_index = random_shuffle_indices[i+j]
                batch[j] = np.transpose(self.make_vector_for_num(random_index))
            
            yield batch