from collections import defaultdict
import numpy as np
from .tokennumrelation import TokenNumRelation

class ProbArray:
    """Compute and manage the co-occurance probability matrix.

    Attributes:
        tokencooccurance (Dict[Dict[int]]):         Number of occurance of each context token for each token
        sumContextToken (Dict[int]):                Total number of context token for each token.
        tokenNumRelation (class(TokenNumRelation)): Manage the relation between token and number.

    """

    def __init__(self):
        """Initialize all the attributes."""
        self.tokencooccurance = defaultdict(lambda: defaultdict(int))
        self.sumContextToken = defaultdict(int)
        self.tokenNumRelation = TokenNumRelation()

    def addContext(self, token, contextToken):
        """Add a `contextToken` to a given `token`.
        
        It first transforms or returns a number using the `tokenNumRelation` class instance before
        adding the given number to the co-occurance dictionary.

        Args:
            token (str):        Token to which we are adding a context token.
            contextToken (str): Context token to add as context token.

        """

        tokenNum = self.tokenNumRelation.getNum(token, training=True)
        contextTokenNum = self.tokenNumRelation.getNum(contextToken, training=True)
        self.tokencooccurance[tokenNum][contextTokenNum]+=1
        self.sumContextToken[tokenNum]+=1

    def makeVectorForNum(self, tokenNum):
        """Generate a vector of probability of co-occurance for a given token number.
        
        Args:
            tokenNum (int): Token number for which we want to compute the probability vector.

        """

        count = float(self.sumContextToken[tokenNum])
        probarray = np.zeros((self.tokenNumRelation.maxnum,1))
        for (contextTokenNum, times) in self.tokencooccurance[tokenNum].items():
            probarray[contextTokenNum] = times/count
        return np.sqrt(probarray)

    def makeVector(self, token):
        """Generate a vector of probability of co-occurance for a given token str.
        
        Args:
            token (str): Token str for which we want to compute the probability vector.
        
        """

        tokenNum = self.tokenNumRelation.getNum(token) # not in training anymore.
        if tokenNum is None:
            return None
        return self.makeVectorForNum(tokenNum)

    def getalltokenvecs(self):
        """Generate vectors of probability of co-occurance for each token number stored."""
        for num in self.tokencooccurance.keys():
            yield (num,self.makeVectorForNum(num))