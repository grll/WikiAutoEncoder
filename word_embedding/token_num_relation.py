class TokenNumRelation:
    """Create and manage a one to one relation between token and number.

    Attributes:
        maxnum (int):         Represent the current highest number representation for a token.
        t2n (Dict[str, int]): Matching assigning a number to a token.
        n2t (Dict[int, str]): Matching assigning a token to a number.
    """
    def __init__(self):
        """Initialize the attributes."""
        self.maxnum = 0
        self.t2n = {}
        self.n2t = {}
        
    def get_num(self, token, training=False):
        """Get the number corresponding to a given token or create one.
        
        Note:
            New numbers are not assigned to token after training so when `training` is False new
            token immediatly returns None.

        Args:
            token (str): String representing the token queried.
            training (bool): Boolean representing whether training if over or not.
        """
        if token in self.t2n:
            return self.t2n[token]
        elif training:
            self.t2n[token] = self.maxnum
            self.n2t[self.maxnum] = token
            self.maxnum+=1
            return self.maxnum-1
        else:
            return None
        
    def get_token(self, num):
        """Get the token corresponding to a given number.

        Args:
            num (int): Int representing the number queried.
        """
        if num in self.n2t:
            return self.n2t[num]
        else:
            return None