import re
from nltk.corpus import stopwords

class Tokenizer:
    """Tokenize, Clean and Generate a context for an initial text input.

    Attributes:
        text (str):         Raw text input on which to perform tokenization.
        tokens (List[str]): Generated list of tokens from the text.
        N (int):            Number of tokens.
    """
    def __init__(self, text):
        """Initialize the tokenizer with the initial `text` input.
        
        Args:
            text (str): text input on which to perform tokenization.
        """
        self.text = text.lower()
        self.tokens = []
        self.generate_tokens()
        self.N = len(self.tokens)

    def generate_tokens(self):
        """Generate the tokens for the corresponding `text`."""
        rgx = re.compile("([a-zA-Z0-9][a-zA-Z0-9']*[a-zA-Z0-9])")
        self.tokens = re.findall(rgx, self.text)

    def remove_stop_words(self):
        """Remove the stopwords from the corresponding `tokens`."""
        stop_words = stopwords.words('english')
        self.tokens = [token for token in self.tokens if token not in stop_words]
        self.N = len(self.tokens)

    def prepare_ctx(self, prep_ctx_size, app_ctx_size):
        """Prepare the `token` list to perform the context search.

        Append and prepend the `tokens` list attribute with as many ' ' as the contextSize requires.

        Args:
            prep_ctx_size (int): Number of words considered in context prepending.
            app_ctx_size (int): Number of words considered in context appending.
        """
        temp = [' '] * (prep_ctx_size + self.N + app_ctx_size)
        temp[prep_ctx_size : (prep_ctx_size + self.N)] = self.tokens
        self.tokens = temp

