import re
from nltk.corpus import stopwords

class Tokenizer:
    """Tokenize, Clean and Generate a context for an initial text input.

    Attributes:
        text (str):         Raw text input on which to perform tokenization.
        tokens (List[str]): Generated list of tokens from the text.

    """

    def __init__(self, text):
        """Initialize the tokenizer with the initial `text` input.
        
        Args:
            text (str): text input on which to perform tokenization.

        """
        self.text = text.lower()
        self.tokens = []
        self.generateTokens()

    def generateTokens(self):
        """Generate the tokens for the corresponding `text`."""
        rgx = re.compile("([a-zA-Z0-9][a-zA-Z0-9']*[a-zA-Z0-9])")
        self.tokens = re.findall(rgx, self.text)

    def removeStopWords(self):
        """Remove the stopwords from the corresponding `tokens`."""
        stop_words = stopwords.words('english')
        self.tokens = [token for token in self.tokens if token not in stop_words]

    def prepareCtx(self, prependContextSize, appendContextSize):
        """Prepare the `token` list to perform the context search.

        Append and prepend the `tokens` list attribute with as many ' ' as the contextSize requires.

        Args:
            prependContextSize (int): Number of words considered in context prepending.
            appendContextSize (int): Number of words considered in context appending.

        """
        N = len(self.tokens)
        temp = [' '] * (prependContextSize + N + appendContextSize)
        temp[prependContextSize : (prependContextSize + N)] = self.tokens
        self.tokens = temp

if __name__ == "__main__":
    prependContextSize = 3
    appendContextSize = 4
    tk = Tokenizer('Hello world how is it going today ?')
    print(tk.tokens)
    tk.removeStopWords()
    print(tk.tokens)
    tk.prepareCtx(prependContextSize, appendContextSize)
    print(tk.tokens)