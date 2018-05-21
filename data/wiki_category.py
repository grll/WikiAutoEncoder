import requests
from data_helpers import batch

class WikiCategory:
    """Represent and fetch all articles related to a specific wikipedia Category.

    Attributes:
        category_str (str):             Used to query the wikipedia Category.
        pageids (List[int]):            Store the already fetched pageids.
        texts (List[str]):              Store the extracted texts corresponding to the pageids.
        subcategories_str (List[str]):  Store the query strings for the wikipedia subcategories.
        payload (Dict[str, str]):       Payload used to perform queries to wikipedia API.
        n (int):                        Number of articles to take in the wikipedia Category.

    """

    # Constant
    BASE_URL = "https://en.wikipedia.org/"

    def __init__(self, category_str, n=1000):
        """Initialize the wikipedia Category defined by the `category_str` query string.

        Initialize all the Attributes. Fetch pages immediatly accessible under the `category_str`
        using the `fetch_pageids` method. Fetch all the subcategories for the given category using
        the `fetch_subcategories` method.

        Args:
            category_str (str): Query string used to fetch the wikipedia Category.
            N (int, optional):  Number of articles to take in the wikipedia Category.

        """
        self.category_str = category_str
        self.pageids = []
        self.texts = []
        self.subcategories_str = []
        self.payload = {
            'action':  'query',
            'format':  'json',
            'list':    'categorymembers',
            'cmlimit': 'max'
        }
        self.n = n

        self.fetch_pageids()
        self.fetch_subcategories()


    def fetch_pageids(self, cmtitle=None):
        """Fetch the pageids of the specified Category on the wikipedia API endpoint.

        The pageids fetched by the method are immediately stored in the `pageids` attribute
        providing that they don't exist before. The method breaks out when `pageids` become
        bigger than the wanted number of articles `N`.

        Note:
            If no `cmtitle` argument is specified then the method will fetch pageids for the
            `category_str` attribute.

        Args:
            cmtitle (str, optional): Query string corresponding to the wikipedia Category to fetch.

        """
        self.payload['cmtitle'] = cmtitle or self.category_str
        self.payload['cmtype'] = 'page'

        data = requests.get(WikiCategory.BASE_URL + "/w/api.php", self.payload).json()
        pages = data['query']['categorymembers']
        pageids = [x['pageid'] for x in pages]

        for pageid in pageids:
            if not pageid in self.pageids:
                self.pageids.append(pageid)
                if len(self.pageids) == self.n:
                    break

    def fetch_subcategories(self):
        """Fetch and populate the subcategories of the wikipedia Category."""
        self.payload['cmtitle'] = self.category_str
        self.payload['cmtype'] = 'subcat'

        data = requests.get(WikiCategory.BASE_URL + "/w/api.php", self.payload).json()
        subcategories = data['query']['categorymembers']
        self.subcategories_str = [x['title'] for x in subcategories]

    def fetch_all_pageids(self):
        """Fetch all pageids of the subcategories until reaching `N` articles."""
        for subcategorie_str in self.subcategories_str:
            self.fetch_pageids(cmtitle=subcategorie_str)
            if len(self.pageids) == self.n:
                break

    def fetch_all_text(self):
        """Fetch the text corresponding to all the `pageids`.

        Fetch the text corresponding to the current `pageids` stored as attribute. It uses
        batches of 20 articles as it is the limit on the wikipedia API.

        """
        for pageids in batch(self.pageids, 20):
            payload = {
                'action': 'query',
                'format': 'json',
                'prop': 'extracts',
                'explaintext': '',
                'exintro': ''
            }
            payload['pageids'] = '|'.join([str(x) for x in pageids])
            data = requests.get(WikiCategory.BASE_URL + "/w/api.php", payload).json()
            for pageid in pageids:
                txt = data['query']['pages'][str(pageid)]['extract']
                # perform some quick data cleaning
                txt = txt.replace('\t', '').replace('\n', '').strip()
                self.texts.append(txt)

if __name__ == "__main__":
    physics = WikiCategory('Category:Physics')
    physics.fetch_all_pageids()
    physics.fetch_all_text()
    print(physics.texts[0:5])
