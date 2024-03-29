import random
import pickle
from .wiki_category import WikiCategory

class WikiDataLoader:
    """Handle data loading and saving of articles from multiple wikipedia category.

    Randomly select `N` wikipedia categories among `CATEGORIES` and load articles text extract for
    each of them using the `WikiCategory` class.

    Note:
        When loading a category it starts by looking at a corresponding pickle. If None is found
        then it will load the data directly from wikipedia API.

    Attributes:
        data (List[Dict]): Store for each categorie its name and all the retrieved texts.
    """
    # Constant
    CATEGORIES = [
        "Category:Physics",
        "Category:Arts",
        "Category:Biology",
        "Category:Electronics",
        "Category:Earth sciences",
        "Category:Diseases and disorders",
        "Category:Chemistry",
        "Category:Astronomy",
        "Category:Sports",
        "Category:Nutrition"
    ]
    
    def __init__(self, N):
        """Initialize attributes and load the Categories.

        Args:
            N (int): Number of categories to load (must be between 1 and 10).
            categories (List[str]): Store the selected categories.
        """
        self.data = []
        self.categories = []
        try:
            # category_indices = random.sample(range(10), N)
            category_indices = range(N)
        except ValueError:
            print('The number of category N must be between 1 and 10')
            
        for category_index in category_indices:
            category_str = WikiDataLoader.CATEGORIES[category_index]
            self.categories.append(category_str)
            self.load(category_str)
             
    def load(self, category_str):
        """Load the corresponding category either from pickle or `WikiCategory`.

        Args:
            category_str (str): Query string representing the category.
        """
        filename = category_str.replace('Category:', '')
        try:
            texts = pickle.load(open('./saved_states/pickles/' + filename + '.pickle', "rb"))
            print(category_str + " retrieved from file!")
        except (OSError, IOError):
            category = WikiCategory(category_str)
            category.fetch_all_pageids()
            category.fetch_all_text()
            category.save_to_file()
            texts = category.texts

        self.data.append({
            'category': self.categories.index(category_str),
            'texts': texts
        })

    def getFullCorpus(self):
        """Return the whole corpus as a list of text."""
        for data in self.data:
            for text in data['texts']:
                yield (data['category'], text)
