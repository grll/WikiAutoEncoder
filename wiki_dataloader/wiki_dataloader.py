import random
import pickle
from .wiki_category import WikiCategory

class WikiDataLoader:
    """Handle data loading and saving of articles from multiple wikipedia category.

    Randomly select `N` wikipedia categories among `CATEGORIES` and load articles text
    extract for each of them using the `WikiCategory` class.

    Note:
        When loading a category it starts by looking at a corresponding pickle. If None
        is found then it will load the data directly from wikipedia API.

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
        """Initialize attributes and load the Categories

        Args:
            N (int): Number of categories to load (must be between 1 and 10)

        """
        self.data = []

        try:
            category_indices = random.sample(range(10), N)
        except ValueError:
            print('The number of category N must be between 1 and 10')
            
        for category_index in category_indices:
            self.load(WikiDataLoader.CATEGORIES[category_index])
             
    def load(self, category_str):
        """Load the corresponding category either from pickle or `WikiCategory`

        Args:
            category_str (str): Query string representing the category.
        """
        filename = category_str.replace('Category:', '')
        try:
            texts = pickle.load(open('./pickles/' + filename + '.pickle', "rb"))
            print(category_str + " retrieved from file!")
        except (OSError, IOError):
            category = WikiCategory(category_str)
            category.fetch_all_pageids()
            category.fetch_all_text()
            category.save_to_file()
            texts = category.texts

        self.data.append({
            'category_str': category_str,
            'texts': texts
        })
