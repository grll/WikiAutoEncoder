import pickle
import numpy as np

from wiki_dataloader.wiki_dataloader import WikiDataLoader
from word_embedding.embedding import Embedding
from classification.classify import Classify
from classification.visualize import Visualize

# Load 1k articles for 5 wikipedia categories.
wiki_data_loader = WikiDataLoader(5)

# Tokenize, Compute co-occurance, Compact and Convert these articles into a matrices.
embedding = Embedding(wiki_data_loader.getFullCorpus)
embedding.create_prob_array()

# Generate the autoencoded version of the articles.
# embedding.init_auto_encoder(skip_training=True)
# articles_encoded = embedding.create_encoded_article_array()
articles_encoded = pickle.load(
    open('./saved_states/pickles/articles_encoded.pickle', 'rb'))

# Generate a Dataset for classification.
X = np.array([article_encoded['data']
              for article_encoded in articles_encoded])
y = np.array([article_encoded['category']
              for article_encoded in articles_encoded])

# Initialize classifier.
classifier = Classify(X, y)
# Perform Logistic Regression.
y_pred_lr, results_lr = classifier.logistic_regression()
print("=== Logistic Regression ===")
print("Best Parameters from gridSearch:", results_lr['best_params_'])
print("Accuracy obtained on test set: {0:.2f}%".format(results_lr['accuracy'] * 100))
# Perform Random Forest.
y_pred_rf, results_rf = classifier.random_forest()
print("=== Random Forest ===")
print("Best Parameters from gridSearch:", results_rf['best_params_'])
print("Accuracy obtained on test set: {0:.2f}%".format(results_rf['accuracy'] * 100))

# Visualize the results on a confusion Matrix.
viz = Visualize()
viz.plot_confusion_matrix(classifier.y_test, y_pred_lr,
                          wiki_data_loader.categories, title="Logistic Regression RAW")
viz.plot_confusion_matrix(classifier.y_test, y_pred_rf,
                          wiki_data_loader.categories, title="Random Forest RAW")