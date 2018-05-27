# WikiAutoEncoder
<hr>
### An auto-encoder to help classify article by category on Wikipedia. (Swisscom task)

### The Task:

1. Take N classes of wikipedia articles (laziest way possible)
    * for each class - 1000 articles 
2. Create an autoencoder to compress the articles.
3. Perform classification with typical classifier.
4. Discussion:
    * Compare to a classification on plain text.
    * Compare to PCA

### What have I used for this task ?

* Crawling Wikipedia:
    * mediawiki API.
    * `requests` python package.
* Word / Article Representation:
    * stopwords removal from `nltk`.
    * co-occurance probability with a default context size of 6 for each token.
    * article representation through simple sum of co-occurance probability for each token.
* Compression:
    * AutoEncoder:
        * a simple 4 layers NN (implemented with `tensorflow`).
    * PCA:
        * `sklearn` implementation of PCA.
* Classification:
    * LogisticRegression with `sklearn pipeline`.
    * RandomForest with `sklearn pipeline`.
* Visualization:
    * Visualization using `matplotlib`.
