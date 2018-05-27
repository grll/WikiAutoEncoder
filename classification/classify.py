import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

class Classify:
    """Implement and manage several classifiers with sklearn pipeline.
    
    The class get initialized with the dataset and takes care of splitting it. Then it is ready to
    perform one of the classifying pipeline such as LogisticRegression or RandomForest. The
    pieplines run a standard scaling on the train set and a gridsearch cross validation to find some
    optimized hyperparameters.

    Attributes:
        X_train (2darray): 2D array representing the training data (each row one sample).
        X_test (2darray): 2D array representing the testing data (each row one sample).
        y_train (1darray): 1D array representing the training groundtruth values.
        y_test (1darray): 1D array representing the testing groundtruth values.
    """
    def __init__(self, X, y, test_size=0.2, random_state=None):
        """Initialize the attributes by generating testing and training sets.

        Args:
            X (2darray): 2D array representing the whole dataset (each row one sample).
            y (1darray): 1D array representing every output values from the dataset.
            test_size (float): Number representing the percentage of data in the testing set.
            random_state (int): Random seed for the shuffling of the data.
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=random_state)

    def logistic_regression(self, c=np.arange(1., 5., 0.5), cv=3):
        """Create and run a logistic regression pipeline on the data.
        
        Args:
            c (List[float]): List of values to try in the gridsearch for c (penalyt parameter).
            cv (int): Number of folds to use for the gridsearch cross validation.
        """
        pipeline = make_pipeline(StandardScaler(), 
                                 GridSearchCV(LogisticRegression(),
                                 param_grid={'C': c},
                                 cv=cv,
                                 refit=True))
        pipeline.fit(self.X_train, self.y_train)
        y_pred_lr = pipeline.predict(self.X_test)

        return (y_pred_lr, { 'best_params_': pipeline.named_steps.gridsearchcv.best_params_,
                 'accuracy': accuracy_score(self.y_test, y_pred_lr) })

    def random_forest(self, n_estimators=[20, 30, 40], max_depth=np.arange(5, 15), cv=3):
        """Create and run a Random Forest pipeline on the data.
        
        Args:
            n_estimators (List[int]): Values to try in the gridsearch (number of trees).
            max_depth (List[int]): Values to try in the gridsearch (max depth of the trees).
            cv (int): Number of folds to use for the gridsearch cross validation.
        """
        pipeline = make_pipeline(StandardScaler(), 
                                 GridSearchCV(RandomForestClassifier(),
                                 param_grid={
                                     'n_estimators': [20, 30, 40],
                                     'max_depth': np.arange(5, 15),
                                 },
                                 cv=3,
                                 refit=True))
        pipeline.fit(self.X_train, self.y_train)
        y_pred = pipeline.predict(self.X_test)

        return (y_pred, { 'best_params_': pipeline.named_steps.gridsearchcv.best_params_,
                 'accuracy': accuracy_score(self.y_test, y_pred) })