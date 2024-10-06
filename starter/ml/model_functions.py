from sklearn.metrics import fbeta_score, precision_score, recall_score
import logging
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import joblib

# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    try:
        model = RandomForestClassifier()
        smote = SMOTE(random_state=0)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        model.fit(X_train, y_train)
        logging.info('SUCCESS!:Model trained and saved')
        return model
    except BaseException:
        logging.info('ERROR!:Model not trained and not saved')


def model_predictions(X_test, model):
    try:
        predictions = model.predict(X_test)
        logging.info('SUCCESS!:Model predictions generated')
        return predictions
    except BaseException:
        logging.info('ERROR!:Model predictions not generated')


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model
    using precision, recall, and F1.
    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    try:
        fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
        precision = precision_score(y, preds, zero_division=1)
        recall = recall_score(y, preds, zero_division=1)
        logging.info('SUCCESS Get metrics models')
        return precision, recall, fbeta
    except BaseException:
        logging.info('ERROR: Error occurred when cal metrics models')


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    preds = model.predict(X)
    return preds


def save_model(model, path):
    """ Save the trained model to a file.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    path : str
        Path to save the model.
    """
    joblib.dump(model, path)


def load_model(path):
    """ Load a trained model from a file.

    Inputs
    ------
    path : str
        Path to the model.
    Returns
    -------
    model
        Trained machine learning model.
    """
    logging.info(f"Loading model from {path + '/model.joblib'}")
    model = joblib.load(path + "/model.joblib")
    encoder = joblib.load(path + "/encoder.joblib")
    lb = joblib.load(path + "/lb.joblib")
    return model, encoder, lb