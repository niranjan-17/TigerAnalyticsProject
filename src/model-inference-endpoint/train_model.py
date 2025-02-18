import pickle
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import sys
import os


# Get the absolute path of the 'src' directory dynamically
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "MLE_TakeHomeAssignment-DEV", "src"))

# Add it to the Python path
sys.path.append(project_root)

# Import DataLoader correctly
from text_loader.loader import DataLoader


class TweetClassifier:
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader
        self.model = None

    def train_and_evaluate(self):
        """Trains an XGBoost model and evaluates it."""
        X = self.data_loader.preprocess_tweets()
        y = self.data_loader.preprocess_labels()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        param_grid = {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }

        xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, scoring='accuracy', cv=3, verbose=1,
                                   n_jobs=-1)
        grid_search.fit(X_train, y_train)

        self.model = grid_search.best_estimator_
        y_pred = self.model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.4f}")
        print(classification_report(y_test, y_pred, target_names=self.data_loader.encoder.classes_))
        print("Best parameters found:", grid_search.best_params_)

        # Save the model, vectorizer, and encoder
        with open("xgb_model.pkl", "wb") as model_file:
            pickle.dump(self.model, model_file)
        with open("vectorizer.pkl", "wb") as vectorizer_file:
            pickle.dump(self.data_loader.vectorizer, vectorizer_file)
        with open("encoder.pkl", "wb") as encoder_file:
            pickle.dump(self.data_loader.encoder, encoder_file)


if __name__ == "__main__":
    data_loader = DataLoader()
    classifier = TweetClassifier(data_loader)
    classifier.train_and_evaluate()
