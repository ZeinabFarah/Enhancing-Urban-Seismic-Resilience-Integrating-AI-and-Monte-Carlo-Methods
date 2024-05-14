
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

class AIModelDeveloper:
    def __init__(self, dataframe, feature_columns, target_column):
        self.dataframe = dataframe
        self.feature_columns = feature_columns
        self.target_column = target_column
        self.model = None

    def split_data(self):
        """Split the data into training and testing sets."""
        X = self.dataframe[self.feature_columns]
        y = self.dataframe[self.target_column]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    def train_model(self):
        """Train a RandomForestRegressor model."""
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(self.X_train, self.y_train)

    def evaluate_model(self):
        """Evaluate the model using mean squared error."""
        predictions = self.model.predict(self.X_test)
        mse = mean_squared_error(self.y_test, predictions)
        return mse

    def predict(self, new_data):
        """Make predictions on new data."""
        return self.model.predict(new_data)

    def get_feature_importances(self):
        """Retrieve and return feature importances."""
        if not self.model:
            raise ValueError("Model has not been trained yet.")
        return self.model.feature_importances_
