import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, mean_squared_error, r2_score

class RiskModel:
    def __init__(self, task_type):
        self.task_type = task_type
        self.model = None
        self.features = None

    def train(self, df, features, test_size, random_state):
        X = df[features].fillna(df[features].mean())
        y = df["Risk"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=random_state)
        self.features = features
        if self.task_type == 'classification':
            self.model = RandomForestClassifier(n_estimators=100, random_state=random_state)
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)
            report = classification_report(y_test, y_pred)
            return acc, cm, report, None, None
        else:
            self.model = RandomForestRegressor(n_estimators=100, random_state=random_state)
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            return None, None, None, mse, r2

    def predict(self, input_data):
        input_df = pd.DataFrame([input_data])
        return self.model.predict(input_df)[0] 