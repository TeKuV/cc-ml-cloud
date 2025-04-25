import pandas as pd

class DataManager:
    def __init__(self, user_file, default_path):
        self.df = self.load_data(user_file, default_path)

    @staticmethod
    def load_data(user_file, default_path):
        if user_file is not None:
            return pd.read_csv(user_file)
        else:
            return pd.read_csv(default_path)

    def detect_task_type(self, target_col="Risk"):
        risk_unique = self.df[target_col].dropna().unique()
        if len(risk_unique) <= 10 and all([str(x).isdigit() for x in risk_unique]):
            return 'classification'
        else:
            return 'regression' 