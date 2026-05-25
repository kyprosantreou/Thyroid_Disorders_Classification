import os
import pandas as pd

def load_data():
    # Import file with data
    data = os.path.expanduser(os.path.join("~", "Documents", "Python", "Datasets", "Thyroid Datasets"))
    df = pd.read_csv(os.path.join(data, "graves-negative.csv"))

    df= pd.get_dummies(df, columns=['sex'], drop_first=False)

    return df
