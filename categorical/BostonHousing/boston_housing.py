# This dataloader loads the "Boston Housing Prices"
# http://???
#
# Usage:
# from boston_housing import Loader
# dataloader = Loader()
# x_train, y_train, x_test, y_test = dataloader.boston_housing_data()
import os
import re
import glob
from random import shuffle
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from scipy.stats import zscore

# Only for debugging
import matplotlib.pyplot as plt

# typing imports
from typing import Tuple, List, Any, Union, Optional
from numpy import ndarray
from pandas import DataFrame, Series
from scipy.sparse import csr_matrix

DEBUG = False


class Loader:
    def __init__(self):
        self.home_path = os.getcwd()
        self.data_path = os.path.join("data")
        self.dataset_path = os.path.join(
            self.home_path, self.data_path, "BostonHousing"
        )
        self.filename = r""
        # Labels of input columns
        self.input_signal_types = [
            "income",
            "aspect",
            "subscriptions",
            "dist_healthy",
            "save_rate",
            "dist_unhealthy",
            "age",
            "pop_dense",
            "retail_dense",
            "crime",
            "job_11",
            "job_al",
            "job_am",
            "job_ax",
            "job_bf",
            "job_by",
            "job_cv",
            "job_de",
            "job_dz",
            "job_e2",
            "job_f8",
            "job_gj",
            "job_gv",
            "job_kd",
            "job_ke",
            "job_kl",
            "job_kp",
            "job_ks",
            "job_kw",
            "job_mm",
            "job_nb",
            "job_nn",
            "job_ob",
            "job_pe",
            "job_po",
            "job_pq",
            "job_pz",
            "job_qp",
            "job_qw",
            "job_rn",
            "job_sa",
            "job_vv",
            "job_zz",
            "area_a",
            "area_b",
            "area_c",
            "area_d",
        ]
        # Output classes to learn how to classify
        self.label = "product"
        self.classes = ["a", "b", "c", "d", "e", "f", "g"]

    def boston_housing_data(self):
        # Read the data set
        df = pd.read_csv(
            "https://data.heatonresearch.com/data/t81-558/jh-simple-dataset.csv",
            na_values=["NA", "?"],
        )

        # Generate dummies for job
        df = pd.concat([df, pd.get_dummies(df["job"], prefix="job")], axis=1)
        df.drop("job", axis=1, inplace=True)

        # Generate dummies for area
        df = pd.concat([df, pd.get_dummies(df["area"], prefix="area")], axis=1)
        df.drop("area", axis=1, inplace=True)

        # Missing values for income
        med = df["income"].median()
        df["income"] = df["income"].fillna(med)

        # Standardize ranges
        df["income"] = zscore(df["income"])
        df["aspect"] = zscore(df["aspect"])
        df["save_rate"] = zscore(df["save_rate"])
        df["age"] = zscore(df["age"])
        df["subscriptions"] = zscore(df["subscriptions"])

        # Convert to numpy - Classification
        x_columns = df.columns.drop("product").drop("id")
        x = df[x_columns].values
        dummies = pd.get_dummies(df["product"])  # Classification
        products = dummies.columns
        y = dummies.values

        # Split into train/test
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.25, random_state=42
        )

        return x_train, y_train, x_test, y_test
