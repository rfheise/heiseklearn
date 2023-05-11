import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import os


class DataSet:

    def __init__(self, name, file_name, target, ignore=[]):
        self.test_size = .25
        self.target = target
        self.types = None
        self.file_name = self.get_file_name(file_name)
        self.name = name
        self.ignore = ignore
        self.rand = 42

    def get_file_name(self, file_name):
        path = os.path.dirname(__file__)
        path = os.path.join(path, "raw", file_name)
        return path
    
    def load_dataset(self, file_name):
        data = pd.read_csv(file_name)
        y = data[self.target]
        data.drop(self.target, axis=1, inplace=True)
        return data, y
    
    def save_dataset(self, x, y, file_name):
        x[self.target] = y 
        x.to_csv(file_name,index=False)
        x.drop(self.target, axis=1, inplace=True)

    def load(self, cache=True):
        prev = os.path.normpath(os.path.dirname(__file__) + os.sep + os.pardir)
        datasets = os.path.join(prev, "datasets",".data_cache")
        train_set = os.path.join(datasets, f"{self.name}-train-cache.csv")
        test_set = os.path.join(datasets, f"{self.name}-test-cache.csv")
        if os.path.exists(train_set) and os.path.exists(test_set):
            self.train_x, self.train_y = self.load_dataset(train_set)
            self.test_x, self.test_y = self.load_dataset(test_set)
            return
        self.initialize_data(self.file_name)
        self.clean_data()
        if cache:
            if not os.path.exists(datasets):
                os.makedirs(datasets)
            self.save_dataset(self.train_x, self.train_y, train_set)
            self.save_dataset(self.test_x, self.test_y, test_set)


    def initialize_data(self, file_name):
        data = pd.read_csv(file_name)
        for ig in self.ignore:
            data.drop(ig, axis=1, inplace=True)
        if not self.types:
            self.identify_types(data)
        if self.target not in data.columns:
            print("Invalid option passed to target")
            print(f"Options are: {data.columns}")
        data_y = data[self.target]
        data.drop(self.target, axis=1, inplace=True)
        data_x = data
        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(
            data_x, data_y, test_size=self.test_size, random_state=self.rand
        )


    def clean_data(self):
        self.train_x, self.train_y = self.clean_dataset(self.train_x, self.train_y, True)
        self.test_x, self.test_y = self.clean_dataset(self.test_x, self.test_y)

    def balance(self, data):
        le = LabelEncoder()
        data[self.target] = le.fit_transform(data[self.target])
        pos = data.loc[data[self.target] == 1]
        neg = data.loc[data[self.target] == 0]
        minimum = min(len(pos), len(neg))
        frames = [pos.head(minimum), neg.head(minimum)]
        return pd.concat(frames).sample(frac=1, random_state=self.rand)
    
    def clean_dataset(self, x, y, train=False):
        
        init_params = (train or not self.attributes)
        if init_params:
            self.attributes = {}
        x[self.target] = y
        if self.types[self.target] == "binary" and init_params:
            x = self.balance(x)
        x = self.clean_strings(x)
        x = self.clean_real(x, init_params)
        x = self.clean_binary(x, init_params)
        x = self.clean_categorical(x, init_params)
        x = self.clean_columns(x)

        y = x[self.target]
        x.drop(self.target, axis=1, inplace=True)
        return x, y
    def clean_strings(self,data):
        # clean up column names and strings
        for col in data.columns:
            if data[col].dtype == "object":
                data[col] = data[col].str.lower()
        return data
    
    def clean_columns(self, data):
        for col in data.columns:
            name=col.replace("<", "_less_than_")
            name=name.replace(">","_greater_than_")
            name=name.replace(" ","_space_")
            name=name.replace(",","_comma_")
            if name != col:
                data.rename(columns={col:name}, inplace=True)
        return data
    
    def clean_real(self, data, init_params=False):
        for column in data.columns:
            if self.types[column] == "real":
                if init_params:
                    self.attributes[column] = {}
                    self.attributes[column]['mean'] = data[column].mean()
                    self.attributes[column]['std'] = data[column].std()
                #normalize data
                data[column] = data[column].fillna(self.attributes[column]['mean'])
                data[column] = (data[column] - self.attributes[column]["mean"])/(self.attributes[column]['std'] + 1e-10)
        return data
    
    def clean_binary(self, data, init_params=False):
        for column in data.columns:
            if self.types[column] == "binary":
                if data[column].dtype == "object":
                    le = LabelEncoder()
                    data[column] = le.fit_transform(data[column])
        return data
    
    def clean_categorical(self, data, init_params=False):
        for column in data.columns:
            if self.types[column] == "string":
                vals = data[column].value_counts()
                vals = vals.nsmallest(len(vals) - 100)
                vals = set(dict(vals).keys())
                data[column] = data[column].apply(lambda x: x if x not in vals else "other")
                if init_params:
                    self.attributes[column] = OneHotEncoder(handle_unknown='infrequent_if_exist', sparse=False)
                    self.attributes[column].fit(data[column].unique().reshape(-1,1))
                one_hot_arr = self.attributes[column].transform(data[column].to_numpy().reshape(-1,1))
                names = [column +"_"+ name for name in self.attributes[column].get_feature_names_out()]
                one_hot_df = pd.DataFrame(one_hot_arr, columns=names, index=data.index)
                data = pd.concat([data, one_hot_df], axis=1, join="outer").drop([column], axis=1)
        return data


    def identify_types(self, data):
        self.types = {}
        for column in data.columns:
            self.types[column] = self.identify_type(data[column])
    
    def identify_type(self, column):
        
        if column.nunique() == 2:
            return "binary"

        if column.dtype == "object":
            return "string"
        
        return "real"
    

