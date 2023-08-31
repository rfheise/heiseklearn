import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import os
from scipy import stats

# generic dataset class
class DataSet:

    def __init__(self, name, file_name, target, ignore=[]):
        #testing data percentage
        self.test_size = .25

        #target data column name
        self.target = target

        #types of each column
        self.types = None

        #file name for dataset
        self.file_name = self.get_file_name(file_name)

        #name of the datset
        self.name = name

        #columns to ignore 
        self.ignore = ignore

        #random seed
        self.rand = 42

    def get_file_name(self, file_name):
        # computes absolute path of dataset
        path = os.path.dirname(__file__)
        path = os.path.join(path, "raw", file_name)
        return path
    
    def load_dataset(self, file_name):
        # loads cached dataset
        data = pd.read_csv(file_name)
        y = data[self.target]
        data.drop(self.target, axis=1, inplace=True)
        return data, y
    
    def save_dataset(self, x, y, file_name):
        # caches dataset
        x[self.target] = y 
        x.to_csv(file_name,index=False)
        x.drop(self.target, axis=1, inplace=True)

    def load(self, cache=True):
        #loads dataset from disk if cached 
        # otherwise it processes the data
        # takes a while to caching is suggested

        # computes the file path of cached 
        # training and test files
        prev = os.path.normpath(os.path.dirname(__file__) + os.sep + os.pardir)
        datasets = os.path.join(prev, "datasets",".data_cache")
        train_set = os.path.join(datasets, f"{self.name}-train-cache.csv")
        test_set = os.path.join(datasets, f"{self.name}-test-cache.csv")

        # loads processed dataset from disk if cached
        if os.path.exists(train_set) and os.path.exists(test_set):
            self.train_x, self.train_y = self.load_dataset(train_set)
            self.test_x, self.test_y = self.load_dataset(test_set)

            self.test_y = np.reshape(self.test_y.to_numpy(), (self.test_y.shape[0],1))
            self.train_y = np.reshape(self.train_y.to_numpy(), (self.train_y.shape[0],1))
            return
        
        #initializes the raw data & cleans it
        self.initialize_data(self.file_name)
        self.clean_data()

        #caches dataset if caching is set to True
        if cache:
            if not os.path.exists(datasets):
                os.makedirs(datasets)
            self.save_dataset(self.train_x, self.train_y, train_set)
            self.save_dataset(self.test_x, self.test_y, test_set)
        self.test_y = np.reshape(self.test_y.to_numpy(), (self.test_y.shape[0],1))
        self.train_y = np.reshape(self.train_y.to_numpy(), (self.train_y.shape[0],1))

    def initialize_data(self, file_name):
        # collects raw data from the passed in file 

        data = pd.read_csv(file_name)

        # ignores columns if any are specified
        for ig in self.ignore:
            data.drop(ig, axis=1, inplace=True)

        #automatically specifies types if not specified by default
        if not self.types:
            self.identify_types(data)
        
        # checks to make sure target vector is in dataset
        if self.target not in data.columns:
            print("Invalid option passed to target")
            print(f"Options are: {data.columns}")
        
        # splits data into training and test
        data_y = data[self.target]
        data.drop(self.target, axis=1, inplace=True)
        data_x = data
        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(
            data_x, data_y, test_size=self.test_size, random_state=self.rand
        )


    def clean_data(self):
        #cleans training data
        self.train_x, self.train_y = self.clean_dataset(self.train_x, self.train_y, True)

        #cleans testing data
        self.test_x, self.test_y = self.clean_dataset(self.test_x, self.test_y)

    def balance(self, data):
        # balances binary data
        #makes sure target is encoded already
        le = LabelEncoder()
        data[self.target] = le.fit_transform(data[self.target])

        #makes sure pos and neg have equal samples
        pos = data.loc[data[self.target] == 1]
        neg = data.loc[data[self.target] == 0]
        minimum = min(len(pos), len(neg))

        #combine pos and negative 
        frames = [pos.head(minimum), neg.head(minimum)]
        # reshuffle the data using the random seed
        return pd.concat(frames).sample(frac=1, random_state=self.rand)
    
    def clean_dataset(self, x, y, train=False):
        # cleans the given dataset 

        #will initialize parameters if it is the training set
        #i.e. use training mean and stdev for normalizing all data
        init_params = (train or not self.attributes)
        if init_params:
            self.attributes = {}
        
        # combines into one large df
        x[self.target] = y
        # if binary target balance the feature sets
        # does not mess with the balance of the testing data
        if self.types[self.target] == "binary" and init_params:
            x = self.balance(x)
    
        # performs various cleaning activities
        x = self.clean_strings(x)
        x = self.clean_real(x, init_params)
        x = self.clean_binary(x, init_params)
        x = self.clean_categorical(x, init_params)
        x = self.clean_columns(x)
        
        #go back to original feature set and target vector
        y = x[self.target]

        # sets all values between 0,1
        # x = (x - x.min())/(x.max() - x.min() + 1e-10)

        x.drop(self.target, axis=1, inplace=True)
        return x, y
    def clean_strings(self,data):
        # cleans up strings 

        for col in data.columns:
            if data[col].dtype == "object":
                # makes sure all data points in category have same case
                data[col] = data[col].astype(str).str.lower()
        return data
    
    def clean_columns(self, data):
        # makes sure feature names are formatted correctly 

        for col in data.columns:
            # do not allow the following characters in feature name
            # replace with symbol spelled out
            name=col.replace("<", "_less_than_")
            name=name.replace(">","_greater_than_")
            name=name.replace(" ","_space_")
            name=name.replace(",","_comma_")
            if name != col:
                data.rename(columns={col:name}, inplace=True)
        return data
    
    def clean_real(self, data, init_params=False):
        #normalizes real data 

        for column in data.columns:
            if self.types[column] == "real":
                if self.target == column:
                    continue
                # initalize mean and stdev from training set
                if init_params:
                    self.attributes[column] = {}
                    self.attributes[column]['mean'] = data[column].mean()
                    self.attributes[column]['std'] = data[column].std()
                #normalize data
                data[column] = data[column].fillna(self.attributes[column]['mean'])
                data[column] = (data[column] - self.attributes[column]["mean"])/(self.attributes[column]['std'] + 1e-10)
        return data
    
    def clean_binary(self, data, init_params=False):
        # label encodes binary data
        for column in data.columns:
            if self.types[column] == "binary":
                if data[column].dtype == "object":
                    # only if in string format does anything need to be done
                    le = LabelEncoder()
                    data[column] = le.fit_transform(data[column])
                if init_params:
                    # sets nan to mode 
                    # should look into later
                    self.attributes[column] =  np.around(np.sum(data[column]))
                data[column] = data[column].fillna(self.attributes[column])
                # cheap way to convert boolean column to int
                data[column] = data[column] + 0

        return data
    
    def clean_categorical(self, data, init_params=False):
        # one hot encodes categorical data

        for column in data.columns:
            if self.types[column] == "string":
                # if over 100 categories put rest into other
                vals = data[column].value_counts()
                vals = vals.nsmallest(len(vals) - 100)
                vals = set(dict(vals).keys())
                data[column] = data[column].apply(lambda x: x if x not in vals else "other")
                
                # initializes one hot on training data
                if init_params:
                    self.attributes[column] = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
                    self.attributes[column].fit(data[column].unique().reshape(-1,1))
               
                # creates numpy 2D matrix of one hot vectors
                one_hot_arr = self.attributes[column].transform(data[column].to_numpy().reshape(-1,1))

                # modifies names of one hot features to not have dups
                names = [column +"_"+ name for name in self.attributes[column].get_feature_names_out()]

                # creates dataframe from one numpy arr
                one_hot_df = pd.DataFrame(one_hot_arr, columns=names, index=data.index)

                #concats with original data and drops old categorical column
                data = pd.concat([data, one_hot_df], axis=1, join="outer").drop([column], axis=1)

        return data


    def identify_types(self, data):
        # identifies the various type for each column
        # tries to generalize data cleaning based upon type

        self.types = {}
        for column in data.columns:
            self.types[column] = self.identify_type(data[column])
    
    def identify_type(self, column):
        # identifies type of column's data 

        # will need to be modified in the future 
        # for more complex datasets 

        # if only two options it is binary
        if column.nunique() == 2:
            return "binary"

        # if type is object than it's probably and string
        if column.dtype == "object":
            return "string"
        
        # otherwise it is a real value
        return "real"
    

