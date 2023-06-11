from .DataSet import DataSet


class DataScience(DataSet):

    def __init__(self):
        # initializes banking data with parameters
        super().__init__("data_science","ds_salaries.csv","salary",[])


if __name__ == "__main__":
    # processes data and creates cache
    DataScience().load()