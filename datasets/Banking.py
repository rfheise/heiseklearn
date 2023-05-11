from .DataSet import DataSet


class Banking(DataSet):

    def __init__(self):
        # initializes banking data with parameters
        super().__init__("banking","banking_data.csv","loan_paid",['ID'])


if __name__ == "__main__":
    # processes data and creates cache
    Banking().load()