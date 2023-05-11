from .DataSet import DataSet


class Banking(DataSet):

    def __init__(self):
        super().__init__("banking","banking_data.csv","loan_paid",['ID'])


if __name__ == "__main__":
    Banking().load()