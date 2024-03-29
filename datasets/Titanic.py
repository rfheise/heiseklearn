from .DataSet import DataSet


class Titanic(DataSet):

    def __init__(self):
        # initializes banking data with parameters
        super().__init__("titanic","titanic.csv","Survived",['PassengerId'])


if __name__ == "__main__":
    # processes data and creates cache
    Titanic().load()