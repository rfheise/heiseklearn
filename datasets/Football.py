from .DataSet import DataSet


class Football(DataSet):

    def __init__(self):
        # initializes banking data with parameters
        super().__init__("football","heisepow.csv","won",['wins','losses','ties'])


if __name__ == "__main__":
    # processes data and creates cache
    Football().load()