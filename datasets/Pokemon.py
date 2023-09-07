from .DataSet import DataSet


class Pokemon(DataSet):

    def __init__(self):
        # initializes banking data with parameters
        super().__init__("pokemon","pokemon.csv","Legendary",['#'])


if __name__ == "__main__":
    # processes data and creates cache
    Pokemon().load()