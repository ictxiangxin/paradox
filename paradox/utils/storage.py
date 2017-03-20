import pickle


def save(file: str, thing):
    with open(file, 'wb') as save_file:
        pickle.dump(thing, save_file)


def load(file: str):
    with open(file, 'rb') as load_file:
        return pickle.load(load_file)
