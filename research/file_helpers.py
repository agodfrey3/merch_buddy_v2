import pickle


def save_pkl_file(o, p):
    output = open(p, 'wb')
    pickle.dump(o, output)
    output.close()


def load_pkl_file(p):
    pkl_file = open(p, 'rb')
    obj = pickle.load(pkl_file)
    pkl_file.close()

    return obj