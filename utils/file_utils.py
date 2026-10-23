import pickle

def save_pkl(filename, save_object):
    with open(filename, 'wb') as writer:
        pickle.dump(save_object, writer)
