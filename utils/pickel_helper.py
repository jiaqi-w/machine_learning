import pickle
import os
import pathlib

'''
Updated on Oct 15, 2017
@author: Created by Slugbot Team, updated by Jiaqi
'''

# PICKLES
class Pickle_Helper():

    @staticmethod
    def save_model_to_pickle(model, pickle_fname):
        if not os.path.exists(pickle_fname):
            dir_name = os.path.dirname(pickle_fname)
            os.makedirs(dir_name, exist_ok=True)
        with open(pickle_fname, 'wb') as pfile:
            pickle.dump(model, pfile)

    @staticmethod
    def load_model_from_pickle(pickle_fname):
        if os.path.exists(pickle_fname):
            with open(pickle_fname, 'rb') as pfile:
                model = pickle.load(pfile)
                return model
        return None
