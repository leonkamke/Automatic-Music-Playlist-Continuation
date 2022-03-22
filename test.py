import pandas as pd
import numpy as np

# skript for creating the vocabulary (list of all tracks)
if __name__ == '__main__':
    list = np.array(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i'])
    list2 = np.array(['j', 'k', 'l'])
    list3 = np.array(['m', 'n'])
    list4 = np.array(['o'])
    a = np.array([list, list2], dtype=object)
    b = np.array([list3, list4], dtype=object)
    train_data = np.array([])
    train_data = np.insert(train_data, a)
    print(train_data)

