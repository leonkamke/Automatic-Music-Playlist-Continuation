"""
 For playlist titles,
we deal with 41 characters including 26 alphabets (a-z), 10 numbers
(0-9), and 5 special characters (/<>+-)
"""
import csv
import pickle

import numpy as np

PATH_TRACK_SEQUENCES = '/ds/audio/MPD/spotify_million_playlist_dataset_csv/data/track_sequences.csv'
OUTPUT_PATH = "/netscratch/kamke/dictionaries/"

VOCABULARY = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',  # 26 alphabets
              'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
              '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',  # 10 numbers
              '+', '-', '/', '<', '>', '?', '!']  # 7 special characters


def write_all_titles_in_csv():
    titles_file = open('/ds/audio/MPD/spotify_million_playlist_dataset_csv/data/titles.csv',
                       'w', newline='', encoding='utf8')
    playlists_writer = csv.writer(titles_file)
    with open(PATH_TRACK_SEQUENCES, encoding='utf8') as read_obj:
        csv_reader = csv.reader(read_obj)
        # Iterate over each row in the csv file and create lists of track uri's
        counter = 0
        for row in csv_reader:
            title_str = row[1]
            playlists_writer.writerow(title_str)
            counter += 1
    print("read ", counter, " titles")


def convert_title(title_str):
    output = title_str.lower()
    title_str = title_str.lower()
    for char in title_str:
        if char not in VOCABULARY:
            output = output.replace(char, "")
    return output


def build_one_hot_encoded_dict():
    vocab_size = len(VOCABULARY)
    char_to_vec = {}
    for i in range(vocab_size):
        char = VOCABULARY[i]
        vector = np.zeros((vocab_size,), dtype=int)
        vector[i] = 1
        char_to_vec[char] = vector
    with open(OUTPUT_PATH + 'char_dict.pkl', 'wb') as f:
        pickle.dump(char_to_vec, f)


def get_index(char_in):
    for i, character in enumerate(VOCABULARY):
        if character == char_in:
            return i


def title2index_seq(title):
    title = convert_title(title)
    index_seq = []
    for char in title:
        index_seq.append(get_index(char))
    return index_seq

def get_char_dict():
    with open(OUTPUT_PATH + 'char_dict.pkl', 'rb') as f:
        loaded_dict = pickle.load(f)
        return loaded_dict


if __name__ == "__main__":
    """with open(OUTPUT_PATH + 'char_dict.pkl', 'rb') as f:
        loaded_dict = pickle.load(f)
        print(len(loaded_dict))
        print(loaded_dict["!"])
        print(type(loaded_dict["!"]))
        print(type(loaded_dict))"""
    title = ""

    print(title2index_seq(title))
