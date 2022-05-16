START_PATH = 1  # line in which the paths start
START_TRAINING = 9  # line in which the training parameters start

"""
Which line contains which path
"""
TRACK_SEQUENCES = 1
ARTIST_SEQUENCES = 2
TRACK_ARTIST_DICT_UNIQUE = 3
TRACK_TO_VEC_MODEL = 4
ARTIST_TO_VEC_MODEL = 5
OUTPUT_PATH_MODEL = 6
EMBED_WEIGHTS = 26

"""
Which line contains which training parameter
"""
N_EPOCHS = 9
BATCH_SIZE = 10
LEARNING_RATE = 11
NUM_PLAYLISTS_FOR_TRAINING = 12

"""
Which line contains which evaluation parameter
"""
START_IDX = 15
END_IDX = 16

"""
Which line contains which device will be used
"""
DEVICE = 19

"""
Which line contains which device will be used
"""
NUM_RECURRENT_LAYERS = 22
NUM_RECURRENT_DIMENSION  = 23


# Methods for getting the network architecture
def get_num_recurrent_layers():
    file = open("attributes", "r")
    for i, row in enumerate(file):
        if i == NUM_RECURRENT_LAYERS:
            file.close()
            return int(row[:-1])


def get_recurrent_dimension():
    file = open("attributes", "r")
    for i, row in enumerate(file):
        if i == NUM_RECURRENT_DIMENSION:
            file.close()
            return int(row[:-1])


# Method for getting the setted device
def get_device():
    file = open("attributes", "r")
    for i, row in enumerate(file):
        if i == DEVICE:
            file.close()
            return row[:-1]


# Methods for the evaluation parameters -----------------------------------------------
def get_start_idx():
    file = open("attributes", "r")
    for i, row in enumerate(file):
        if i == START_IDX:
            file.close()
            return int(row[:-1])


def get_end_idx():
    file = open("attributes", "r")
    for i, row in enumerate(file):
        if i == END_IDX:
            file.close()
            return int(row[:-1])


# Methods for the training parameters -------------------------------------------------
def get_learning_rate():
    file = open("attributes", "r")
    for i, row in enumerate(file):
        if i == LEARNING_RATE:
            file.close()
            return float(row[:-1])


def get_num_playlists_training():
    file = open("attributes", "r")
    for i, row in enumerate(file):
        if i == NUM_PLAYLISTS_FOR_TRAINING:
            file.close()
            return int(row[:-1])


def get_num_epochs():
    file = open("attributes", "r")
    for i, row in enumerate(file):
        if i == N_EPOCHS:
            file.close()
            return int(row[:-1])


def get_batch_size():
    file = open("attributes", "r")
    for i, row in enumerate(file):
        if i == BATCH_SIZE:
            file.close()
            return int(row[:-1])


# Methods for getting the pathes -------------------------------------------------------
def path_track_sequences_path():
    file = open("attributes", "r")
    for i, row in enumerate(file):
        if i == TRACK_SEQUENCES:
            file.close()
            return row[:-1]


def path_artist_sequences_path():
    file = open("attributes", "r")
    for i, row in enumerate(file):
        if i == ARTIST_SEQUENCES:
            file.close()
            return row[:-1]


def path_track_artist_dict_unique():
    file = open("attributes", "r")
    for i, row in enumerate(file):
        if i == TRACK_ARTIST_DICT_UNIQUE:
            file.close()
            return row[:-1]


def path_track_to_vec_model():
    file = open("attributes", "r")
    for i, row in enumerate(file):
        if i == TRACK_TO_VEC_MODEL:
            file.close()
            return row[:-1]


def path_artist_to_vec_model():
    file = open("attributes", "r")
    for i, row in enumerate(file):
        if i == ARTIST_TO_VEC_MODEL:
            file.close()
            return row[:-1]


def output_path_model():
    file = open("attributes", "r")
    for i, row in enumerate(file):
        if i == OUTPUT_PATH_MODEL:
            file.close()
            return row[:-1]


def path_embedded_weights():
    file = open("attributes", "r")
    for i, row in enumerate(file):
        if i == EMBED_WEIGHTS:
            file.close()
            return row[:-1]
