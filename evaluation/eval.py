"""
EVALUATION OF THE SPOTIFY CHALLENGE:
  1000 examples for each scenario:
  1)  Title only (no tracks)
  2)  Title and first track
  3)  Title and first 5 tracks
  4)  First 5 tracks only
  5)  Title and first 10 tracks
  6)  First 10 tracks only
  7)  Title and first 25 tracks
  8)  Title and 25 random tracks
  9)  Title and first 100 tracks
  10) Title and 100 random tracks

"""

import load_eval_data as eval_data
import numpy as np
from sklearn.metrics import ndcg_score, dcg_score
import torch


def evaluate_model(model, word2vec, last_n_playlists, src_size):
    model.eval()
    # create evaluation dataset
    evaluation_dataset = eval_data.EvaluationDataset(word2vec, last_n_playlists, src_size)
    # loop over all evaluation playlists
    for i, (src, trg, trg_len) in enumerate(evaluation_dataset):
        print("todo")


# functions for calculating R-Precision, NDCG, and Click
# All metrics will be evaluated at both the track level (exact track match)
# and the artist level (any track by the same artist is a match)


def calc_r_precision(prediction, target):
    rel_tracks = np.intersect1d(prediction, target)
    return len(rel_tracks)/len(target)


if __name__ == "__main__":
    y = torch.FloatTensor([[5, 4, 3, 2, 1]])
    y_pred = torch.FloatTensor([[1, 2, 3, 4, 5]])
    print(ndcg_score(y, y_pred))

