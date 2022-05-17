import torch


class Ensemble:
    def __init__(self, model_list):
        self.model_list = model_list
        # number of unique tracks in the MPD dataset = 2262292
        self.vocab_size = 2262292

    def predict(self, input, num_predictions):
        """# x.shape == (vocab_size)
        _, top_k = torch.topk(x, dim=0, k=num_predictions)
        # top_k.shape == (num_predictions)"""
        rankings = torch.zeros(self.vocab_size, dtype=torch.float)
        for model in self.model_list:
            prediction = model.predict(input, num_predictions)
            for i, track_id in enumerate(prediction):
                rankings[track_id] += (num_predictions - i)
        _, top_k = torch.topk(rankings, dim=0, k=num_predictions)
        return top_k


if __name__ == "__main__":
    model_list = []

