class Ensemble:
    def __init__(self, model_list):
        self.model_list = model_list
        # number of unique tracks in the MPD dataset = 2262292
        self.vocab_size = 2262292

    def predict(self, input, num_predictions):
        """# x.shape == (vocab_size)
        _, top_k = torch.topk(x, dim=0, k=num_predictions)
        # top_k.shape == (num_predictions)"""
        print("todo")

        return top_k

    def predict(self, ):
        print("d")

    def score(self, ):
        print("d")


if __name__ == "__main__":
    print("dk")