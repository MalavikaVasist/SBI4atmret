class estimator(nn.Module):
    def __init__(self, flow, embedding):
        super().__init__()
        self.flow = flow
        self.embedding = embedding

    def forward(self, x_dict, theta):
        x_emb = self.embedding(x_dict)
        return self.flow(theta, x_emb)

    def flow_forward(self, x_dict):
        x_emb = self.embedding(x_dict)
        return self.flow.flow(x_emb)