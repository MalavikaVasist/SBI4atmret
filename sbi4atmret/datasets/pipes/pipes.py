






class BasePipe:
    def __init__(self, config):
        self.config = config

    def __call__(self, *batches):
        """
        batches = [(theta1, x1), (theta2, x2), ...]
        """
        return self.forward(*batches)

    def forward(self, *batches):
        raise NotImplementedError


class MiriGeminiHSTPipe(BasePipe):
    def __init__(self, config):
        super().__init__(config)

        self.mask = ...  # from config if needed


    def forward(self, batches: dict):
        thetac, xc = batches["miri"]
        thetagc, xgc = batches["gemini"]
        thetahc, xhc = batches["hst"]

        xc = xc[:, 1:1299]
        xgc = xgc[:, self.mask]

        theta = thetac
        x = torch.cat([xc, xgc, xhc], dim=-1)

        return theta, x




    

