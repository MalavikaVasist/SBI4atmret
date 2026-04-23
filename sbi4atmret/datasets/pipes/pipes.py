






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

        # example: build masks from config if needed
        self.mask = ...

    def forward(self, *batches):
        # unpack
        (thetac, xc), (thetagc, xgc), (thetahc, xhc) = batches

        # apply your slicing / masking logic
        xc = xc[:, 1:1299]
        xgc = xgc[:, self.mask]

        # combine however you want
        theta = thetac  # or concatenate
        x = torch.cat([xc, xgc, xhc], dim=-1)

        return theta, x
    

