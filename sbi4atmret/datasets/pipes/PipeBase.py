

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

    def _build_mask(self):
        return NotImplementedError