



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

    def get_inst(self, batches, condition, instrument):
        try:
            return batches[condition][instrument]
        except KeyError:
            raise KeyError(f"Missing {condition}/{instrument} in batches")


class MiriGeminiHSTPipe(BasePipe):
    def __init__(self, config):
        super().__init__(config)

        self.mask = ...  # from config if needed

    def forward(self, batches: dict):
        thetac, xc = self.get_inst(batches, "cloudfree", "miri")
        thetagc, xgc = self.get_inst(batches, "cloudfree", "gemini")
        thetahc, xhc = self.get_inst(batches, "cloudfree", "hst")

        xc = xc[:, 1:1299]
        xgc = xgc[:, self.mask]

        theta = thetac
        x = torch.cat([xc, xgc, xhc], dim=-1)

        return theta, x
    






    
    """
    Returns:
        {train:{
            cloudfree: {
                "miri": loader,
                "gemini": loader,
                "hst": loader
            },
            cloudy :{
                "miri": loader,
                "gemini": loader,
                "hst": loader
            },
        
        "valid": {...},
        "test": {...}
        }}

    i want to assign each loader to batches now as theta, x = 

    turn it into 
    dict[cf][miri] == trainset_cloudfree_miri


    """

