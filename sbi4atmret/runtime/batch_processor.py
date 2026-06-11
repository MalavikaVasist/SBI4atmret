class BatchProcessor:

    def __init__(
        self,
        pipe,
        noise,
        device,
    ):

        self.pipe = pipe
        self.noise = noise
        self.device = device

    def to_device(self, batch):

        if isinstance(batch, (list, tuple)):
            return [b.to(self.device) for b in batch]

        if isinstance(batch, dict):
            return {
                k: v.to(self.device)
                for k, v in batch.items()
            }

        return batch.to(self.device)

    def prepare_batch(
        self,
        batch_dict,
        add_noise=True,
        mode="train",
    ):

        processed_batch = self.pipe(batch_dict, mode=mode)

        if add_noise:
            processed_batch = self.noise(processed_batch)

        theta, x = self.pipe.build_input(processed_batch)

        theta = self.to_device(theta)
        x = self.to_device(x)

        return theta, x
    


