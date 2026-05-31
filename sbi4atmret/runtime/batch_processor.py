class BatchProcessor:

    def __init__(
        self,
        dataset,
        pipe,
        noise,
        device,
    ):

        self.dataset = dataset
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
        batches,
        keys,
        add_noise=True,
    ):

        batch_dict = self.dataset.reconstruct_batch(keys, batches)
        processed_batch = self.pipe(batch_dict)

        if add_noise:
            processed_batch = self.noise(processed_batch)

        theta, x = self.pipe.build_input(processed_batch)

        theta = self.to_device(theta)
        x = self.to_device(x)

        return theta, x
    


