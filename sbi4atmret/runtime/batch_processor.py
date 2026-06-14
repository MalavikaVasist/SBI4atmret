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

        if batch is None:
            return batch

        if isinstance(batch, (list, tuple)):
            return [b.to(self.device) for b in batch]

        if isinstance(batch, dict):
            return {
                k: v.to(self.device)
                for k, v in batch.items()
            }

        return batch.to(self.device)


    def process(self, batch_dict, mode="train", add_noise=True):
        """
        Apply pipe transforms (and optionally noise).
        Returns the processed batch_dict (per-instrument dict).
        """
        processed = self.pipe(batch_dict, mode=mode)
        if add_noise:
            processed = self.noise(processed)
        return processed

    def merge(self, processed_batch, mode= "train"):
        """
        Merge processed batch_dict into (theta, x) tensors on device.
        """
        theta, x = self.pipe.build_input(processed_batch, mode= mode)
        return self.to_device(theta), self.to_device(x)

    def prepare_batch(self, batch_dict, add_noise=True, mode="train"):
        """
        Full pipeline: process + merge. Used during training.
        """
        processed = self.process(batch_dict, mode=mode, add_noise=add_noise)
        return self.merge(processed, mode=mode)



