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

        def reconstruct(
            self,
            batches,
            keys,
        ):

            return self.dataset.reconstruct_batch(
                keys,
                batches,
            )

        def process(
            self,
            batch_dict,
        ):

            return self.pipe(batch_dict)

        def corrupt(
            self,
            processed_batch,
            add_noise=True,
        ):

            if add_noise:
                return self.noise(processed_batch)

            return processed_batch

        def build_input(
            self,
            batch_dict,
        ):

            theta, x = self.pipe.build_input(
                batch_dict
            )

            return (
                self.to_device(theta),
                self.to_device(x),
            )

        def prepare_batch(
            self,
            batches,
            keys,
            add_noise=True,
        ):

            batch_dict = self.reconstruct(
                batches,
                keys,
            )

            processed = self.process(batch_dict)

            corrupted = self.corrupt(
                processed,
                add_noise=add_noise,
            )

            return self.build_input(corrupted)

        def to_device(self, batch):

            if isinstance(batch, (list, tuple)):
                return [b.to(self.device) for b in batch]

            if isinstance(batch, dict):
                return {
                    k: v.to(self.device)
                    for k, v in batch.items()
                }

            return batch.to(self.device)