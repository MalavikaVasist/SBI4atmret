
def setup_pipe(self):
    """Set up training pipeline."""
    if self.loss is None:
        raise ValueError("Loss must be set up before setting up pipe")

    config = self.selected_config if self.selected_config is not None else self.config
    pipe_config = config.pipe
    
    if pipe_config is None:
        raise KeyError('Pipe configuration not found')

    # Use attribute access from PipeConfig
    pipe_func = load_callable(pipe_config.module, pipe_config.function)
    self.pipe = pipe_func(self.loss)
    return self.pipe