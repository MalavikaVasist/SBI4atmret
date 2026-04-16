from dataclasses import Field
from typing import Any
from pydantic import BaseModel
from scripts.train_general import load_callable


class EmbeddingConfig(BaseModel):
    """Configuration for embedding model."""
    module: str
    class_name: str = Field(alias="class")
    instrument: str
  
    
    def __init__(self, config):
        self.config = config
        self.estimator = EstimatorConfig

    @field_validator('type')
    def check_estimator_type(cls: Any, v: str) -> str:
        try:
            getattr(load_callable, v)
        except AttributeError:
            raise ValueError(f"Invalid estimator type '{v}'. Must be a valid class in load_callable.")
        return v

    

    