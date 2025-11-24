from agents.interface import Model
from models.glm_model import GLMModel

class Config:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = self.load_model()

    def load_model(self) -> Model:
        return GLMModel(model_name=self.model_name)

    def get_model(self) -> Model:
        return self.model

    def validate(self):
        if not isinstance(self.model, Model):
            raise ValueError("Loaded model does not adhere to the Model interface.")