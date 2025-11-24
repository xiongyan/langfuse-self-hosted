from .interface import Model

class GLMModel(Model):
    def get_response(self, prompt: str) -> str:
        # Implement the logic to get a response from the GLM model
        pass

    def stream_response(self, prompt: str):
        # Implement the logic to stream a response from the GLM model
        pass