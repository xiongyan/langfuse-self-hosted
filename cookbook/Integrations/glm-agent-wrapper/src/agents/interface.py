class ModelResponse:
    def __init__(self, output: str, stream: bool = False):
        self.output = output
        self.stream = stream


class Model:
    def get_response(self, prompt: str) -> ModelResponse:
        raise NotImplementedError

    def stream_response(self, prompt: str):
        raise NotImplementedError


class GLMModel(Model):
    def __init__(self, model_name: str):
        self.model_name = model_name

    def get_response(self, prompt: str) -> ModelResponse:
        # Implement the logic to get a response from the GLM model
        output = f"Response from {self.model_name} for prompt: {prompt}"
        return ModelResponse(output)

    def stream_response(self, prompt: str):
        # Implement the logic to stream responses from the GLM model
        yield f"Streaming response from {self.model_name} for prompt: {prompt}"