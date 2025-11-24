from agents.interface import Model, ModelResponse
from models.glm_model import GLMModel

class LangfuseGLMWrapper(Model):
    def __init__(self, model: GLMModel):
        self.model = model

    def get_response(self, prompt: str) -> ModelResponse:
        response = self.model.generate(prompt)
        return ModelResponse(final_output=response)

    async def stream_response(self, prompt: str):
        async for chunk in self.model.a_generate(prompt):
            yield ModelResponse(final_output=chunk)