import unittest
from src.agents.interface import ModelResponse
from glm_agent_wrapper.models.glm_model import GLMModel


class TestGLMModel(unittest.TestCase):

    def setUp(self):
        self.model = GLMModel(model_name="GLM-4.5")

    def test_get_response(self):
        prompt = "What is the importance of evaluating AI agents?"
        response = self.model.get_response(prompt)
        self.assertIsInstance(response, str)
        self.assertNotEqual(response, "")

    def test_stream_response(self):
        prompt = "Explain the benefits of AI agent evaluation."
        responses = list(self.model.stream_response(prompt))
        self.assertTrue(all(isinstance(resp, str) for resp in responses))
        self.assertTrue(all(resp != "" for resp in responses))

    def test_model_name(self):
        self.assertEqual(self.model.get_model_name(), "GLM-4.5")


if __name__ == "__main__":
    unittest.main()
