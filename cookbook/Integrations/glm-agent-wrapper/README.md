# GLM Agent Wrapper

## Overview

The GLM Agent Wrapper is a Python package designed to provide a seamless interface for integrating GLM models into agent-based architectures. This project adheres to the `agents.models.interface.Model` interface, ensuring compatibility and ease of use within the agent framework.

## Features

- Implements the `GLMModel` class that adheres to the `Model` interface.
- Provides methods for obtaining responses from the GLM model, including `get_response` and `stream_response`.
- Integrates with Langfuse for monitoring and instrumentation.
- Supports easy configuration and management of agent settings.

## Installation

To install the required dependencies, run:

```
pip install -r requirements.txt
```

## Usage

To use the GLM Agent Wrapper, you can create an instance of the `GLMModel` and utilize its methods to interact with the model. Here is a basic example:

```python
from agents.runner import Runner
from models.glm_model import GLMModel

# Initialize the GLM model
glm_model = GLMModel("GLM-4.5")

# Get a response from the model
response = glm_model.get_response("What is the importance of evaluating AI agents?")
print(response)
```

## Testing

Unit tests for the `GLMModel` class are provided in the `tests/test_glm_model.py` file. To run the tests, use:

```
pytest tests/test_glm_model.py
```

## Contributing

Contributions are welcome! Please submit a pull request or open an issue for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.
