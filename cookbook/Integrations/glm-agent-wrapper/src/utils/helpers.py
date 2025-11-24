def get_model_response(model, prompt):
    response = model.get_response(prompt)
    return response

def stream_model_response(model, prompt):
    for response in model.stream_response(prompt):
        yield response

def validate_model_response(response):
    if not isinstance(response, dict):
        raise ValueError("Response must be a dictionary.")
    if 'output' not in response:
        raise KeyError("Response must contain 'output' key.")
    return response['output']