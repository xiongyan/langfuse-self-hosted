class Runner:
    def __init__(self, model):
        self.model = model

    async def run(self, task: str):
        response = await self.model.get_response(task)
        return response

    async def stream(self, task: str):
        async for response in self.model.stream_response(task):
            yield response