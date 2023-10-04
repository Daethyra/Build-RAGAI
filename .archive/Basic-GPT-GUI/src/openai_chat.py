class OpenAI_Chat:
    def __init__(self, model=os.getenv('MODEL', 'gpt-3.5-turbo'), temperature=os.getenv('TEMPERATURE', .5)):
        self.model = model
        self.temperature = float(temperature)
        self.messages = []
