from langchain_ollama import OllamaLLM

class ModelClient:
    def __init__(self):
        # Initialize Anthropic (Claude)
        self.ollama = OllamaLLM(model="llama3.2")

    def ask_llama(self, prompt):
        response = self.ollama.invoke(prompt)
        return response