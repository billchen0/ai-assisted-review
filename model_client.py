import os
import openai
from langchain_ollama import OllamaLLM
from anthropic import Anthropic
from dotenv import load_dotenv

class ModelClient:
    def __init__(self):
        load_dotenv()

        # Ollama setup
        self.ollama = OllamaLLM(model="llama3.2")

        # OpenAI setup
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        openai.api_key = self.openai_api_key
        self.chatgpt = openai.OpenAI()

        # Anthropic setup
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        self.anthropic_client = Anthropic(api_key=self.anthropic_api_key)

    def ask_llama(self, prompt):
        response = self.ollama.invoke(prompt)
        return response
    
    def ask_chatgpt(self, prompt, model="gpt-4o-mini"):
        response = self.chatgpt.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content

    def ask_claude(self, prompt, model="claude-3-5-haiku-20241022"):
        response = self.anthropic_client.messages.create(
            model=model,
            max_tokens=4000,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text

    
def get_prompt(paper_text):
    screening_prompt = f"""
    You are a research paper reviewer. Review the following paper text and determine if it meets all inclusion criteria.

    Inclusion Criteria (ordered from most important to least important):
        1. Published in English
        2. Original peer-reviewed research (no books, dissertations, case reports, protocols, reviews or meta-analyses)
        3. Algorithm uses data from human subjects (collected by authors or from datasets)
        4. Primary aim is one of:
           - Proposing new algorithm
           - Extending/modifying existing algorithm
           - Training/evaluating existing algorithm(s) on new dataset/features
        5. Minimum 100 subjects (including pre-training dataset for transfer learning)
        6. Uses wearable device data as only time-series input (can include demographics)
        7. Uses data-driven model fitting with validation on held-out set

    Wearable Device Definition:
        - Unobtrusive electronic device worn on body
        - Continuously records physiological/inertial data
        - Comfortable for >1 hour wear
        - Minimal external wiring
        - Excludes: stationary medical equipment, clinical-grade equipment, implants, handheld tools, transient measurement tools, invasive devices, AR/VR headsets

    =====================================================================================
    Paper Text:
    {paper_text}
    =====================================================================================

    Provide your response in this format and keep the following important formatting instructions:
        - Do not have any other text in your response
        - This should go directly into ast.literal_eval and become a Python dictionary without further processing
        - Make sure the Key and Value are both wrapped in quotes and no out-of-place quotation marks are present
        - If a Value field is blank, just have a blank quotation field e.g. ""
        - Leave Reason and Note blank if the decision is to include
        - Do not leave < and > in your response

        {{
            "Decision": "<Include/Exclude>",
            "Reason": "<
                IF include: leave blank
                IF exclude: Choose the reason(s) from following list (ordered from most to least important): 
                Sample size too small, Not wearable devices, Wrong primary aim, 
                Not data-driven fitting procedure, Wrong input data, Not human subjects, Not original research, Study not in English.
                >",
            "Note": "[relevant quotes from paper supporting your exclusion reason. If no relevant quote just leave blank]"
        }}
    """

    return screening_prompt