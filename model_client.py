from langchain_ollama import OllamaLLM

class ModelClient:
    def __init__(self):
        self.ollama = OllamaLLM(model="llama3.2")

    def ask_llama(self, prompt):
        response = self.ollama.invoke(prompt)
        
        return response
    

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

    Paper Text:
    {paper_text}

    Provide your response in this format (with no other text at all please):
        Paper Title: [extract from text, usually after mention of Article and before author list],
        Decision: [Include/Exclude],
        Reason: [IF exclude: 
                choose the reasons from following list ordered from most to least important 
                -- Sample size too small, Not wearable devices, Wrong primary aim, 
                Not data-driven fitting procedure, Wrong input data, Not human subjects, Not original research, Study not in English.
                IF include: leave blank],
        Note: [relevant quotes from paper supporting your exclusion reason. If no relevant quote just leave blank]
    """

    return screening_prompt