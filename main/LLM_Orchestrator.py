import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class Orchestrator:
    def __init__(self, threshold: float = 0.95, model: str = "openrouter/free"):
        self.threshold = threshold
        self.model = model
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY")
        )

    def call_llm(self, original: str, revised: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "Identify if this change affects legal liability or financial obligations. Be concise."
                },
                {
                    "role": "user",
                    "content": f"ORIGINAL:\n{original}\n\nREVISED:\n{revised}"
                }
            ]
        )
        return response.choices[0].message.content

    def analyze(self, paragraphs_a: list[str], paragraphs_b: list[str], scores: list[float]) -> list[dict]:
        if not (len(paragraphs_a) == len(paragraphs_b) == len(scores)):
            raise ValueError("paragraphs_a, paragraphs_b, and scores must all be the same length.")

        results = []
        for i, score in enumerate(scores):
            if score < self.threshold:
                print(f"Clause {i} flagged (score={score:.2f}) — calling LLM...")
                analysis = self.call_llm(paragraphs_a[i], paragraphs_b[i])
                results.append({
                    "clause_index": i,
                    "original_text": paragraphs_a[i],
                    "revised_text": paragraphs_b[i],
                    "score": round(score, 4),
                    "analysis": analysis
                })
            else:
                print(f"Clause {i} passed (score={score:.2f}) — skipping")

        return results