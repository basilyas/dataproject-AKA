import numpy as np
import pickle
import os
import requests
from sentence_transformers import SentenceTransformer

#1.download ollama
#2.on terminal, "ollama run mistral"


class ChocolateVectorDB:
    def __init__(self):
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.vectors = []
        self.data = []

        # Initialize with chocolate data
        chocolate_data = [
            "Dark chocolate contains 50-90% cocoa solids, cocoa butter, and sugar. It's rich in antioxidants and has a bitter taste.",
            "Milk chocolate contains 10-50% cocoa solids, cocoa butter, milk, and sugar. It has a creamy, sweet taste.",
            "White chocolate contains cocoa butter, milk, and sugar but no cocoa solids. It has a sweet, vanilla taste.",
            "Ruby chocolate is made from ruby cocoa beans, has a natural pink color, and berry-like taste.",
            "The main cocoa producing countries are Ivory Coast, Ghana, and Ecuador.",
            "Chocolate making process: harvesting, fermenting, drying, roasting, grinding, conching, and tempering.",
            "Health benefits of dark chocolate include improved heart health, lower blood pressure, and mood enhancement.",
            "Popular chocolate brands include Lindt, Godiva, Ghirardelli, and Valrhona.",
            "Chocolate storage: keep at 65-70°F, away from direct sunlight, in airtight containers."
        ]

        for text in chocolate_data:
            self.add_text(text)

    def add_text(self, text):
        vector = self.encoder.encode([text])[0]
        self.vectors.append(vector)
        self.data.append(text)

    def search(self, query, k=3):
        query_vector = self.encoder.encode([query])
        similarities = [np.dot(query_vector, vec) / (np.linalg.norm(query_vector) * np.linalg.norm(vec))
                        for vec in self.vectors]
        return [self.data[i] for i in sorted(range(len(similarities)),
                                             key=lambda i: similarities[i], reverse=True)[:k]]


def chat_about_chocolate():
    db = ChocolateVectorDB()
    print("Ask questions about chocolate (type 'quit' to exit)")

    while True:
        query = input("\nYou: ")
        if query.lower() == 'quit':
            break

        # Search vector DB
        relevant_info = db.search(query)
        context = " ".join(relevant_info)

        # Query Mistral with context
        prompt = f"Using this information about chocolate: {context}\n\nQuestion: {query}\nAnswer:"

        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "mistral", "prompt": prompt, "stream": False}
        )
        print(f"\nMistral: {response.json()['response']}")


if __name__ == "__main__":
    chat_about_chocolate()