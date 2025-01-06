# Import required libraries for vector processing, API calls, and text encoding
import numpy as np
import pickle
import os
import requests
from sentence_transformers import SentenceTransformer

#1.download ollama
#2.on terminal, "ollama run mistral"

class ChocolateVectorDB:
    def __init__(self):
        # Initialize the encoder that converts text to vectors
        # all-MiniLM-L6-v2 is a good balance of speed and accuracy for semantic search
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.vectors = []  # Will store vector representations of our text data
        self.data = []  # Will store the original text data

        # Predefined chocolate knowledge base
        # Each entry represents a distinct piece of information about chocolate
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

        # Add each piece of information to our database
        for text in chocolate_data:
            self.add_text(text)

    def add_text(self, text):
        # Convert text to a numerical vector representation
        # This allows us to mathematically compare pieces of text
        vector = self.encoder.encode([text])[0]
        self.vectors.append(vector)
        self.data.append(text)

    def search(self, query, k=3):
        # Convert the search query to a vector
        query_vector = self.encoder.encode([query])

        # Calculate cosine similarity between query and all stored vectors
        # Cosine similarity measures how similar the directions of two vectors are
        similarities = [np.dot(query_vector, vec) / (np.linalg.norm(query_vector) * np.linalg.norm(vec))
                        for vec in self.vectors]

        # Return the k most similar texts
        # sorted(range(len(similarities))...) creates an index list sorted by similarity scores
        return [self.data[i] for i in sorted(range(len(similarities)),
                                             key=lambda i: similarities[i], reverse=True)[:k]]


def chat_about_chocolate():
    # Initialize our chocolate knowledge base
    db = ChocolateVectorDB()
    print("Ask questions about chocolate (type 'quit' to exit)")

    while True:
        # Get user input
        query = input("\nYou: ")
        if query.lower() == 'quit':
            break

        # Search for relevant information in our vector database
        relevant_info = db.search(query)

        # Combine all relevant information into a single context string
        context = " ".join(relevant_info)

        # Construct a prompt that includes both context and question
        # This helps Mistral provide more accurate, informed answers
        prompt = f"Using this information about chocolate: {context}\n\nQuestion: {query}\nAnswer:"

        # Send the prompt to Mistral API and get response
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": "mistral", "prompt": prompt, "stream": False}
            )
            print(f"\nMistral: {response.json()['response']}")
        except Exception as e:
            print(f"Error communicating with Mistral: {e}")


# Entry point of the program
if __name__ == "__main__":
    try:
        chat_about_chocolate()
    except Exception as e:
        print(f"An error occurred: {e}")