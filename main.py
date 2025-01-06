import requests
import json
import sys


def generate_stream(prompt, model="mistral"):
    url = "http://localhost:11434/api/generate"

    data = {
        "model": model,
        "prompt": prompt,
        "stream": True
    }

    try:
        response = requests.post(url, json=data, stream=True)
        response.raise_for_status()

        for line in response.iter_lines():
            if line:
                json_response = json.loads(line)
                if 'response' in json_response:
                    sys.stdout.write(json_response['response'])
                    sys.stdout.flush()
                if json_response.get('done', False):
                    print()  # New line after response is complete
    except requests.exceptions.RequestException as e:
        print(f"Error occurred: {e}")


# Example usage
if __name__ == "__main__":
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'quit':
            break
        print("\nMistral:", end=" ")
        generate_stream(user_input)
