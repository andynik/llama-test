#!/bin/python
import os
import sys
from termcolor import colored
from llama_cpp import Llama

MODEL_PATH = "Llama-2-7B-Chat-GGUF/llama-2-7b-chat.Q4_K_M.gguf"
MAX_TOKENS = 2048

def run_console_app():
    if not os.path.isfile(MODEL_PATH):
        print(f"Error: Invalid model path '{MODEL_PATH}'")
        sys.exit(1)

    llm = Llama(model_path=MODEL_PATH, n_gpu_layers=-1)

    print(colored("\n## Welcome to the Llama Console App! ##\n", "yellow"))
    print("Enter your prompt (or 'q' to quit):")

    while True:
        user_input = input("> ")
        if user_input.lower() == 'q':
            print("Exiting...")
            break

        user_input = f"Q: {user_input} A:"
        print(colored("Processing...", "yellow"))

        try:
            output = llm(user_input, max_tokens=MAX_TOKENS, stop=["Q:"], echo=True)

            choices_text = output["choices"][0]["text"]
            choices_text = choices_text.replace(user_input, "").strip()
            formatted_text = colored(choices_text, "green")
            print(f"\n{formatted_text}\n")
        except Exception as e:
            print(f"Error: Failed to generate response. {str(e)}")

if __name__ == '__main__':
    run_console_app()
