import gradio as gr
from llama import Llama, Dialog
from typing import Optional

# Hardcoded values
CKPT_DIR = "llama-2-7b-chat/"
TOKENIZER_PATH = "tokenizer.model"
MAX_SEQ_LEN = 2048
MAX_BATCH_SIZE = 6
TEMPERATURE = 0.6
TOP_P = 0.9
MAX_GEN_LEN = None  # Optional value

# Global variable to store the generator
generator = None

def chat_response(prompt: str) -> str:
    """Function to get the model's response based on the user prompt."""
    dialog = [{"role": "user", "content": prompt}]
    result = generator.chat_completion(
        [dialog],
        max_gen_len=MAX_GEN_LEN,
        temperature=TEMPERATURE,
        top_p=TOP_P,
    )[0]
    return result['generation']['content']

def main_gradio():
    """Main function to launch Gradio interface."""
    global generator
    generator = Llama.build(
        ckpt_dir=CKPT_DIR,
        tokenizer_path=TOKENIZER_PATH,
        max_seq_len=MAX_SEQ_LEN,
        max_batch_size=MAX_BATCH_SIZE,
    )
    
    # Define Gradio interface
    iface = gr.Interface(
        fn=chat_response,
        inputs="text",
        outputs="text",
        live=False,  # Set to False to add a submit button
        title="AI Chat",
        description="Chat with AI Finetuned",
    )
    iface.launch(share=True)

if __name__ == "__main__":
    main_gradio()
