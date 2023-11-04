# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from typing import Optional

from llama import Llama, Dialog


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 6,
    max_gen_len: Optional[int] = None,
):
    """
    Entry point of the program for generating text using a pretrained model.
    """
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    

    while True:  # Infinite loop to keep asking the user for input
        user_input = input("User: ")  # Get input from the user
        
        if user_input.lower() in ['exit', 'quit']:  # Allow user to exit the loop
            print("Goodbye!")
            break

        dialog = [{"role": "user", "content": user_input}]
        result = generator.chat_completion(
            [dialog],  # type: ignore
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )[0]

        print(f"> Assistant: {result['generation']['content']}\n")


if __name__ == "__main__":
    main(ckpt_dir="llama-2-7b-chat/", tokenizer_path="tokenizer.model")
