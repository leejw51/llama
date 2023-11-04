torchrun --nproc_per_node 1 text_generator.py \
    --ckpt_dir="llama-2-7b-chat/" \
    --tokenizer_path="tokenizer.model" \
    --user_input="What is the recipe of mayonnaise?" \
    --max_seq_len=4000 \
    --max_batch_size=6
