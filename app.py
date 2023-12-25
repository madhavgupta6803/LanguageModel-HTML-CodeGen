import os
import chainlit as cl
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

@cl.on_chat_start
def main():
    # Create the model
    base_model_id = "TinyPixel/CodeLlama-7B-Instruct-bf16-sharded"
    adapter_id = "MG650/CodeLlama_HTML_FineTuned"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    llm = AutoModelForCausalLM.from_pretrained(
          base_model_id,
          quantization_config=bnb_config,
          device_map="auto",
          trust_remote_code=True,
    )
    llm.load_adapter(adapter_id)

    # Store the model and tokenizer in the user session
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, add_bos_token=True, trust_remote_code=True)
    cl.user_session.set("llm", llm)
    cl.user_session.set("tokenizer", tokenizer)

    # Set the chat title and icon
    cl.chat_title("Code Generation")
    cl.chat_icon("https://icons8.com/icon/fyoqSj1xVmwr/roll-of-tickets")

@cl.on_message
async def generate_html(message: cl.Message):
    # Retrieve the model and tokenizer from the user session
    llm = cl.user_session.get("llm")
    tokenizer = cl.user_session.get("tokenizer")

    # Create a message object
    msg = cl.Message(content="")

    # Define the prompt for HTML generation
    eval_prompt = f"Write HTML code for a simple school project website: {message.content}"
    model_input = tokenizer(eval_prompt, return_tensors="pt").to("cuda")

    # Generate the HTML code
    with torch.no_grad():
        generated_tokens = llm.generate(**model_input, max_new_tokens=300, repetition_penalty=1.2, top_k=4, do_sample=True)
        text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

    # Send the generated HTML code
    await msg.stream_token(text)
    await msg.send()

if __name__ == "__main__":
    main()
