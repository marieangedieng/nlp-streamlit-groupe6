

# %%
import os
import tempfile

offload_dir = os.path.join(tempfile.gettempdir(), "offload_dir")
os.makedirs(offload_dir, exist_ok=True)

import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from huggingface_hub import login

# %%
MODEL_ID = "mistralai/Mistral-7B-v0.1"
ADAPTER_PATH = os.path.abspath("/result/mistral-finetuned-summarization")



secret_value="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"
login(secret_value)

# %%
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        trust_remote_code=True
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        offload_folder=offload_dir,
    )
    model = PeftModel.from_pretrained(
        base_model, 
        ADAPTER_PATH,
        offload_folder=offload_dir
    )
    model.eval()
    return tokenizer, model

# %%
tokenizer, model = load_model()

# %%
st.title("📚 Résumeur d'abstracts scientifiques (Mistral LoRA)")

abstract = st.text_area("✍️ Colle un abstract scientifique :", height=300)

if st.button("Générer le résumé") and abstract.strip():
    with st.spinner("Génération en cours..."):
        prompt = f"""
        ### INSTRUCTION:
        Résume l'abstract scientifique suivant en français.
        ### ABSTRACT:
        {abstract}
        ### RÉSUMÉ:
        """
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=256, pad_token_id=tokenizer.eos_token_id)
        resume = tokenizer.decode(outputs[0], skip_special_tokens=True).split("### RÉSUMÉ:")[-1].strip()

    st.subheader("📝 Résumé généré")
    st.write(resume)


