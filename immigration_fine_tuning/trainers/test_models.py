import os
from transformers import AutoModel, AutoTokenizer

# Ensure HF cache location is set
os.environ["HF_HOME"] = "/data/resource/huggingface"



# Load LegalBERT
print("Loading LegalBERT...")
legal_bert_tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
legal_bert_model = AutoModel.from_pretrained("nlpaueb/legal-bert-base-uncased")

# Test with some example text
sample_text = "This is a test sentence for model evaluation."


# Tokenize and test LegalBERT
print("\nTesting LegalBERT...")
legal_bert_inputs = legal_bert_tokenizer(sample_text, return_tensors="pt")
legal_bert_outputs = legal_bert_model(**legal_bert_inputs)
print("LegalBERT output:", legal_bert_outputs.last_hidden_state.shape)

