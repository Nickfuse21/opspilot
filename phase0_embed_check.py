from sentence_transformers import SentenceTransformer

# Load the embedding model directly onto the GPU (device="cuda").
# First run downloads the model (~80 MB) from Hugging Face, then caches it.
model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")

sentence = "How do I reset my password?"

# encode() turns the text into its 384-number meaning vector.
vector = model.encode(sentence)

print("Model device:", model.device)        # should mention cuda
print("Vector length:", len(vector))         # should be 384
print("First 5 numbers:", vector[:5])        # a peek at the vector
