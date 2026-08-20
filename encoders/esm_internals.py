from transformers import AutoModel, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
model = AutoModel.from_pretrained("facebook/esm2_t6_8M_UR50D")

# Full vocabulary (token -> id mapping)
print(tokenizer.get_vocab())

# sort by id for readability
vocab = tokenizer.get_vocab()
for token, idx in sorted(vocab.items(), key=lambda x: x[1]):
    print(idx, repr(token))

peptide = "MKTAYIAKX"  # sequence with "X" for unknown AA
peptide2 = "MKTAYIAK-" # sequence with "-" for unknown AA
encoded = tokenizer(peptide, return_tensors="pt")
print(encoded["input_ids"])
print(tokenizer.convert_ids_to_tokens(encoded["input_ids"][0]))


# raw embedding matrix
embed_matrix = model.embeddings.word_embeddings.weight  # shape [vocab_size, hidden_dim]
print(embed_matrix.shape)

vocab = tokenizer.get_vocab()
standard_aas = list("ACDEFGHIKLMNPQRSTVWY")
standard_ids = [vocab[aa] for aa in standard_aas]

# Compare X's embedding to the average of standard AA embeddings
x_id = vocab["X"]
x_embed = embed_matrix[x_id]
avg_embed = embed_matrix[standard_ids].mean(dim=0)

cos_sim = torch.nn.functional.cosine_similarity(x_embed.unsqueeze(0), avg_embed.unsqueeze(0))
print("X vs mean(standard AAs) cosine similarity:", cos_sim.item())

# Compare norms
print("X embedding norm:", x_embed.norm().item())
print("Standard AA embedding norms:", embed_matrix[standard_ids].norm(dim=1))
