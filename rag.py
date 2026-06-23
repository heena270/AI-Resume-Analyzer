from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")


def create_vector_store(text):

    chunks = []

    chunk_size = 1000

    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i + chunk_size])

    embeddings = model.encode(chunks)

    return {
        "chunks": chunks,
        "embeddings": embeddings
    }


def retrieve_context(vector_store, question):

    question_embedding = model.encode([question])[0]

    similarities = np.dot(
        vector_store["embeddings"],
        question_embedding
    )

    top_indices = similarities.argsort()[-3:][::-1]

    context = "\n".join(
        vector_store["chunks"][i]
        for i in top_indices
    )

    return context
