"""
ingest_and_query.py -- Research Paper RAG Assistant

Loads PDFs from the papers/ folder, builds a vector index using
OpenAI embeddings (paced to avoid Tier 1 rate limits), saves the
index to disk, then runs a couple of test queries using Claude.
"""

import time
from dotenv import load_dotenv

load_dotenv()

from llama_index.core import Settings
from llama_index.embeddings.openai import OpenAIEmbedding

Settings.embed_model = OpenAIEmbedding(
    model="text-embedding-3-small",
    embed_batch_size=5,
)

from llama_index.llms.anthropic import Anthropic

Settings.llm = Anthropic(model="claude-sonnet-4-6")

from llama_index.core import SimpleDirectoryReader

documents = SimpleDirectoryReader(
    input_dir="papers",
    recursive=True,
    required_exts=[".pdf"],
).load_data()

print(f"Loaded {len(documents)} document chunks")
print(documents[0].metadata)
print(documents[0].text[:500])

# ---------------------------------------------------------------------
# Build the index in small, paced batches to stay under OpenAI's
# Tier 1 rate limits (avoids repeated 429 Too Many Requests errors).
# ---------------------------------------------------------------------
from llama_index.core import VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter

splitter = SentenceSplitter(chunk_size=512)
nodes = splitter.get_nodes_from_documents(documents)
print(f"Split into {len(nodes)} nodes")

index = VectorStoreIndex(nodes=[])

batch_size = 5
total_batches = (len(nodes) // batch_size) + 1
for i in range(0, len(nodes), batch_size):
    batch = nodes[i : i + batch_size]
    index.insert_nodes(batch)
    print(f"Inserted batch {i // batch_size + 1}/{total_batches}")
    time.sleep(2)  # pause between batches to respect rate limits

print("Index built successfully")

index.storage_context.persist(persist_dir="storage")
print("Index saved to storage/")

# ---------------------------------------------------------------------
# Quick test queries -- confirms retrieval + generation work end to end
# ---------------------------------------------------------------------
query_engine = index.as_query_engine()

test_questions = [
    "What problem does the attention mechanism solve?",
    "What is the main idea behind LoRA?",
]

for q in test_questions:
    print("\n" + "=" * 60)
    print(f"QUESTION: {q}")
    print("=" * 60)
    response = query_engine.query(q)
    print(response)