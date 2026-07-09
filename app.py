"""
app.py -- Gradio front end for the Research Paper RAG Assistant.

Loads the already-built vector index from storage/, combines it with a
BM25 keyword retriever for hybrid search, and wraps both in a chat engine
with conversational memory (so follow-up questions like "how does it
compare to X?" resolve correctly instead of being treated in isolation).

Run:
    python app.py
"""

from dotenv import load_dotenv
load_dotenv()

from llama_index.core import Settings
from llama_index.embeddings.openai import OpenAIEmbedding

Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

from llama_index.llms.anthropic import Anthropic
Settings.llm = Anthropic(model="claude-sonnet-4-6")

from llama_index.core import StorageContext, load_index_from_storage

print("Loading vector index from storage...")
storage_context = StorageContext.from_defaults(persist_dir="storage")
index = load_index_from_storage(storage_context)

# ---------------------------------------------------------------------
# Hybrid search: combine dense (vector) retrieval with BM25 (keyword)
# retrieval. BM25 needs the raw nodes -- rebuilding them from the source
# PDFs is fast and free (no embedding API calls involved), since BM25
# doesn't use embeddings at all.
# ---------------------------------------------------------------------
print("Rebuilding nodes for BM25 keyword search...")
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter

documents = SimpleDirectoryReader(
    input_dir="papers", recursive=True, required_exts=[".pdf"]
).load_data()
splitter = SentenceSplitter(chunk_size=512)
nodes = splitter.get_nodes_from_documents(documents)

from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever

vector_retriever = index.as_retriever(similarity_top_k=5)
bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=5)

fusion_retriever = QueryFusionRetriever(
    [vector_retriever, bm25_retriever],
    similarity_top_k=5,
    num_queries=1,  # set >1 to also generate query variations (RAG Fusion-style)
    mode="reciprocal_rerank",
)

# ---------------------------------------------------------------------
# Conversational memory: condense_plus_context rewrites each follow-up
# question into a standalone question (using chat history) BEFORE
# retrieval runs, so pronouns like "it" or "that" resolve correctly.
# ---------------------------------------------------------------------
from llama_index.core.chat_engine import CondensePlusContextChatEngine

chat_engine = CondensePlusContextChatEngine.from_defaults(
    retriever=fusion_retriever,
    llm=Settings.llm,
)

print("Ready. Starting Gradio app...")

import gradio as gr


def answer_question(question, history):
    if not question.strip():
        return "Please enter a question."
    response = chat_engine.chat(question)
    return str(response)


demo = gr.ChatInterface(
    fn=answer_question,
    title="Research Paper RAG Assistant",
    description=(
        "Ask questions about the indexed papers (Transformer, BERT, RAG, "
        "GPT-3, LoRA). Uses hybrid search (semantic + keyword) and "
        "remembers conversation context, so follow-up questions work "
        "naturally."
    ),
)

if __name__ == "__main__":
    demo.launch()
