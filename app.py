"""
app.py -- Gradio front end for the Research Paper RAG Assistant.

Loads the already-built index from storage/ (built by ingest_and_query.py)
and serves a simple web UI where you can type any question and get an
answer grounded in your indexed papers.

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

print("Loading index from storage...")
storage_context = StorageContext.from_defaults(persist_dir="storage")
index = load_index_from_storage(storage_context)
query_engine = index.as_query_engine()
print("Index loaded. Starting Gradio app...")

import gradio as gr


def answer_question(question: str) -> str:
    if not question.strip():
        return "Please enter a question."
    response = query_engine.query(question)
    return str(response)


demo = gr.Interface(
    fn=answer_question,
    inputs=gr.Textbox(
        label="Ask a question about your indexed papers",
        placeholder="e.g. What is the main idea behind LoRA?",
        lines=2,
    ),
    outputs=gr.Textbox(label="Answer", lines=10),
    title="Research Paper RAG Assistant",
    description="Ask questions about the papers indexed in this system. Answers are grounded in the actual paper text via retrieval-augmented generation.",
)

if __name__ == "__main__":
    demo.launch()
