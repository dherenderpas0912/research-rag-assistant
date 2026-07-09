# Research Paper RAG Assistant

A Retrieval-Augmented Generation (RAG) system for asking natural-language questions
about a collection of ML research papers (Transformer, RAG, LoRA, and others),
with answers grounded in the actual paper text rather than the LLM's memorized
training knowledge.

Built as part of a hands-on ML/Applied AI portfolio, while learning RAG fundamentals
end-to-end: embeddings, vector indexing, retrieval, and generation.

---

## What this project does

You ask a question like *"What is the main idea behind LoRA?"* — the system:

1. Converts your question into a vector embedding
2. Searches a pre-built vector index of ~500 chunks pulled from the indexed papers,
   finding the chunks most semantically similar to your question
3. Hands those retrieved chunks to Claude as context, alongside your question
4. Claude generates an answer **grounded in that retrieved text** — and will say
   so explicitly if the indexed papers don't cover what you asked, rather than
   guessing from general knowledge

---

## Why LlamaIndex (and not just LangChain)

Both LlamaIndex and LangChain can build RAG systems, but they're optimized for
different jobs:

| | LlamaIndex | LangChain |
|---|---|---|
| **Core strength** | Data ingestion, indexing, and retrieval | Multi-step orchestration, agents, tool-calling |
| **Best fit** | "Get the right chunk of my data in front of the model" | "Wire an LLM into a multi-step workflow with tools and memory" |
| **This project's needs** | A document set, a query, an answer — no branching logic | Not needed here — no multi-agent decision-making involved |

This project is a **pure retrieval problem** — no agent needs to decide between
multiple tools or take sequential actions. That's exactly LlamaIndex's specialty:
its document loaders, chunking, and indexing abstractions get a clean RAG pipeline
working in far fewer lines than assembling the same thing in raw LangChain.

(A separate project in this portfolio uses LangChain/LangGraph instead, since that
project *does* need multi-step agent orchestration and tool-routing across multiple
data sources. The two frameworks are complementary, not competing — this portfolio
deliberately uses each where it fits best.)

---

## Why OpenAI embeddings instead of Anthropic

Anthropic's API provides chat/generation models (Claude) but does not offer its
own embedding model — there is no "Claude embedding" to call. Anthropic's own
recommended embedding partner is Voyage AI; OpenAI's `text-embedding-3-small` is
another widely-used, low-cost option.

This project uses:
- **OpenAI (`text-embedding-3-small`)** for embeddings — turning text into vectors
  for retrieval
- **Anthropic (`claude-sonnet-4-6`)** for generation — reading retrieved context
  and writing the final answer

This split (one provider for embeddings, another for generation) is a common,
fully supported pattern — the two steps are independent and don't need to come
from the same vendor.

A local, free embedding option (Sentence-Transformers via HuggingFace) was
initially attempted, but PyTorch no longer publishes builds for Intel-based
Macs past version 2.2.2, which is incompatible with current `transformers`/
`sentence-transformers` releases requiring PyTorch 2.4+. Switching to a cloud
embedding API sidestepped this hardware limitation entirely.

---

## Architecture

```
papers/*.pdf
     │
     ▼
SimpleDirectoryReader   (extracts raw text from each PDF)
     │
     ▼
SentenceSplitter        (splits text into ~500-token chunks)
     │
     ▼
OpenAIEmbedding         (converts each chunk into a vector)
     │
     ▼
VectorStoreIndex        (stores chunks + vectors, saved to storage/)
     │
     ▼
Query Engine  ◄──────── your question (embedded the same way)
     │
     ▼
Claude (Anthropic)      (generates an answer from the retrieved chunks)
```

---

## Code walkthrough

### `ingest_and_query.py` — builds the index (run once, or after adding papers)

```python
from dotenv import load_dotenv
load_dotenv()
```
Loads `ANTHROPIC_API_KEY` and `OPENAI_API_KEY` from the local `.env` file into
the environment, so the rest of the script can authenticate with both APIs.

```python
from llama_index.core import Settings
from llama_index.embeddings.openai import OpenAIEmbedding

Settings.embed_model = OpenAIEmbedding(
    model="text-embedding-3-small",
    embed_batch_size=5,
)
```
`Settings` is a global configuration object LlamaIndex checks whenever it needs
a model, so this only has to be set once. `embed_batch_size=5` caps how many
chunks get embedded per API call — set low deliberately to stay under OpenAI's
Tier 1 rate limits (new accounts get a low requests-per-minute ceiling
regardless of account balance).

```python
from llama_index.llms.anthropic import Anthropic
Settings.llm = Anthropic(model="claude-sonnet-4-6")
```
Sets Claude as the model used for the *generation* step — writing the final
answer once relevant chunks have been retrieved.

```python
from llama_index.core import SimpleDirectoryReader

documents = SimpleDirectoryReader(
    input_dir="papers",
    recursive=True,
    required_exts=[".pdf"],
).load_data()
```
Reads every `.pdf` in `papers/` (and subfolders, via `recursive=True`) and
extracts their raw text into a list of `Document` objects. Internally, this
relies on `pypdf` and `llama-index-readers-file` to properly parse the PDF
format — without them, this step silently falls back to reading raw,
unparsed PDF bytes instead of clean text.

```python
from llama_index.core.node_parser import SentenceSplitter

splitter = SentenceSplitter(chunk_size=512)
nodes = splitter.get_nodes_from_documents(documents)
```
Breaks each document into smaller pieces ("nodes") of roughly 512 tokens each.
Chunking matters because embedding an entire paper as one giant vector would
lose precision — smaller chunks let retrieval find the *specific* passage
relevant to a question, not just "some paper is vaguely relevant."

```python
index = VectorStoreIndex(nodes=[])

batch_size = 5
for i in range(0, len(nodes), batch_size):
    batch = nodes[i : i + batch_size]
    index.insert_nodes(batch)
    print(f"Inserted batch {i // batch_size + 1}/{total_batches}")
    time.sleep(2)
```
Rather than embedding all ~500 nodes in one call (which triggers rate-limit
errors on OpenAI's lowest usage tier), this manually embeds and inserts them
in small groups of 5, pausing 2 seconds between each group. Slower, but
reliable regardless of account tier.

```python
index.storage_context.persist(persist_dir="storage")
```
Saves the completed index — all chunks and their vectors — to disk in the
`storage/` folder, so it can be loaded instantly later without re-embedding
(re-embedding costs API calls and takes minutes; loading from disk takes
seconds).

```python
query_engine = index.as_query_engine()
response = query_engine.query(q)
```
Wraps the index in a query engine, which handles the full retrieve-then-
generate pipeline in one call: embed the question → find the most similar
chunks → send those chunks + the question to Claude → return the answer.

---

### `app.py` — the interactive front end

```python
storage_context = StorageContext.from_defaults(persist_dir="storage")
index = load_index_from_storage(storage_context)
```
This is the key difference from `ingest_and_query.py`: instead of rebuilding
the index from the PDFs, this loads the **already-built** index straight from
`storage/` — instant, and free (no embedding API calls needed just to load it).

```python
def answer_question(question: str) -> str:
    if not question.strip():
        return "Please enter a question."
    response = query_engine.query(question)
    return str(response)
```
A plain Python function: takes a question string in, returns an answer string
out. This is the bridge between the UI (Gradio) and the RAG logic (LlamaIndex)
— Gradio doesn't know anything about embeddings or retrieval, it just knows
"call this function with whatever the user typed, and display whatever comes
back."

```python
demo = gr.Interface(
    fn=answer_question,
    inputs=gr.Textbox(label="...", placeholder="...", lines=2),
    outputs=gr.Textbox(label="Answer", lines=10),
    title="Research Paper RAG Assistant",
)
```
Builds the actual web UI: a text box for the question (`inputs`), a text box
for the answer (`outputs`), and wires them together via `fn` — whatever you
type gets passed as the `question` argument to `answer_question`, and whatever
that function returns gets displayed in the output box.

```python
demo.launch()
```
Starts a local web server and prints the URL to open in a browser.

---

## Setup

```bash
conda create -n rag-env python=3.11
conda activate rag-env
pip install llama-index llama-index-llms-anthropic llama-index-embeddings-openai python-dotenv gradio pypdf llama-index-readers-file
```

Create a `.env` file:
```
ANTHROPIC_API_KEY=your-anthropic-key
OPENAI_API_KEY=your-openai-key
```

Add PDFs to a `papers/` folder, then:
```bash
python ingest_and_query.py   # builds and saves the index (run once, or after adding papers)
python app.py                 # launches the Gradio UI at http://127.0.0.1:7860
```

---

## Papers currently indexed

- *Attention Is All You Need* (Vaswani et al., 2017) — the Transformer architecture
- *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* (Lewis et al., 2020)
- *LoRA: Low-Rank Adaptation of Large Language Models* (Hu et al., 2021)

---

## Possible next steps

- Source citations in answers (show which paper/chunk backed each response)
- A small evaluation set to measure retrieval accuracy systematically
- Incremental indexing (only re-embed new/changed papers, not the whole set)

---

## Deployment (public, shareable link)

Running `python app.py` locally only serves the app on your own machine
(`http://127.0.0.1:7860`) — nobody else can open that link. To get a permanent,
public link suitable for a resume or portfolio:

1. Create a free [Hugging Face](https://huggingface.co/join) account
2. Create a new [Space](https://huggingface.co/new-space), choosing **Gradio**
   as the SDK
3. Push `app.py`, the `storage/` folder (the pre-built index — needed since HF
   Spaces doesn't rebuild it from PDFs), and a `requirements.txt` to the Space's
   git repo
4. Add `ANTHROPIC_API_KEY` and `OPENAI_API_KEY` as **Secrets** in the Space's
   settings (not committed in code)
5. The Space builds automatically and serves a permanent public URL, e.g.
   `https://huggingface.co/spaces/<username>/research-paper-rag-assistant`
