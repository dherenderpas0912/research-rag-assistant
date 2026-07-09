# RAG & LlamaIndex — Interview Questions and Answers

A study guide covering Retrieval-Augmented Generation and LlamaIndex concepts,
organized by difficulty. Built alongside the Research Paper RAG Assistant
project in this repo.

---

## Beginner

### 1. What is RAG, in one sentence?
RAG (Retrieval-Augmented Generation) is a technique that retrieves relevant
external documents at query time and feeds them to an LLM as context, so the
model answers using that retrieved information instead of relying only on its
frozen training-time knowledge.

### 2. Why can't we just ask an LLM directly instead of using RAG?
LLMs have a training cutoff — they know nothing about events, documents, or
data created after that point, and nothing about private/internal data they
were never trained on. Their knowledge is also baked into billions of frozen
parameters, not stored as retrievable facts, so it can't be selectively
updated. RAG solves both problems by injecting current, relevant, or private
information directly into the prompt at the moment of the question.

### 3. What are the core steps in a basic RAG pipeline?
1. **Load** documents (PDFs, web pages, database rows, etc.)
2. **Chunk** them into smaller pieces
3. **Embed** each chunk into a vector
4. **Store** those vectors in a vector database/index
5. **Retrieve** the most relevant chunks for a given question (via similarity
   search)
6. **Augment** the prompt with those retrieved chunks
7. **Generate** an answer using an LLM, grounded in that context

### 4. What is an embedding?
A numerical vector representation of text, where semantically similar text
produces vectors that are close together in vector space, and unrelated text
produces vectors that are far apart. This is what allows a system to match a
question to relevant content by meaning, not just exact keyword overlap.

### 5. What is chunking, and why does it matter?
Chunking splits large documents into smaller pieces before embedding them.
It matters because embedding an entire long document as a single vector loses
precision — a giant, averaged-out vector can't represent multiple distinct
ideas well. Smaller chunks let retrieval pinpoint the specific passage
relevant to a question, rather than vaguely matching "some big document."

### 6. What is a vector database, and why not just use a regular SQL database?
A vector database is optimized for storing embeddings and performing fast
similarity search (e.g. "find the 5 vectors closest to this one") across
potentially millions of vectors. Regular SQL databases are built for exact-
match or range queries on structured columns — they aren't designed to
efficiently answer "which rows are semantically closest to this query,"
which requires specialized indexing (like approximate nearest neighbor
search).

### 7. What does "grounded" mean in the context of RAG?
An answer is "grounded" when it's directly supported by the retrieved
context provided to the LLM, rather than generated purely from the model's
internal, memorized knowledge. Grounded answers are more verifiable and less
prone to hallucination.

### 8. What is hallucination, and how does RAG help reduce it?
Hallucination is when an LLM generates fluent, confident-sounding text that
is factually incorrect or fabricated — because the model is optimized to
produce plausible continuations, not to signal uncertainty. RAG reduces this
by giving the model real, relevant text to read and summarize/reason over,
which is a fundamentally easier and more reliable task than recalling a fact
from memory. A well-instructed RAG system can also explicitly tell the model
to say "I don't know" if the retrieved context doesn't contain the answer.

### 9. What is LlamaIndex, at a high level?
A data framework purpose-built for connecting LLMs to external data —
focused on document loading, chunking, indexing, and retrieval. It's
narrower in scope than a general orchestration framework like LangChain, but
gives faster, more opinionated defaults specifically for RAG use cases.

### 10. In LlamaIndex, what does `SimpleDirectoryReader` do?
It reads all files of specified types from a given folder (optionally
including subfolders via `recursive=True`), extracts their text content, and
returns a list of `Document` objects ready for chunking and embedding.

---

## Intermediate

### 11. What's the difference between cosine similarity and dot product for retrieval?
Cosine similarity measures the angle between two vectors, ignoring their
magnitude — it captures whether two pieces of text point in the same
"semantic direction," regardless of length or intensity. Dot product
factors in both direction and magnitude, so longer or more emphatic text can
score higher purely due to magnitude, not just semantic alignment. Cosine
similarity is generally preferred when you want a pure "directional"
similarity signal; dot product can be preferable when the embedding model
was specifically trained such that magnitude carries meaningful signal (some
models normalize vectors so the two become equivalent).

### 12. What is the role of `top K` in retrieval, and how do you choose K?
K is the number of most-similar chunks retrieved for a given query. Too low
a K risks missing relevant context (especially if the answer spans multiple
chunks); too high a K adds noise, increases token usage/cost, and can dilute
the LLM's attention across less-relevant material. K is typically chosen
empirically — starting around 3-5 and tuning based on evaluation results
(does retrieval quality improve or plateau as K increases?).

### 13. What happens during "augmentation" in RAG, concretely?
The retrieved chunks are inserted directly into the prompt sent to the LLM,
typically alongside an instruction (e.g. "answer using only the context
below") and the original user question. The LLM never receives the chunks
and question as separate inputs — they're combined into one text block that
becomes the model's context window for that generation.

### 14. Why might you choose LlamaIndex over LangChain for a project, and vice versa?
LlamaIndex is optimized for retrieval-heavy workflows — document ingestion,
indexing, and querying, with strong out-of-the-box support for advanced
retrieval patterns (hybrid search, reranking, query decomposition).
LangChain is optimized for orchestration — multi-step agents, tool-calling,
branching logic, and stateful workflows via LangGraph. A pure "chat with my
documents" RAG system fits LlamaIndex's strengths; a system where an agent
needs to decide between multiple tools or coordinate multi-step actions
fits LangChain's strengths. Many production systems use both together:
LlamaIndex for the retrieval layer, LangChain for orchestration on top.

### 15. What's the difference between a `Document` and a `Node` in LlamaIndex?
A `Document` typically represents a whole source file (e.g. one PDF). A
`Node` is a smaller chunk derived from a Document after splitting (e.g. via
`SentenceSplitter`) — Nodes are the actual units that get embedded and
stored in the index, since a whole Document is usually too large to embed
as a single vector meaningfully.

### 16. Why does chunk size matter, and what's the tradeoff?
Smaller chunks give more precise retrieval (less irrelevant text bundled
into a single embedding) but risk losing context that spans chunk
boundaries, and increase the total number of chunks (more storage, more
embedding calls). Larger chunks preserve more context per chunk but produce
noisier, less-precise embeddings and risk retrieving irrelevant text
alongside the relevant part. Chunk size is typically tuned based on the
nature of the content (dense technical text vs. conversational text) and
validated through evaluation.

### 17. What is `Settings` in LlamaIndex, and why use it instead of passing config everywhere?
`Settings` is a global configuration object where you set defaults once
(e.g. which embedding model, which LLM) rather than passing those arguments
into every single function call throughout your code. This centralizes
configuration and reduces repetition — set it once at the top of your
script, and every subsequent LlamaIndex operation uses those defaults
automatically.

### 18. Why can persisting and reloading an index matter for a real application?
Building an index requires embedding every chunk — an API call (and cost)
per chunk or batch. Persisting the index to disk (`persist(persist_dir=...)`)
and reloading it later (`load_index_from_storage`) avoids re-embedding
everything each time the application starts, which is both faster and
cheaper. This is the difference between a one-time ingestion step and a
fast-starting production application.

### 19. What causes a RAG system to retrieve irrelevant chunks, and how would you debug it?
Common causes: chunk size too large or too small for the content, poor
embedding model fit for the domain, ambiguous or under-specified queries,
or documents whose relevant content is scattered rather than localized.
Debugging approach: inspect what chunks were actually retrieved for a given
query (most frameworks expose retrieved source nodes), check if they're
plausible near-misses or completely unrelated, and iterate on chunk size,
embedding model choice, or query rephrasing/transformation accordingly.

### 20. Why use a separate provider for embeddings versus generation (e.g. OpenAI + Anthropic)?
Not all LLM providers offer an embeddings API — Anthropic, for example,
provides chat/generation models but no embedding model of its own,
recommending Voyage AI as a partner instead. Embeddings and generation are
functionally independent steps in a RAG pipeline, so there's no requirement
they come from the same vendor — you can mix providers based on cost,
quality, or availability for each specific step.

---

## Hard

### 21. How would you evaluate a RAG system's quality, beyond "does the answer look right"?
Systematically, using metrics like: **retrieval precision/recall** (did the
system retrieve the chunks that actually contain the answer, out of all
relevant chunks available), **faithfulness** (does the generated answer
actually follow from the retrieved context, or does it introduce
unsupported claims), and **answer relevance** (does the answer actually
address the question asked, independent of whether it's grounded).
Frameworks like RAGAS formalize these into computable metrics using an LLM
as a judge against a labeled evaluation set of question/expected-answer/
expected-source triples.

### 22. What is RAG Fusion, and when would it be worth the added complexity?
RAG Fusion generates multiple reformulations of the original query (e.g.
via an LLM), retrieves chunks for each reformulation independently, then
merges/re-ranks the combined results (often via reciprocal rank fusion).
It's worth the complexity when a single query phrasing risks missing
relevant chunks due to vocabulary mismatch between the question and the
source documents, or when questions are ambiguous enough that considering
multiple interpretations improves recall. For small, well-defined document
sets (like a handful of research papers), the added latency and complexity
usually isn't justified — the benefit shows up more at scale or with vaguer
queries.

### 23. What is hybrid search, and why combine keyword and semantic search?
Hybrid search combines dense retrieval (embedding-based semantic similarity)
with sparse retrieval (traditional keyword/BM25-style matching), then
merges the results. Semantic search excels at conceptual matches even
without shared vocabulary, but can miss exact-match cases that matter (e.g.
a specific error code, product SKU, or proper noun) where keyword search is
more reliable. Combining both catches cases either method alone would miss.

### 24. How would you handle document updates in a production RAG system serving tens of thousands of documents across multiple platforms?
Manual, scheduled re-indexing doesn't scale at that volume. A production
approach uses event-driven triggers (webhooks or file-watchers on each
source platform) that re-chunk and re-embed only the specific document that
changed, rather than rebuilding the entire index. Metadata such as
last-updated timestamps and version tags let retrieval prioritize current
content, and old/stale chunk vectors need to be explicitly deleted from the
vector store — not just left to accumulate alongside newer versions, which
would risk the LLM retrieving and blending outdated information with
current information.

### 25. Why might raw BERT be a poor choice for RAG embeddings, even though sentence-embedding models are BERT-based under the hood?
BERT was originally trained for masked-token prediction and next-sentence
prediction, not for producing a single, meaningful vector representing an
entire sentence or passage. Naively pooling BERT's token-level output
(e.g. averaging) produces embeddings that don't reliably place similar
sentences close together in vector space. Sentence-Transformer models
(e.g. `all-MiniLM-L6-v2`) are BERT-family architectures specifically
fine-tuned (often using contrastive learning on sentence-pair similarity
tasks) to produce embeddings well-suited for retrieval — the architecture
family is similar, but the training objective is what actually makes an
embedding model good at this task.

### 26. Two chunks contain almost the same information, but one is up to date and one is outdated. How could you prevent the LLM from being confused by both?
Options, from simplest to most robust: (a) ensure the ingestion pipeline
deletes outdated chunk vectors when a document is updated, rather than only
adding new ones; (b) attach metadata (e.g. `effective_date` or
`is_current`) to each chunk and filter retrieval to only consider current
chunks; (c) at generation time, explicitly instruct the LLM to prefer the
most recent source if conflicting information appears in retrieved context,
and to flag the conflict rather than silently picking one. The most robust
production systems handle this primarily at the ingestion/indexing layer
(preventing stale data from coexisting at all) rather than relying on the
LLM to sort it out at generation time.

### 27. What's a realistic failure mode of RAG in a legal/HR/compliance context, and how would you mitigate it beyond "just add RAG"?
Even with RAG, the LLM can misread retrieved context, blend it with
outdated internal knowledge, or misinterpret an ambiguous clause —
grounding reduces hallucination risk but doesn't eliminate it. For
high-stakes domains, mitigations include: requiring the system to cite the
exact source paragraph for every claim (so a human can verify), setting
confidence thresholds below which the system defers to a human rather than
answering, maintaining strict document versioning so retrieval can't
surface superseded policy text, and adding a mandatory human review step
before any legal/HR-related answer is treated as final or acted upon.

### 28. In a system indexing hundreds of thousands of chunks, why might a naive vector similarity search become a bottleneck, and how is this typically addressed?
Exact nearest-neighbor search (comparing a query vector against every
stored vector) scales linearly with the number of vectors, which becomes
too slow at large scale. Production vector databases instead use
Approximate Nearest Neighbor (ANN) algorithms (e.g. HNSW, IVF) that trade a
small amount of retrieval accuracy for large gains in search speed,
allowing sub-second retrieval even across millions of vectors.

### 29. Why might embedding batch size and rate limiting matter architecturally, not just as an inconvenience during development?
API providers enforce rate limits (requests-per-minute, tokens-per-minute)
that vary by account tier — a production ingestion pipeline processing
large document sets needs to handle this as a first-class design concern:
batching requests appropriately, implementing backoff/retry logic, and
potentially queuing/throttling ingestion jobs so they don't fail
unpredictably under load. Treating this only as a local development
annoyance risks a pipeline that silently fails or drops documents in
production when volume increases.

### 30. How would you decide between "prompted" RAG (write a general prompt, retrieve, generate) versus fine-tuning a smaller model for a specific retrieval-augmented task?
Prompted RAG is faster to build, requires no training infrastructure, and
adapts immediately to new documents or instructions — a strong default for
most use cases. Fine-tuning becomes worth considering when: the task is
narrow and repetitive enough that a smaller, cheaper model can match a
larger general model's performance once specialized; latency or cost at
scale make a large general-purpose model impractical; or prompted
performance plateaus below what's needed despite prompt iteration. A
common evaluation approach is to benchmark prompted performance first,
then measure whether fine-tuning provides a meaningful lift that justifies
the added training/maintenance complexity — rather than assuming
fine-tuning is superior by default.

---

## Hybrid Search — Detailed

### 31. What exactly is BM25, and how is it different from embedding-based search?
BM25 (Best Matching 25) is a classic keyword-scoring algorithm — the same
family of technique search engines used before embeddings existed. It
scores a document's relevance to a query based on term frequency (how
often query words appear), inverse document frequency (how rare those
words are across the whole corpus — common words like "the" contribute
almost nothing, rare words like "GELU" contribute a lot), and document
length normalization. Critically, BM25 has **no understanding of meaning**
— it only matches literal tokens. Embedding-based search is the opposite:
it captures semantic similarity even with zero shared vocabulary, but can
occasionally under-weight an exact, rare, specific term that really matters
for the answer.

### 32. Why combine BM25 and vector search instead of just using one?
Because they fail in different, complementary ways. A question like "what
is GELU?" is a case where exact term matching (BM25) is highly reliable —
the answer is almost certainly in a chunk containing the literal string
"GELU." A question like "why does the model struggle with long-range
dependencies?" has no single exact keyword to search for — it requires
semantic/conceptual matching (vector search) to find chunks discussing that
idea, even if phrased differently. Running both and merging results catches
cases either method alone would miss.

### 33. How does Reciprocal Rank Fusion actually combine two ranked lists?
Each retriever produces its own ranked list of chunks for a query. RRF
assigns each chunk a score of `1 / (rank + k)` from *each* list it appears
in (where `k` is a small constant, often 60, that dampens the impact of
very top ranks dominating too heavily), then sums those scores across all
lists a chunk appeared in. A chunk that ranks reasonably well in *both*
the vector and BM25 lists ends up with a higher combined score than a
chunk that ranked #1 in only one list and didn't appear in the other at
all — the method rewards agreement between independent retrieval signals,
not just a single method's top pick.

### 34. In this project's implementation, why is BM25 rebuilt from the source PDFs each time the app starts, rather than persisted like the vector index?
BM25 doesn't require any embedding computation — its score comes purely
from token statistics over the raw text of the nodes. Rebuilding it from
the source documents at startup is fast and free (no API calls), so
there's no real benefit to persisting it separately the way the vector
index (which *did* cost API calls to build) is persisted. This is a
reasonable tradeoff for a small, portfolio-scale document set; at much
larger scale, persisting the BM25 index too would avoid repeated
recomputation on every restart.

---

## Conversational Memory — Detailed

### 35. Why doesn't simply "remembering" the conversation history solve follow-up questions on its own?
Storing prior turns doesn't automatically fix retrieval, because retrieval
still only searches based on whatever text gets embedded for the *current*
turn. If a follow-up question like "how does it compare to full
fine-tuning?" is embedded as-is, the ambiguous pronoun "it" produces a
vague, weakly-specified query vector — retrieval will return *something*
(similarity search doesn't have a built-in "not confident enough" refusal),
but that something may not be the right chunks, since the query itself
doesn't specify what "it" refers to.

### 36. What does "condense_plus_context" mode actually do, step by step?
Two distinct steps run for every turn after the first: **(1) Condense** —
before any retrieval happens, the system sends the current question plus
the conversation history to the LLM, asking it to rewrite the question into
a fully standalone version (e.g. "how does it compare to full fine-tuning?"
becomes "how does LoRA compare to full fine-tuning?", using the earlier
turn about LoRA to resolve "it"). **(2) Context** — that rewritten,
standalone question is what actually gets embedded and used for retrieval,
so the search runs against a clear, well-specified query instead of an
ambiguous one. The chat history is also still available to the final
generation step, so tone and earlier detail can carry through naturally.

### 37. Real example encountered in this project: what happened when a follow-up question was asked without memory in place, and why?
Before memory was added, asking "What is LoRA?" followed by "What does it
do?" in a fresh, memoryless query produced an answer comparing LoRA to
full fine-tuning (storage efficiency, training throughput numbers) —
technically grounded in retrieved content, but not actually responsive to
the literal question asked. What happened: the vague question "What does
it do?" was embedded on its own, with no prior turn to resolve "it."
Retrieval still returned its closest matches regardless — in this case,
they happened to be from a chunk discussing LoRA-vs-fine-tuning tradeoffs,
likely because generic action-oriented wording in the question weakly
aligned with that content. The LLM then correctly generated an answer
*from that retrieved context* — the generation step worked exactly as
designed — but the retrieval step had no way to know what "it" should
have meant, since there was no memory of the prior "What is LoRA?" turn to
condense against. This is a clean illustration of why grounding alone
doesn't guarantee a *relevant* answer — retrieval quality depends on the
query itself being well-specified, which memory-aware condensing directly
addresses.

### 38. Why use `CondensePlusContextChatEngine` specifically, rather than just concatenating the full conversation history into the retrieval query?
Dumping the entire conversation history into the retrieval query would
pollute the search with irrelevant earlier text — old questions and
answers on unrelated subtopics would compete for similarity weight
alongside the actual current question, likely diluting retrieval
precision. Condensing first produces one clean, standalone query
specifically representing the current information need, which is a much
sharper input for similarity search than a long, mixed-topic history
blob.

### 39. Does adding hybrid search and memory change how the project should be evaluated?
Yes — retrieval evaluation (precision/recall against a labeled test set)
should now be run against the fusion retriever's combined output, not just
the vector retriever alone, since the actual system serving answers uses
both signals together. For memory specifically, evaluation should also
include multi-turn test cases (a first question, then a deliberately
ambiguous follow-up) to confirm the condense step correctly resolves
references — a single-turn evaluation set alone wouldn't catch a
regression in follow-up handling.

### 40. What's a limitation that still remains even after adding hybrid search and memory?
Neither addition changes the underlying document set's freshness or
coverage — if a question falls entirely outside what's indexed, hybrid
search and memory won't manufacture an answer from nothing (and
shouldn't). Memory also only persists within a single running session in
this implementation — restarting the app clears conversation history,
since it isn't being saved to any persistent store. A production version
serving multiple users would also need per-user session isolation, so one
user's conversation history doesn't leak into another's context.
