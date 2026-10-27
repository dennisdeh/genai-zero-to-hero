## Retrieval Augmented Generation Systems

In this chapter we look at various ways to build Retrieval Augmented Generation (RAG) systems.

We will explore the different options available for building RAG systems, including local deployment, cloud-based services, 
and containerised solutions. 
We will also discuss the trade-offs and considerations when choosing a deployment strategy for RAG systems.

An LLM only knows what was in its training data: nothing about events after its knowledge cut-off and nothing
about your organisation's internal documents. **Retrieval Augmented Generation** (RAG, Lewis et al. 2020) fixes
this without retraining: for every question, the relevant passages are *retrieved* from a document collection
and inserted into the prompt, and the LLM *generates* an answer grounded in these passages. Compared with
fine-tuning, RAG is cheap to update (re-index a document instead of retraining), can cite its sources, and keeps
access control in the retrieval layer. Its quality, however, is bounded by the quality of the retrieval: the
model cannot use a passage that was not retrieved.

The chapter contains two RAG systems built with LangChain, a local **Ollama** server (LLM and embedding model)
and the **Qdrant** vector database, both started with the chapter's `docker-compose.yml`: a minimal example with
three hard-coded documents (`s00_simple_rag_system`) and a RAG system over a folder of PDF and Word documents
from the Swiss financial-markets regulator FINMA (`s01_finma_rag_system`). The same FINMA collection is reused by
the RAG agents in part p08.

### The RAG pipeline

A RAG system has an offline **indexing** pipeline and an online **query** pipeline:

1. **Load** documents (PDF, Word, HTML, wiki pages, tickets, database rows) and extract their text together with
   **metadata** (source file, page, title, date, access rights).
2. **Chunk** the text into passages small enough to be specific and to fit several into the prompt.
3. **Embed** every chunk into a vector with an embedding model.
4. **Store** vectors, text and metadata in a vector database.
5. At query time **embed the question** with the *same* embedding model, **retrieve** the $k$ most similar
   chunks (optionally filtered by metadata), optionally **rerank** them,
6. **construct the prompt** from instructions, retrieved context and question, and **generate** the answer,
   ideally with **citations** of the sources.

#### Loading

Text extraction is often the least glamorous and most underestimated step. PDFs store positioned glyphs, not
paragraphs: multi-column layouts, headers and footers, tables, footnotes and hyphenation all end up in the
extracted text, and scanned PDFs need OCR. Libraries range from simple text extraction (`pypdf`) to
layout-aware parsers (e.g. `unstructured`, Docling) that recognise headings and tables. Keeping the page number
and file name as metadata is what later makes citations possible.

#### Chunking

Chunks that are too large dilute the embedding (one vector has to represent several topics) and waste context;
chunks that are too small lose the context needed to understand them ("it increased by 3%"). Common strategies:

- **fixed-size** chunks of $n$ characters or tokens,
- **recursive character splitting**: split on paragraphs first, then lines, then sentences, then words, until
  every piece is below the size limit - which keeps natural units together as far as possible (LangChain's
  `RecursiveCharacterTextSplitter`, used in this chapter),
- **structure-aware** splitting along the document's headings, sections, Markdown or HTML structure, or code
  syntax,
- **semantic chunking**, which starts a new chunk where the embedding similarity between consecutive sentences
  drops.

A **chunk overlap** (typically 10-20% of the chunk size) repeats the end of one chunk at the beginning of the next,
so that a sentence cut at a boundary is still complete in one of the chunks. Useful refinements are to prepend
the document title and section heading to every chunk, or to embed small chunks but return the larger parent
section to the LLM ("small-to-big" or parent-document retrieval). Typical starting values are 500-1,000
characters (roughly 100-250 tokens) with 10-20% overlap; the best values depend on the documents and questions
and should be determined by evaluation.

#### Embeddings and similarity

An **embedding model** maps a text to a dense vector (a few hundred to a few thousand dimensions) such that
semantically similar texts are close. Modern embedding models are transformer encoders (or decoder LLMs adapted
for this purpose) trained with contrastive objectives on large sets of (query, relevant passage) pairs (e.g.
Sentence-BERT, E5, BGE, the Qwen3 embedding models used here). Retrieval quality differs between models and
languages; the MTEB leaderboard compares them across tasks. Important: queries and documents must be embedded
with the same model, and changing the model means re-indexing everything.

Similarity between a query vector $q$ and a document vector $d$ is measured by

- **cosine similarity** $\cos(q, d) = \frac{q \cdot d}{\lVert q\rVert \, \lVert d\rVert} \in [-1, 1]$ (the default in
  this chapter: Qdrant collections created by LangChain use the cosine distance, and the returned *score* is the
  similarity, higher is better),
- the **dot product** $q \cdot d$ (equal to the cosine similarity for normalised vectors, which many embedding
  models produce), or
- the **Euclidean distance** $\lVert q - d\rVert$ (lower is better).

Absolute similarity values depend on the model, so a **score threshold** (e.g. "only use chunks with cosine
similarity above 0.5") has to be calibrated per embedding model.

#### Vector databases and Qdrant

With millions of chunks, comparing the query with every vector (exact $k$-nearest-neighbour search) becomes too
slow. Vector databases use **approximate nearest-neighbour (ANN)** indexes, most commonly **HNSW** (hierarchical
navigable small-world graphs), which find almost all true nearest neighbours in logarithmic time. Besides the
index, a vector database stores the **payload** (text and metadata), supports **metadata filtering** (e.g. only
documents of a certain type, date or access group), persistence, updates and deletes.

**Qdrant** is an open-source vector database written in Rust with a REST and gRPC API, payload filtering, hybrid
dense/sparse search and quantisation. It organises points (id, vector, payload) in **collections**; each
collection has a fixed vector size and distance metric. In this chapter it runs in a Docker container with its
data in the volume `qdrant_data`; its web UI is available at `http://localhost:<QDRANT_PORT_HOST>/dashboard`.
Alternatives include Chroma and FAISS (libraries for local use), pgvector (vectors in PostgreSQL), Milvus,
Weaviate, Elasticsearch/OpenSearch and managed services such as Pinecone.

#### Retrieval, prompt construction and citations

The number of retrieved chunks $k$ trades **recall** (is the relevant passage among the retrieved?) against
**precision** and prompt length (irrelevant chunks distract the model and cost tokens; with long contexts,
information in the middle is used less reliably, see chapter c05). Typical values are $k = 3$ to $10$, possibly
combined with a score threshold.

The prompt combines a system instruction ("answer using ONLY the provided context; if the answer is not in the
context, say you don't know"), the retrieved chunks and the question. Putting the chunks into the prompt is called
the **stuff** strategy; for many chunks, map-reduce (answer per chunk, then combine) or refine strategies exist.
Grounding instructions reduce, but do not eliminate, hallucinations. **Citations** make answers verifiable:
include source and page in the context passed to the model and ask it to cite them, or - more reliably - return
the metadata of the retrieved chunks alongside the answer, as the scripts below do.

#### Improving retrieval: reranking, hybrid search and query rewriting

- **Reranking**: a fast bi-encoder retrieves, say, 50 candidates, and a slower but more accurate **cross-encoder**
  (which reads query and passage together, e.g. a BGE or Cohere reranker) or an LLM re-scores them; the best 5
  are passed to the generator.
- **Hybrid search** combines dense vector search with lexical search (**BM25** or learned sparse vectors such as
  SPLADE). Dense embeddings capture meaning and paraphrases; lexical search is better at exact terms - product
  codes, article numbers, names, abbreviations, which are frequent in regulatory texts. The two result lists are
  merged, e.g. by **reciprocal rank fusion** $\mathrm{RRF}(d) = \sum_i 1/(c + \mathrm{rank}_i(d))$ with $c = 60$.
- **Query rewriting**: user questions are often short, ambiguous or depend on the conversation ("and what about
  2024?"). An LLM can rewrite the question into a standalone search query, generate several query variants
  (multi-query retrieval), decompose it into sub-questions, or write a hypothetical answer whose embedding is used
  for the search (HyDE, Gao et al. 2022). The supervisor agent with human-in-the-loop in p08 uses an LLM query
  rewriter.
- **Metadata filters** and **routing** restrict the search to the right part of the collection (e.g. only
  circulars in force, only documents the user may see).
- **Agentic RAG** lets an agent decide when and what to retrieve, possibly several times (p08).

#### Evaluation

Evaluate retrieval and generation separately, on a set of realistic questions with known relevant passages and
reference answers (built by experts, from logs, or generated synthetically with an LLM and then reviewed):

- **retrieval**: hit rate / recall@k (is a relevant chunk among the top $k$?), precision@k, mean reciprocal rank
  (MRR) and nDCG for the ranking;
- **generation**: *faithfulness* or groundedness (is every claim supported by the retrieved context?), *answer
  relevance* (does it answer the question?), correctness against the reference answer, and citation accuracy.
  These are usually judged by an LLM with a rubric (LLM-as-a-judge) and spot-checked by humans (chapter c07);
  frameworks such as Ragas implement such metrics.

Track these metrics whenever the chunking, embedding model, $k$, prompt or LLM changes.

#### Failure modes

- the answer is not in the collection, but the model answers anyway from its own knowledge (hallucination);
- the relevant chunk exists but is not retrieved (bad chunking, vocabulary mismatch, embedding model weak for the
  language or domain, $k$ too small, threshold too high);
- the right chunk is retrieved but lost in a long context or contradicted by an outdated document (versioning!);
- extraction errors: tables, multi-column layouts and scanned pages turn into garbled text;
- conflicting sources, or questions that require aggregating over many documents ("how many circulars mention
  ...") - top-$k$ retrieval is the wrong tool for these;
- stale indexes after documents change, and duplicate chunks after repeated indexing;
- prompt injection through retrieved documents, and leakage of documents a user is not allowed to see.

### Section s00_simple_rag_system

`main_simple_rag.py` shows the complete pipeline in about 70 lines:

1. **Configuration**: it loads the chapter's `.env` file (path relative to the repository root, so run from
   there) and builds the URLs `http://localhost:<OLLAMA_PORT_HOST>` and `http://localhost:<QDRANT_PORT_HOST>`.
2. **Models**: the chat model `qwen3:8b` via `get_llm` (chapter c01) and the embedding model `qwen3-embedding` via
   `OllamaEmbeddings`, both served by the chapter's Ollama container.
3. **Documents**: three short LangChain `Document` objects with text from LangChain's website - small enough that
   no chunking is needed.
4. **Indexing**: `QdrantVectorStore.from_documents` embeds the documents and uploads them to the collection `docs`
   (created on first use with the embedding size and cosine distance). Note that every run adds the three
   documents again, so after several runs the collection contains duplicates; pass `force_recreate=True` or check
   whether the collection exists (as `s01` does) to avoid this.
5. **Chain**: `qdrant.as_retriever(search_kwargs={"k": 3})` turns the store into a retriever,
   `create_stuff_documents_chain` formats the retrieved documents into the `{context}` placeholder of a chat prompt
   that instructs the model to answer only from the context, and `create_retrieval_chain` wires retriever and
   document chain together (both helpers come from the `langchain-classic` package).
6. **Query**: `rag_chain.invoke({"input": query})` returns a dictionary with the `answer` and the retrieved
   `context` documents, which are printed as sources.

### Section s01_finma_rag_system

`main_rag_local_files.py` applies the same pattern to real documents: circulars, guidance and reports of FINMA
(https://www.finma.ch/en/documents/).

**The documents.** The folder `s01_finma_rag_system/documents/` is *not* part of the repository (it is
gitignored); create it and put the PDF (`.pdf`) and Word (`.docx`) files you want to query there, e.g. FINMA
circulars and the FINMA Risk Monitor downloaded from the FINMA website. Sub-folders are searched recursively.
Any other collection of PDF/Word documents works as well - adapt the questions and the system prompt.

**Loading** (`utils/document_loaders.py`): `load_documents_from_folder` uses two LangChain `DirectoryLoader`s,
one with `PyPDFLoader` (`mode="page"`: one `Document` per page with `source` and `page` metadata) for
`**/*.pdf` and one with `Docx2txtLoader` for `**/*.docx`, loads them with multithreading and a progress bar, and
raises a clear error if the folder is missing or empty.

**The script**:

1. sets up the LLM (`qwen3:8b`), the embeddings (`qwen3-embedding`) and a `QdrantClient`;
2. **indexes only once**: if the collection `finma_docs` does not exist, the documents are loaded and split with
   `RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)` (measured in characters) and uploaded with
   `QdrantVectorStore.from_documents`; otherwise the existing collection is opened with
   `from_existing_collection`. To re-index after adding documents, delete the collection (e.g. in the Qdrant
   dashboard or with `qdrant_client.delete_collection("finma_docs")`);
3. demonstrates raw retrieval: `similarity_search_with_score` returns the 4 chunks most similar to a question
   about the FINMA Risk Monitor together with their cosine similarity, and prints the best one;
4. builds the RAG chain as in `s00`, but with a retriever that only returns chunks with a similarity of at least
   `score_threshold = 0.5` (up to the default $k = 4$), and a domain-specific system prompt (answer about FINMA
   regulation, only from the context, always quote the source, otherwise say you don't know);
5. asks "What does FINMA say about credit risk developments?" and prints the answer followed by the source file
   and page of every retrieved chunk - simple, reliable citations from the metadata.

Possible extensions, following the techniques above: hybrid search (Qdrant supports sparse vectors), a reranker,
metadata such as document type and publication date for filtering, query rewriting, and an evaluation set of
questions about the circulars with expected source pages to measure recall@k.
