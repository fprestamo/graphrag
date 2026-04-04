# Bitemporal GraphRAG: Temporal Origin Tracking and Conflict Resolution for Continuously Evolving Knowledge Graphs

---

**University of Havana**

**Faculty of Mathematics and Computer Science**

---

**Author:** [Author Name]

**Advisors:** [Advisor 1], [Advisor 2]

**Thesis presented in partial fulfillment of the requirements for the degree of [Academic Degree]**

**[Date]**

---

## Epigraph

*[Quote]*

---

## Acknowledgments

[Acknowledgments text]

---

## Tutor's Report

[Tutor's opinion text]

---

## Resumen

Los sistemas de Generacion Aumentada por Recuperacion (RAG) basados en grafos de conocimiento han demostrado ser una estrategia efectiva para responder preguntas complejas sobre grandes corpus de documentos. Sin embargo, estos sistemas operan sobre una premisa que se rompe silenciosamente en la practica: que el conocimiento que indexan es estatico. El mundo real produce documentos de forma continua y esos documentos no solo anaden hechos nuevos — modifican, contradicen y reemplazan hechos anteriores.

Esta tesis parte de una pregunta mas profunda que la literatura existente: que ocurre cuando un sistema de recuperacion no sabe cuando comenzo a ser verdad cada hecho que almacena? GraphRAG de Microsoft, el framework mas influyente en este dominio, construye su grafo de conocimiento ignorando completamente la dimension temporal de los hechos que extrae. Cada relacion que ingresa al grafo es tratada como si fuera eternamente verdadera desde el momento en que fue indexada. No existe ningun mecanismo para registrar el instante en que un hecho se vuelve verdadero, el instante en que deja de serlo, ni el instante en que el sistema tomo conocimiento de el. Esta **Ceguera Temporal Originaria (TOB)** es el problema raiz del que se derivan todos los demas fallos de consistencia.

La benchmark CRAG (Comprehensive RAG Benchmark, 2024) confirma empiricamente este diagnostico: la precision de los sistemas RAG sobre entidades de rapida evolucion es significativamente inferior a la de entidades estaticas, y los sistemas RAG directos aumentan la exactitud pero frecuentemente reducen la "veracidad" al incorporar hechos desactualizados. Encuestas exhaustivas del campo del Temporal Knowledge Graph (TKG) senalan que la escalabilidad, la gestion de actualizaciones incrementales y la correccion retroactiva son desafios abiertos fundamentales.

Esta tesis propone **BT-GraphRAG**, una extension de GraphRAG que resuelve la ceguera temporal originaria mediante cinco mecanismos integrados: (1) normalizacion temporal de expresiones en lenguaje natural a intervalos ISO 8601 durante la extraccion, (2) resolucion cross-batch de entidades mediante senales multiples para prevenir la fragmentacion en nodos alias, (3) deteccion y resolucion de conflictos factuales en el momento de la ingestion, identificando cuando un hecho entrante contradice o invalida hechos existentes en el grafo, (4) un modelo bitemporal que registra tanto cuando un hecho es verdadero en el mundo como cuando el sistema tomo conocimiento de el, y (5) un modulo de consistencia retroactiva para documentos con llegada tardia.

**Palabras clave:** Grafo de Conocimiento, Generacion Aumentada por Recuperacion, Razonamiento Temporal, Modelo Bitemporal, Resolucion de Entidades, Deteccion de Conflictos, Ingestion en Streaming, Consistencia de Grafos

---

## Abstract

Knowledge Graph-based Retrieval-Augmented Generation (RAG) systems have proven to be an effective strategy for answering complex questions over large document corpora. However, these systems operate on a premise that breaks silently in practice: that the knowledge they index is static. The real world produces documents continuously, and those documents do not merely add new facts — they modify, contradict, and replace prior ones.

This thesis departs from a deeper question than the existing literature: what happens when a retrieval system does not know *when* each fact it stores first became true? Microsoft's GraphRAG, the most influential framework in this domain, builds its knowledge graph while entirely ignoring the temporal dimension of the facts it extracts. Every relationship that enters the graph is treated as if it were eternally true from the moment it was indexed. There is no mechanism to record the instant a fact becomes true, the instant it ceases to be so, nor the instant the system became aware of it. This is **Temporal Origin Blindness (TOB)** — the root problem from which all other consistency failures derive.

The CRAG benchmark (Comprehensive RAG Benchmark, 2024) empirically confirms this diagnosis: the precision of RAG systems on rapidly evolving entities is significantly lower than on static ones, and direct RAG systems increase accuracy but frequently reduce "faithfulness" by incorporating outdated facts. Comprehensive surveys of the Temporal Knowledge Graph (TKG) field point to scalability, incremental update management, and retroactive correction as fundamental open challenges.

This thesis proposes **BT-GraphRAG**, an extension of GraphRAG that resolves Temporal Origin Blindness (TOB) through five integrated mechanisms: (1) temporal normalization of natural-language expressions to ISO 8601 intervals during extraction, (2) cross-batch entity resolution via multiple signals to prevent fragmentation into alias nodes, (3) conflict detection and resolution at ingestion time that identifies when incoming facts contradict or invalidate existing ones in the graph, (4) a bitemporal model that records both when a fact is true in the world and when the system became aware of it, and (5) a retroactive consistency module for late-arriving documents.

**Keywords:** Knowledge Graph, Retrieval-Augmented Generation, Temporal Reasoning, Bitemporal Model, Entity Resolution, Conflict Detection, Streaming Ingestion, Graph Consistency

---

## Table of Contents

- Preliminary Pages
- Introduction
- Chapter 1: State of the Art and Theoretical-Conceptual Framework
  - 1.1 Foundations of Knowledge-Graph RAG Systems
    - 1.1.1 From Knowledge Graphs to GraphRAG
    - 1.1.2 Structure of GraphRAG and the Leiden Community Approach
  - 1.2 Temporal Knowledge Graph Foundations
    - 1.2.1 Temporal Knowledge Graph Representation
    - 1.2.2 Bitemporal Data Models and SQL:2011
    - 1.2.3 Streaming and Event-Time Architectures
  - 1.3 Temporal Extensions to Knowledge-Graph RAG
    - 1.3.1 Valid-Time Temporal RAG Systems
    - 1.3.2 Bitemporal and Conflict-Aware Systems
  - 1.4 Graph Maintenance, Knowledge Editing, and Entity Resolution
    - 1.4.1 Graph Unlearning
    - 1.4.2 Knowledge Editing and Conflict Resolution
    - 1.4.3 Entity Resolution in Evolving Knowledge Graphs
  - 1.5 Evaluation Benchmarks
    - 1.5.1 The CRAG Benchmark
    - 1.5.2 Temporal QA Benchmarks
    - 1.5.3 The Temporal Graph Benchmark (TGB)
  - 1.6 State of the Art Analysis
    - 1.6.1 What Temporal Awareness in a Knowledge Graph Actually Requires
    - 1.6.2 The Failure Modes of Temporal Origin Blindness
    - 1.6.3 Research Gap and Positioning
  - 1.7 Conclusions
- Chapter 2: Proposed Solution
  - 2.1 Baselines
    - 2.1.1 Direct GraphRAG Baselines
    - 2.1.2 Existing Temporal RAG Systems as Baselines
  - 2.2 Design Principles
  - 2.3 System Architecture Overview
  - 2.4 Stage 1: Temporal Expression Normalization
  - 2.5 Stage 2: Cross-Graph Entity Resolution (CGER)
  - 2.6 Stage 3: Edge-Level Temporal Conflict Detection and Resolution (ETCDR)
    - 2.6.1 Four-Way Relation Cardinality Ontology
    - 2.6.2 Bidirectional Conflict Detection
    - 2.6.3 The Decision Router
    - 2.6.4 The Late-Arrival Path
  - 2.7 Stage 4: Bitemporal Graph Store
  - 2.8 Stages 5–6: Incremental Community Update and Selective Summarization
  - 2.9 Query Pipeline
  - 2.10 Knowledge Maintenance Module
- Chapter 3: Implementation
- Chapter 4: Experimental Evaluation
- Conclusions
- Recommendations and Future Work
- Bibliography
- Appendices

## List of Figures

[To be completed]

## List of Tables

[To be completed]

---

## Introduction

### Motivation and Background

Knowledge-graph-based Retrieval-Augmented Generation (RAG) systems work by building a structured representation of a document corpus and then querying that representation to answer user questions. The quality of the answers depends entirely on one thing: whether the knowledge stored in the graph accurately reflects reality at the moment of the query. Microsoft's GraphRAG (Edge et al., 2024), the most influential knowledge-graph RAG framework to date, builds its graph by extracting entities and relationships from text, organizes them into a hierarchical community structure using the Leiden algorithm, and summarizes each community with an LLM. The result is a highly effective system for querying a static corpus.

However, there is an assumption buried inside every knowledge graph that is so obvious it is almost never stated: every fact stored in the graph was true at some point in time. A system that never records *which* point in time — that treats every extracted relationship as if it were simply and permanently true, floating free of any temporal anchor — has no way to distinguish what is currently true from what used to be true, or to know what to do when new information arrives that contradicts what is already stored.

GraphRAG indexes knowledge with no record of *when* any fact began to be true. There is no temporal origin attached to any edge. When documents arrive in a stream — as they do in any real-world deployment that adds new papers, reports, or news articles over time — GraphRAG's behavior on the new documents is simply to add what they say to whatever is already in the graph. There is no check whether a newly extracted fact contradicts an existing one. There is no mechanism to recognize that an incoming fact may structurally invalidate an existing one — that asserting a new exclusive relationship implicitly terminates a prior one. The result is a knowledge representation that accumulates facts indefinitely, with no mechanism to distinguish what remains true from what has been superseded.

The CRAG Benchmark (Yang et al., 2024) provides empirical confirmation of this diagnosis. Across 4,409 question-answer pairs covering 2.6 million entities, CRAG finds that straightforward RAG significantly improves accuracy on static facts but frequently *decreases* truthfulness on fast-changing entities — the system's knowledge base contains outdated facts that are confidently cited as current truth.

Comprehensive surveys of the Temporal Knowledge Graph (TKG) field identify scalability, incremental update management, and retroactive correction as fundamental open challenges. The TKG Completion survey (IJCAI 2023) explicitly lists graph unlearning and correction of erroneous facts as open problems. No existing TKGC method addresses conflict resolution during ingestion, and no existing system in the GraphRAG lineage addresses the object-side conflict detection problem — the case where a new fact about a different subject implies that an existing subject's exclusive relationship has ended.

### Scientific Problem and Problem Scope

Knowledge-graph-based RAG systems that process continuously evolving document streams present difficulties in ensuring that their stored knowledge representation is temporally consistent: accurately reflecting which facts are currently true, which were historically true, and which have been superseded by subsequent information. Microsoft's GraphRAG framework, the most influential in this domain, builds its graph by extracting entities and relationships from text without attaching any temporal origin to the facts it extracts. Every relationship enters the graph as if it were eternally valid, with no mechanism to detect when an incoming fact contradicts or invalidates existing ones, nor to retroactively correct the graph's state when documents arrive outside their true chronological order. This condition — termed **Temporal Origin Blindness (TOB)** in this thesis — produces a silent and monotonic accumulation of inconsistencies that translates into a quantifiable degradation of the system's response fidelity, formalized as Temporal Knowledge Drift: $\text{TKD}(t) = \mathcal{G}(t) \triangle \mathcal{W}(t)$.

The object of study of this research is framed within the maintenance of temporal consistency in knowledge-graph-based RAG systems operating over continuously evolving document streams.

The field of action is circumscribed to the design and implementation of BT-GraphRAG as an end-to-end temporal extension of Microsoft's GraphRAG, focused on temporally-aware cross-graph entity resolution, bidirectional detection and resolution of factual conflicts at the edge level, full bitemporal storage under the SQL:2011 standard, and retroactive graph consistency for late-arriving documents. The scope also includes the empirical evaluation of the system via the TemporalConflict benchmark using alias node rate, ghost rate, and role simultaneity rate metrics — designed specifically to capture the failure modes that existing temporal benchmarks cannot detect.

### Hypotheses

**H1:** Cross-graph entity resolution with temporal overlap signals reduces alias node rate by at least 60% compared to embedding-similarity-only matching.

### General Objective

To design, implement, and evaluate BT-GraphRAG, an end-to-end temporal extension of Microsoft's GraphRAG that resolves Temporal Origin Blindness through cross-graph entity resolution, bidirectional conflict detection with a four-way cardinality ontology, full bitemporal storage, and retroactive graph consistency, enabling knowledge-graph RAG systems to maintain temporal correctness over continuously evolving document streams.

### Specific Objectives

1. Formalize the concept of Temporal Origin Blindness (TOB) and its three systematic failure modes (entity aliasing, edge-level temporal interference, late-arrival entanglement), establishing Temporal Knowledge Drift (TKD) as the quantitative measure of the resulting graph degradation.

2. Design and implement a Cross-Graph Entity Resolution (CGER) algorithm that uses a five-signal composite scoring function with temporal overlap as its distinguishing feature, preventing alias node creation across ingestion batches without destroying temporal context.

3. Develop a four-way relation cardinality ontology (SUBJECT_EXCLUSIVE, OBJECT_EXCLUSIVE, BOTH_EXCLUSIVE, NON_EXCLUSIVE) enabling precise modeling of the structural constraints on both sides of any relationship.

4. Design and implement a bidirectional Edge-Level Temporal Conflict Detection and Resolution (ETCDR) engine that performs both subject-side and object-side conflict queries, with a Decision Router implementing four resolution strategies (Evolution, Correction, Corroboration, Disagreement) grounded in formal bitemporal semantics.

### Thesis Structure

This thesis is organized into four chapters, preceded by this introduction and followed by conclusions and recommendations.

**Chapter 1: State of the Art and Theoretical-Conceptual Framework** surveys 25 related systems across five categories — temporal extensions to knowledge-graph RAG, bitemporal and streaming architectures, temporal knowledge graph foundations, graph maintenance and knowledge editing, and evaluation benchmarks. It establishes the theoretical foundations of knowledge-graph RAG, temporal knowledge graphs, and bitemporal data models. The chapter formally identifies the four levels of temporal awareness, the three failure modes of Temporal Origin Blindness, and the precise research gap that BT-GraphRAG fills.

**Chapter 2: Proposed Solution** presents the BT-GraphRAG architecture in detail, including the five design principles, the seven-stage processing pipeline, the CGER algorithm, the ETCDR engine with its four-way cardinality ontology and bidirectional conflict detection, the bitemporal graph store, the retroactive consistency mechanism, and the temporal query pipeline.

**Chapter 3: Implementation** describes the technology stack, the Neo4j bitemporal schema design, and the implementation details of each module — CGER, ETCDR, the late-arrival processing pipeline, and incremental community detection.

**Chapter 4: Experimental Evaluation** defines the research questions, hypotheses, evaluation datasets (primary TemporalConflict benchmark and five secondary benchmarks), baselines, ablation studies, and evaluation metrics. It presents the experimental protocol for validating each hypothesis.

---

## Chapter 1: State of the Art and Theoretical-Conceptual Framework

### 1.1 Foundations of Knowledge-Graph RAG Systems

The foundations of knowledge-graph-based RAG systems lie at the intersection of knowledge representation, graph-based retrieval, and language model generation. This section introduces the core concepts and architectural components of these systems, with particular focus on Microsoft's GraphRAG framework — the baseline that BT-GraphRAG extends.

#### 1.1.1 From Knowledge Graphs to GraphRAG

Retrieval-Augmented Generation (RAG) systems address the fundamental limitation of large language models: their knowledge is frozen at training time. By retrieving relevant information from an external knowledge source at query time, RAG systems can ground their answers in up-to-date, domain-specific facts. The original RAG formulation (Lewis et al., 2020) retrieves passages from a document store using dense vector similarity. Knowledge-graph-based RAG extends this by structuring the retrieved knowledge as a graph of entities and relationships, enabling multi-hop reasoning and structured queries that passage-level retrieval cannot support.

#### 1.1.2 Structure of GraphRAG and the Leiden Community Approach

Microsoft's GraphRAG (Edge et al., 2024) is the most influential knowledge-graph RAG framework. It processes a document corpus through an extraction pipeline that identifies entities and relationships using an LLM, organizes the resulting knowledge graph into a hierarchical community structure using the Leiden algorithm, and summarizes each community with an LLM-generated report. Queries are answered through two modes: local search (entity-centric, retrieving the immediate neighborhood of relevant entities) and global search (MapReduce over community summaries). GraphRAG's innovation is the community-level summarization, which enables answering broad, corpus-spanning questions that entity-level retrieval alone cannot address.

The Leiden algorithm produces a hierarchical decomposition of the graph into communities at multiple resolution levels, enabling queries at different scales of abstraction. However, the entire pipeline assumes a static corpus: the graph is built once and never updated. There is no mechanism for incremental ingestion, conflict detection, or temporal tracking.

### 1.2 Temporal Knowledge Graph Foundations

Temporal knowledge graphs extend the standard knowledge graph formalism by associating temporal information with facts, enabling the representation and reasoning over how knowledge evolves through time. This section reviews the theoretical foundations of temporal knowledge representation, bitemporal data models, and the streaming architectures that process time-stamped data at scale.

#### 1.2.1 Temporal Knowledge Graph Representation

While a standard knowledge graph stores facts as static triples $(h, r, t)$, a temporal knowledge graph extends this representation to quadruples $(h, r, t, \tau)$, where $\tau$ anchors the relationship in time — capturing not just *what* is true but *when* it is true. This seemingly simple extension has profound consequences for how knowledge is stored, queried, and maintained: it transforms the graph from a static snapshot into a living record that must account for facts that begin, evolve, and expire. The field has consolidated around a taxonomy of 10 categories of representation methods (Chen et al., 2024), evaluated on benchmark families such as ICEWS, GDELT, and the Wikidata temporal subset using Mean Reciprocal Rank (MRR) and Hits@k as standard metrics. Despite significant progress, comprehensive surveys consistently identify scalability and real-time incremental updates as the most critical unresolved challenges (Chen et al., 2024).

From a methodological perspective, temporal knowledge graph completion (TKGC) approaches can be grouped into five main families: tensor decomposition, time-based transformation, dynamic embedding, snapshot-based learning, and temporal point process methods such as Know-Evolve (IJCAI 2023 survey). Methods that handle interval-based validity — such as HyTE, which uses hyperplane projection by time zone, and SpliME, which merges adjacent intervals and splits at change-points — offer finer temporal granularity than point-in-time approaches. However, even the most sophisticated TKGC methods treat the graph as a static artifact to be completed after the fact; no existing method addresses conflict resolution during ingestion, and the IJCAI 2023 survey explicitly lists graph unlearning and correction of erroneous facts as open problems.

Recent architectures have begun to incorporate richer temporal reasoning. DHE-TKG (2024) combines R-GCN for pairwise relations with a dynamic hypergraph encoder for high-order interactions, using GRU-based historical state fusion and temporal attention to capture evolving patterns across the ICEWS and YAGO datasets. However, like all TKGC methods, it operates with blind append-only updates and no mechanism to detect contradictions at ingestion time.

The most architecturally relevant prior system to BT-GraphRAG is EVOKG (2025), which introduces a crucial semantic distinction between *exclusive* relations (where only one value can be true at a time, such as a country's current head of state) and *non-exclusive* relations (where multiple values coexist across time, such as employment history). For exclusive facts, EVOKG computes a confidence score based on source frequency, recency, and reliability to resolve conflicts, and its companion module EVOREASONER achieves up to 23.3% improvement on the TimeQuestions benchmark through multi-route temporal query decomposition. However, EVOKG's exclusivity model is binary and does not capture the structural asymmetry between subject-side and object-side exclusivity. It detects conflicts when the same subject acquires a new value for an exclusive relation, but cannot detect that asserting "B becomes CEO of Company X" should terminate "A is CEO of Company X" — the object-side failure mode that BT-GraphRAG's four-way cardinality ontology specifically addresses.

#### 1.2.2 Bitemporal Data Models and SQL:2011

The bitemporal data model, standardized in SQL:2011, maintains two independent time axes for every stored fact: *valid time* (when the fact was true in the world) and *transaction time* (when the system recorded it). This separation enables four distinct epistemic states — a fact can be currently believed and currently true, currently believed but historically true, historically believed but currently true, or historically believed and historically true — and is essential for distinguishing genuine knowledge updates from retroactive corrections.

The formal semantic grounding for this model is provided by the TLINQ/VLINQ calculi (2022), which define two complementary query languages: TLINQ for transaction-time queries and VLINQ for valid-time queries, both with formal translations to standard LINQ for execution. The system uses closed-open intervals $[start, end)$ for validity periods and enforces well-formedness constraints during updates. A key distinction formalized by TLINQ/VLINQ — between *sequenced updates* (where a new fact extends the timeline, representing evolution) and *nonsequenced updates* (where a fact is corrected retroactively) — maps directly to the Evolution and Correction resolution strategies in BT-GraphRAG's conflict detection engine.

#### 1.2.3 Streaming and Event-Time Architectures

The distinction between *event time* (when something happened in the world) and *processing time* (when the system learned about it) is the stream-processing realization of the bitemporal valid-time/transaction-time model. Two systems demonstrate that this distinction is not merely theoretical but architecturally feasible at production scale.

Pathway (2022), a unified batch/stream engine built with a Python API over a Rust differential dataflow core, implements COMMIT-protocol semantics — stronger than eventual consistency — and supports a genuine bitemporal model. On the LiveJournal benchmark (4.8M nodes, 69M edges), it outperforms both Apache Flink and Apache Spark in latency and throughput, providing architectural proof that bitemporal processing at stream scale is production-viable. Its COMMIT semantics serve as the consistency model that BT-GraphRAG's incremental ingestion pipeline targets.

Aion (2020) addresses the specific problem of late-arriving events — data that arrives after its temporal window has closed, which standard watermark-based stream processing simply discards. Through a combination of proactive caching, predictive cleanup, and a staleness-based trigger optimized via gradient descent, Aion's dual-bucket architecture (hot in-memory state paired with cold persistent state) achieves 50% memory reduction with no accuracy loss. This late-arrival handling mechanism is the stream-processing equivalent of BT-GraphRAG's Retroactive Graph Consistency (RGC) module.

### 1.3 Temporal Extensions to Knowledge-Graph RAG

A growing body of work has attempted to extend knowledge-graph RAG systems with temporal capabilities. This section surveys the landscape of these extensions, organized by the depth of their temporal modeling: from systems that annotate facts with valid-time timestamps to systems that implement bitemporal models with conflict detection.

#### 1.3.1 Valid-Time Temporal RAG Systems

The majority of temporal extensions to knowledge-graph RAG operate exclusively at the valid-time level: they annotate facts with timestamps indicating when they were true in the world, but provide no mechanism for tracking when the system learned about them, detecting contradictions between facts, or handling documents that arrive out of chronological order. Despite this shared limitation, these systems have explored a diverse range of strategies for incorporating temporal information into the retrieval and reasoning pipeline, which can be organized into three broad approaches.

**Graph structure augmentation.** One line of work enriches the graph topology itself with temporal structure. TG-RAG (2025) constructs a bi-level architecture linking a temporal knowledge graph to a hierarchical time graph (year–quarter–month–day), using Personalized PageRank seeded from both query entities and the temporally nearest time node; on the ECT-QA corpus (480 earnings-call transcripts), it achieves Correct=0.599 versus 0.410 for the best baseline and demonstrates that incremental graph updates consume roughly 18 times fewer tokens than full re-indexing. E2RAG (2025) takes a different structural approach with a dual-graph architecture separating entity and event subgraphs connected by bipartite edges, deliberately avoiding entity deduplication — its key empirical finding that standard knowledge graph deduplication destroys temporal context directly motivates BT-GraphRAG's temporally-aware entity resolution design. DynaGRAG (2025), while contributing novel topological traversal methods (Dynamic Similarity-Aware BFS with GCN-based scoring), contains no temporal modeling whatsoever and serves primarily as evidence that topological improvements alone are insufficient for temporally evolving corpora.

**Temporal query decomposition.** A second approach focuses on breaking complex temporal questions into simpler, time-constrained sub-queries. T-GRAG (2025) introduces a five-module pipeline whose Temporal Query Decomposer splits multi-temporal questions into sub-queries with explicit time constraints, achieving 38–47% improvement over standard GraphRAG on the Time-LongQA benchmark (2,292 QA pairs from Audi annual reports 2012–2023). KG-IRAG (2025) uses a dual-LLM architecture — one as planner, the other as judge/verifier — with an iterative feedback loop to improve multi-hop temporal reasoning over a fixed-ontology knowledge graph.

**Temporal scoring and retrieval.** A third family of approaches modifies the retrieval scoring function to incorporate temporal signals. DyG-RAG (2024) introduces Dynamic Event Units carrying explicit timestamps and applies exponential decay edge weights $\exp(-\alpha \cdot \Delta t)$ with a Fourier time encoder for continuous temporal embedding, achieving 10–18% accuracy gains on TimeQA and TempReason. MRAG (2024) decomposes questions into semantic and temporal components and combines them via hybrid ranking ($\text{score} = \text{semantic\_score} \times \text{temporal\_score}$), improving answer recall by 9.3% on a 33.1-million-chunk Wikipedia corpus. STAR-RAG (2025) compresses temporal patterns into rule graphs via MDL-guided construction, achieving 97% token reduction with 9.1% accuracy improvement. DYNAMO (2025) adds a dynamic update loop where news verified through Monte Carlo Tree Search is appended to the graph, though it relies on implicit "latest-wins" conflict resolution. DALK (2024) explores co-augmentation of LLMs and knowledge graphs for Alzheimer's Disease question answering with scientific literature, employing a yearly time-slicing approach to build an evolving domain-specific knowledge graph, though its temporal modeling remains limited to publication-year granularity without explicit valid-time or transaction-time tracking. RAG4DyG (2025) applies RAG principles to dynamic link prediction using exponential temporal decay $\exp(-\lambda \cdot |t_q - t_p|)$ for weighting retrieved historical patterns alongside contrastive learning for retrieval quality; while its primary domain is link prediction rather than text QA, its temporal decay formulation directly informs BT-GraphRAG's retrieval scoring component.

Despite their individual contributions to temporal retrieval, the timestamped systems in this section — TG-RAG, E2RAG, T-GRAG, KG-IRAG, DyG-RAG, MRAG, STAR-RAG, DYNAMO, DALK, and RAG4DyG — share a fundamental limitation: none implements conflict detection or resolution at ingestion time, none tracks transaction time alongside valid time, and none provides a mechanism for handling late-arriving documents. They operate at Level 1 of the temporal awareness hierarchy defined in Section 1.6.1. DynaGRAG, which provides no temporal modeling at all, does not meet even this minimal criterion and serves in this survey solely as evidence that topological improvements alone cannot address the challenges of temporally evolving corpora.

#### 1.3.2 Bitemporal and Conflict-Aware Systems

Among all surveyed systems in the GraphRAG lineage, only ZEP/Graphiti (2025) advances beyond valid-time annotation to implement a genuine bitemporal model. Graphiti builds a three-tier graph — episodes, semantic entities, and communities — where each edge carries four timestamps: valid-time start and end ($t_{valid}$, $t_{invalid}$) and transaction-time start and end ($t'_{created}$, $t'_{expired}$). Entity resolution uses hybrid search combining BGE-m3 vector similarity with BM25 full-text matching, with LLM verification for ambiguous cases, achieving 94.8% accuracy on the DMR benchmark and an 18.5% accuracy improvement on LongMemEval with 90% latency reduction.

ZEP/Graphiti represents the most ambitious prior attempt at temporally-aware knowledge graph management within the RAG paradigm, but its conflict detection mechanism reveals a critical architectural blind spot. When a new fact arrives, ZEP checks only the *subject side*: whether the incoming fact's subject already holds the same relation. It does not perform *object-side queries* — when "B becomes CEO of Company X," ZEP does not search for existing subjects that hold an exclusive relationship with the same object, and therefore does not discover that "A was CEO of Company X" should be closed. This unidirectional conflict detection leaves the most structurally consequential class of temporal contradictions entirely unaddressed, and constitutes the central gap that BT-GraphRAG's bidirectional conflict detection engine resolves.

### 1.4 Graph Maintenance, Knowledge Editing, and Entity Resolution

Maintaining the consistency of a knowledge graph over time requires mechanisms for knowledge editing, conflict resolution, certified unlearning, and temporally-aware entity resolution. This section reviews the complementary techniques that address these challenges.

#### 1.4.1 Graph Unlearning

Regulatory requirements such as the GDPR's right to erasure demand that knowledge graph systems be capable of *certified unlearning* — provably removing the influence of specific data points without retraining the entire model from scratch. The computational challenge is significant: naive retraining is prohibitively expensive at scale, and approximate methods must preserve formal privacy guarantees.

Three complementary approaches have addressed this challenge. ScaleGUN (2024) uses Lazy Local Propagation via the Forward Push algorithm for Generalized PageRank, achieving $(\varepsilon, \delta)$-differential privacy guarantees with $O(L^2 \cdot d)$ amortized complexity; on ogbn-papers100M (111M nodes, 1.6B edges), it removes 5,000 edges in 20 seconds versus 1.91 hours for full retraining. GIF (2023) extends classical influence functions to account for structural dependencies in GNNs up to 2-hop message passing, supporting node, edge, and feature unlearning at 15–100x the speed of retraining. GraphEraser (2022) takes a partition-based approach, training independent models on graph shards so that unlearning requires retraining only the affected shard. BT-GraphRAG's privacy-compliant removal module draws on ScaleGUN's lazy local propagation to propagate unlearning signals through the community structure without full recomputation, and on GraphEraser's shard model for bulk source retraction.

#### 1.4.2 Knowledge Editing and Conflict Resolution

A related but distinct challenge is the tension between *editing* knowledge (adding or updating facts) and *unlearning* knowledge (removing facts). LOKA (2025) formalizes this conflict in the context of LLMs, demonstrating that when the data to be edited and unlearned are semantically similar, their gradient updates interfere destructively. Its Knowledge Codebook Framework resolves this by allocating task-specific memory for editing versus multi-task memory for unlearning, improving Truth Ratio from 0.42 to 0.80. This analysis of editing-unlearning gradient conflicts provides theoretical motivation for BT-GraphRAG's strict separation of its Evolution and Correction resolution strategies into independent processing paths.

#### 1.4.3 Entity Resolution in Evolving Knowledge Graphs

Entity resolution in evolving knowledge graphs presents a unique challenge: the same real-world entity may appear under different names across documents from different time periods (due to rebranding, renaming, or alias usage), and naive deduplication destroys the temporal context that distinguishes the entity's state at different points in time. E2RAG's empirical finding that "standard KG deduplication destroys temporal context" establishes that any entity resolution approach for temporal knowledge graphs must be temporally aware.

ZEP/Graphiti uses hybrid search (vector similarity + BM25 lexical matching) with LLM verification for hard cases, but does not incorporate temporal overlap as a signal. Standard entity resolution approaches based solely on embedding similarity fail when the same entity's description evolves significantly over time, or when different entities share similar names in different time periods.

BT-GraphRAG's CGER module addresses this gap by incorporating temporal overlap as a first-class signal in a five-signal composite scoring function, ensuring that entities are resolved based not only on semantic and lexical similarity but also on the temporal compatibility of their known activity periods.

### 1.5 Evaluation Benchmarks

The empirical evaluation of temporal knowledge graph systems requires benchmarks that specifically test temporal consistency, conflict detection, and late-arrival handling. This section reviews the evaluation frameworks and benchmark datasets relevant to BT-GraphRAG.

#### 1.5.1 The CRAG Benchmark

The CRAG Benchmark (Yang et al., 2024) provides the primary empirical motivation for this thesis. Across 4,409 question-answer pairs covering 2.6 million entities, CRAG evaluates RAG systems using a Truthfulness metric scored on a four-point scale (Perfect=1.0, Acceptable=0.5, Missing=0.0, Incorrect=-1.0). Its central finding is that for rapidly-changing domains such as finance and sports, RAG systems achieve notably lower scores than on stable domains, precisely because their knowledge stores cannot distinguish current from historical facts. CRAG provides direct empirical confirmation that Temporal Origin Blindness degrades retrieval quality in proportion to the dynamism of the domain.

#### 1.5.2 Temporal QA Benchmarks

The temporal QA evaluation landscape spans several complementary benchmarks, each testing a different facet of temporal reasoning. TempRAGEval (MRAG, 2024) probes temporal constraint sensitivity by perturbing the temporal anchors in TimeQA and SituatedQA questions. Time-LongQA (T-GRAG, 2025) offers 2,292 QA pairs derived from Audi annual reports spanning 2012–2023, specifically testing multi-temporal reasoning across long document histories. ChronoQA (E2RAG, 2025) provides 497 QA pairs from 9 public-domain narratives designed to test causal consistency and character identity tracking over time. ECT-QA (TG-RAG, 2025) uses 480 earnings-call transcripts to measure both retrieval accuracy and the computational cost of incremental graph updates. TimeQuestions and MultiTQ are established temporal QA benchmarks for evaluating knowledge graph reasoning, used by EVOKG (2025), among others, to validate temporal inference capabilities. Collectively, these benchmarks cover the spectrum from single-fact temporal filtering to multi-hop temporal reasoning, but none specifically tests conflict detection, late-arrival handling, or bitemporal audit — the capabilities central to BT-GraphRAG's contribution.

#### 1.5.3 The Temporal Graph Benchmark (TGB)

The Temporal Graph Benchmark (TGB, NeurIPS 2023) provides 9 curated datasets for continuous-time temporal graph learning, with temporal spans ranging from weeks to decades. Using filtered MRR for dynamic link prediction and NDCG@10 for node property prediction, TGB establishes standardized streaming evaluation protocols and demonstrates the computational requirements of temporal graph methods at scale. While TGB's focus is on link prediction rather than question answering, its streaming evaluation methodology informs the design of BT-GraphRAG's TemporalConflict benchmark.

### 1.6 State of the Art Analysis

The preceding survey of 25 related systems reveals a structured landscape of temporal capabilities and systematic gaps. This section synthesizes the findings into a formal analysis of temporal awareness levels, identifies the three failure modes of Temporal Origin Blindness, and positions BT-GraphRAG relative to the identified research gaps.

#### 1.6.1 What Temporal Awareness in a Knowledge Graph Actually Requires

The survey above reveals that existing systems address temporal awareness at four distinct levels, with most systems providing only the first one or two:

**Level 1 — Timestamps on Facts (Valid Time).** The minimal condition: every fact $(h, r, t)$ is annotated with the time at which it was true in the world. It enables temporal filtering but provides no conflict resolution, no error correction, and no system-side audit trail. The majority of systems surveyed in Section 1.3.1 operate at this level, including TG-RAG, T-GRAG, DyG-RAG, MRAG, STAR-RAG, E2RAG, KG-IRAG, DYNAMO, DALK, RAG4DyG, and DHE-TKG.

**Level 2 — Fact Lifecycle Management.** Facts are not just timestamped but managed through a lifecycle: they can be opened, closed, and corroborated. This requires an active data management layer, not just annotation. EVOKG operates at this level with subject-centric lifecycle management based on its binary exclusive/non-exclusive ontology. ZEP/Graphiti also operates at this level and additionally records transaction time — which gives it partial Level 4 capability — but without the retroactive consistency mechanism required for full Level 4 classification.

**Level 3 — Conflict-Aware Bidirectional Updates.** When a new fact arrives, the system checks not only whether the *subject* already holds the same relation (subject-side conflict) but also whether the *object* already has a subject holding an exclusive relation (object-side conflict). This requires both a four-way cardinality ontology and bidirectional query execution. No existing system in the GraphRAG lineage operates at this level. BT-GraphRAG is the first system to do so.

**Level 4 — Full Bitemporality with Retroactive Consistency.** The system maintains two independent time axes: valid time and transaction time, enabling four distinct epistemic states. Late-arriving documents can be processed against the historical state of the graph at the document's event time. TLINQ/VLINQ, Pathway, and Aion implement this at the infrastructure level. ZEP/Graphiti achieves the transaction-time axis but lacks the retroactive consistency mechanism that makes Level 4 complete. BT-GraphRAG is the first system to implement full Level 4 bitemporality at the knowledge graph level.

#### 1.6.2 The Failure Modes of Temporal Origin Blindness

From the survey, three systematic failure modes of Temporal Origin Blindness are identified. These are not edge cases — they are the structural consequences of operating without a temporal origin model.

**Failure Mode 1: Entity Aliasing.** When documents from different time periods are ingested in sequence, the same real-world entity — a company after a merger and renaming, a person after a name change, a government ministry after a reorganization — may be extracted as multiple distinct nodes. Without cross-batch entity resolution that is aware of temporal overlap in the entities' activity periods, these alias nodes persist independently. E2RAG's finding that "standard KG deduplication destroys temporal context" shows why naive deduplication is not the solution — the resolution must be temporally aware.

**Failure Mode 2: Edge-Level Temporal Interference.** The most commonly overlooked failure mode. When a new fact asserts a relationship for a subject that already holds an exclusive relationship, the system must not only detect the conflict on the subject side but also perform an *object-side* query: which other subjects currently hold an exclusive relationship to the same object? This is the CEO succession problem — if "B becomes CEO of Company X," a temporally correct system must find that "A was CEO of Company X" and close that edge. All systems in the GraphRAG lineage, including ZEP/Graphiti, perform only subject-side detection. The closest any surveyed system comes to addressing this failure mode is EVOKG — a temporal knowledge graph completion system outside the GraphRAG lineage — but its exclusive/non-exclusive distinction uses a binary ontology that cannot distinguish the four structural forms of exclusivity.

**Failure Mode 3: Late-Arrival Entanglement.** In any real-world document corpus, documents that describe historical events arrive after documents that describe more recent events. If the historical report contains facts that were later superseded, processing it as if its facts are "new" creates retroactive inconsistency — the system "learns" an outdated state that is newer in transaction time but older in valid time. No existing RAG system tracks this distinction.

#### 1.6.3 Research Gap and Positioning

The gap analysis of the primary surveyed systems in the knowledge-graph RAG and temporal KG lineage, across eleven capability dimensions, is shown in Table 1.

**Table 1: Capability Gap Analysis**

| System | Valid Time | Transaction Time | Object-Side Conflict | Retroactive Correction | Bitemporal Audit | Cross-Batch Entity Resolution | Cardinality Ontology | Incremental Community | Late Arrival Handling | Certified Unlearning | Temporal Query Decomposition |
|---|---|---|---|---|---|---|---|---|---|---|---|
| GraphRAG | x | x | x | x | x | x | x | x | x | x | x |
| TG-RAG | Y | x | x | x | x | x | x | Partial | x | x | x |
| T-GRAG | Y | x | x | x | x | x | x | x | x | x | Y |
| DyG-RAG | Y | x | x | x | x | x | x | x | x | x | Partial |
| DynaGRAG | x | x | x | x | x | x | x | x | x | x | x |
| ZEP/Graphiti | Y | Y | x | x | Y | Y | Implicit | Y | x | x | x |
| MRAG | Y | x | x | x | x | x | x | x | x | x | Y |
| STAR-RAG | Y | x | x | x | x | x | x | x | x | x | x |
| E2RAG | Y | x | x | x | x | x | x | x | x | x | x |
| EVOKG | Y | x | x | Partial | x | Y | Binary | x | x | x | Y |
| DHE-TKG | Y | x | x | x | x | x | x | x | x | x | x |
| KG-IRAG | Y | x | x | x | x | x | Fixed | x | x | x | Y |
| DYNAMO | Partial | x | x | x | x | x | x | Partial | x | x | x |
| DALK | Partial | x | x | x | x | x | x | x | x | x | x |
| Pathway | Y | Y | n/a | Y | Y | n/a | n/a | n/a | Y | x | n/a |
| Aion | Y | Y | n/a | Y | Y | n/a | n/a | n/a | Y | x | n/a |
| ScaleGUN | x | x | x | x | x | x | x | x | x | Y | x |
| **BT-GraphRAG** | **Y** | **Y** | **Y** | **Y** | **Y** | **Y** | **4-way** | **Y** | **Y** | **Y** | **Y** |

The three critical gaps that BT-GraphRAG uniquely fills are: (1) object-side conflict detection, which no prior system addresses; (2) full bitemporality within the GraphRAG framework, which ZEP achieves only partially (transaction time without retroactive consistency); and (3) the four-way cardinality ontology, which extends EVOKG's binary exclusive/non-exclusive distinction to capture the structural asymmetry between subject-exclusive and object-exclusive relations.

### 1.7 Conclusions

This chapter has established the theoretical and empirical foundations for the problem of Temporal Origin Blindness in knowledge-graph RAG systems. Through a comprehensive survey of 25 related systems across five categories, three systematic failure modes have been identified (entity aliasing, edge-level temporal interference, and late-arrival entanglement), and four levels of temporal awareness have been formalized. The gap analysis demonstrates that no existing system provides the combination of capabilities required to address all three failure modes: object-side conflict detection, full bitemporality with retroactive consistency, and temporally-aware cross-graph entity resolution. BT-GraphRAG is positioned as the first system to fill this gap.

---

## Chapter 2: Proposed Solution

### 2.1 Baselines

#### 2.1.1 Direct GraphRAG Baselines

The primary baseline is static GraphRAG (Edge et al., 2024) with no temporal extensions, establishing the magnitude of the overall temporal consistency problem. A secondary baseline adds minimal temporal metadata (publication-date timestamps only) to GraphRAG without any conflict detection mechanism, isolating the contribution of timestamps alone.

#### 2.1.2 Existing Temporal RAG Systems as Baselines

The following existing systems serve as external baselines for comparison: TG-RAG (valid-time temporal structure with incremental updates but no conflict detection); ZEP/Graphiti (bitemporal with subject-centric conflict detection but no object-side query); EVOKG (binary exclusive/non-exclusive distinction with no object-side conflict detection); and DynaGRAG (no temporal modeling, serving as an ablation baseline for topological contribution).

### 2.2 Design Principles

BT-GraphRAG extends Microsoft's GraphRAG with a temporal layer that operates according to five design principles derived from the problem analysis:

1. **Principle of Temporal Origin.** Every edge entering the graph must carry a valid-time interval specifying when the described fact was true in the world. No edge may be written without this information.

2. **Principle of Epistemic Honesty.** The system must maintain an independent record of when it came to believe each fact (transaction time). This enables distinguishing current truth from historical belief and supports temporal audit queries.

3. **Principle of Cross-Batch Continuity.** Entity resolution must operate across the full existing graph, not only within the current ingestion batch. An entity mentioned for the first time in batch 12 must be matched against all entities from batches 1–11.

4. **Principle of Bidirectional Conflict Detection.** For any exclusive relation $r$ and any new edge $(s', r, o)$, the system must query: (a) does $s'$ already hold $r$ with any object? (subject-side); and (b) does $o$ already have any subject holding $r$ with it? (object-side). Both queries must be executed before any edge is committed.

5. **Principle of Retroactive Consistency.** When a document with event time $t_{event}$ is processed at system time $t_{now} > t_{event}$, conflict detection must execute against the graph state at $t_{event}$, not the current state.

### 2.3 System Architecture Overview

BT-GraphRAG processes incoming documents through seven sequential stages:

```
[Document Stream] → [Stage 1: Temporal Extraction] → [Stage 2: Cross-Graph Entity Resolution (CGER)]
→ [Stage 3: Edge-Level Temporal Conflict Detection and Resolution (ETCDR)]
→ [Stage 4: Bitemporal Graph Store (G_BTC)]
→ [Stage 5: Incremental Community Update]
→ [Stage 6: Selective LLM Summarization]
→ [Stage 7: Query Pipeline]
```

The system is designed as a stream processor in the Pathway/Aion model: documents arrive with an event timestamp (when the document was written or when the described events occurred) and a processing timestamp (when the system ingests the document). The gap between these two timestamps determines whether a document follows the normal path or the late-arrival path.

### 2.4 Stage 1: Temporal Extraction and Document Ingestion

Stage 1 is the system's ingestion front-end. It transforms raw incoming documents into structured extraction units with fully resolved temporal metadata, and produces the entities and relationships — annotated with valid-time intervals — that feed the rest of the pipeline.

**Document dating.** Each incoming document $d$ is assigned two timestamps that become the foundation of the bitemporal model:

- $t_{valid}$: when the facts described were true in the world, derived from datelines, explicit temporal expressions in the document header, or the publication date as a fallback. This corresponds to valid time — the time at which facts hold in the modeled reality.
- $t_{tx}$: the system time at which the document is ingested ($t_{now}$). This corresponds to transaction time — when the system came to know these facts.

The pair $(t_{valid}, t_{tx})$ is attached to the document record and propagates to every entity and edge extracted from it.

**Text unit segmentation.** The document is split into text units (sentences or coherent passages) that serve as the atomic extraction contexts. Each text unit inherits the document-level $(t_{valid}, t_{tx})$ pair as its default temporal scope. This segmentation follows the GraphRAG extraction model, extended here with temporal anchoring at the unit level.

**Temporal expression normalization.** Before entity and relationship extraction, every text unit is scanned for temporal expressions. Absolute dates ("January 15, 2023"), relative expressions ("last year"), natural language intervals ("from 2010 to 2015"), and implicit expressions ("during the merger") are converted to explicit ISO 8601 intervals using dateparser, a contextual reference resolver that uses $t_{valid}$ as the reference point for resolving relative expressions. The output is a set of temporal anchors attached to each text unit, providing fine-grained valid-time signals that override the document-level default when present.

**Entity and relationship extraction.** Each text unit is processed by an LLM-based extraction prompt (following the GraphRAG pattern) that elicits triples extended with temporal scope. Wherever a temporal expression is present — "As of Q3 2023...", "from 2018 through 2021...", "until last Tuesday..." — the extraction prompt converts it to an explicit valid-time interval using the anchors produced in the previous step. Each extracted relationship is represented as:

$$c_i = (\text{subject}_i,\ \text{relation}_i,\ \text{object}_i,\ [t_{valid\_start},\ t_{valid\_end}],\ \text{confidence}_i)$$

Where no temporal expression is present in the text unit, $t_{valid\_start}$ is set to the document's $t_{valid}$ and $t_{valid\_end}$ is left open ($\infty$), to be narrowed by conflict detection in Stage 3.

Each document also produces a **provenance record** $\varepsilon_d = \{\text{embedding\_id},\ \text{source\_url},\ t_{valid},\ t_{tx},\ \text{trust\_score},\ \text{text\_hash}\}$ that is stored alongside every edge for full audit traceability.

**Late-arrival detection.** A document whose valid time is more than $\theta_{late}$ before the current system time (i.e., $t_{tx} - t_{valid} > \theta_{late}$) is tagged as a **late arrival** and routed through the Retroactive Consistency path (Section 2.6.4). This tag is assigned at this stage so that all downstream components — CGER, ETCDR, and the graph store — can apply the appropriate historical-state queries.

### 2.5 Stage 2: Cross-Graph Entity Resolution (CGER)

CGER is designed to prevent the Entity Aliasing failure mode by resolving every newly extracted entity against the full existing graph before any new node is created. Unlike standard within-batch deduplication or naive embedding similarity (which E2RAG shows destroys temporal context), CGER uses a five-signal composite scoring function:

$$\text{score}(e_{new}, e_{existing}) = w_1 \cdot \text{cosine}(d_{new}, d_{existing}) + w_2 \cdot \text{BM25}(n_{new}, n_{existing}) + w_3 \cdot \text{Jaccard}(n_{new}, n_{existing}) + w_4 \cdot \text{TemporalOverlap}(e_{new}, e_{existing}) + w_5 \cdot \text{RelationContext}(e_{new}, e_{existing})$$

where $d$ is the entity description embedding, $n$ is the entity name, TemporalOverlap measures the intersection of the entities' known active periods, and RelationContext is the cosine similarity of the relationship-neighbor embedding vectors.

Candidate retrieval uses FAISS approximate nearest neighbor search on description embeddings combined with Elasticsearch BM25 lexical matching; the top-20 candidates from each signal are merged. Multi-signal scoring is applied to all merged candidates. Hard cases — pairs scoring between 0.55 and 0.85 on the composite score — are escalated to LLM verification with a structured prompt that presents both entities' names, types, descriptions, known relationships, and active periods.

The temporal overlap signal is the key innovation over prior entity resolution approaches. When the existing graph contains "Apple Inc. (active: 1976–present)" and the incoming document discusses "Apple Computer Company (active: 1976–1977)," the temporal overlap signal is high and correctly identifies them as the same entity. When the documents discuss a company and a later spin-off with the same name, the temporal overlap is low and correctly produces separate nodes.

#### 2.5.1 Cross-Graph Relationship Resolution (CGRR)

Entity aliasing has a less-studied but equally damaging counterpart: **Relationship Aliasing**. When different documents describe the same real-world relationship using different natural language expressions — e.g., "is CEO of," "leads," "serves as chief executive of," "heads" — the extraction pipeline produces distinct `relation_type` strings for what is semantically the same predicate. Because ETCDR's conflict sub-queries filter on `relation_type` (Section 2.6.2), unresolved relationship aliases cause two systematic failures:

1. **Missed conflicts.** An existing edge `(Alice, IS_CEO_OF, Acme)` will not be found by the subject-side query when the candidate arrives as `(Bob, LEADS, Acme)` because the `relation_type` filter does not match. Both edges survive as simultaneously active, violating the BOTH_EXCLUSIVE cardinality constraint — the exact failure mode ETCDR was designed to prevent.

2. **Spurious duplication.** Corroborative evidence for the same fact gets inserted as separate edges instead of incrementing `support_count`, inflating the graph with semantically redundant edges and diluting confidence signals.

BT-GraphRAG addresses this with **Cross-Graph Relationship Resolution (CGRR)**, which runs after CGER and before ETCDR. For every candidate relationship, CGRR resolves its `relation_type` against the canonical relation types already present in the graph, using a three-signal composite scoring function:

$$\text{score}(r_{new}, r_{existing}) = w_1 \cdot \text{BM25}(n_{new}, n_{existing}) + w_2 \cdot \text{SemanticSim}(d_{new}, d_{existing}) + w_3 \cdot \text{EndpointMatch}(r_{new}, r_{existing})$$

where:

- **BM25 lexical matching** ($w_1 = 0.35$): Measures term overlap between the normalized relation type strings (e.g., "IS_CEO_OF" vs "IS_CHIEF_EXECUTIVE_OF").
- **Semantic similarity** ($w_2 = 0.40$): Cosine similarity between the full relationship description embeddings. This captures cases where surface forms differ entirely (e.g., "leads" vs "is CEO of") but the underlying semantics are equivalent.
- **Endpoint match** ($w_3 = 0.25$): A binary signal boosted when the candidate shares the same subject AND object (or resolved entity equivalents after CGER) as an existing edge. Two relationships between the same pair of entities with similar descriptions are overwhelmingly likely to be the same predicate.

The resolution pipeline operates as follows:

1. **Candidate collection.** For each unique `relation_type` in the incoming batch, CGRR queries Neo4j for all distinct `relation_type` values currently in the graph, along with a sample description for each.

2. **Composite scoring.** Each candidate relation type is scored against all existing relation types using the three-signal function.

3. **Automatic normalization** ($\text{score} \geq \theta_{merge} = 0.80$). The candidate's `relation_type` is rewritten to the canonical existing form. All relationships in the batch with that type are updated.

4. **LLM verification** ($0.50 \leq \text{score} < 0.80$). An LLM is prompted with both relation types, their sample descriptions, and example edges to determine if they refer to the same predicate. This resolves ambiguous cases like "works for" vs "employed by" (SAME) versus "works for" vs "works with" (DIFFERENT).

5. **Cardinality inheritance.** When a candidate relation type is merged into an existing canonical type, it inherits the existing type's cardinality classification. This ensures that ETCDR's bidirectional queries activate correctly even when the original extraction used a novel surface form.

**Why CGRR must precede ETCDR.** If relationship aliases are not resolved before conflict detection, ETCDR's Cypher queries — which filter on `e.relation_type = $relation_type` — will systematically miss conflicts across alias boundaries. No amount of sophistication in the Decision Router can compensate for conflicts that were never surfaced by the query layer. CGRR ensures that by the time ETCDR runs, all semantically equivalent relationships share a canonical `relation_type`, making the conflict queries complete.

**Comparison with prior work.** ZEP/Graphiti performs no relationship normalization, relying entirely on exact string matching of relation types. EVOKG normalizes predicates via a fixed ontology, but cannot handle open-domain extraction where relation types are generated by LLMs with no ontological constraints. BT-GraphRAG's CGRR combines the flexibility of open-domain extraction with the precision of ontology-based matching through its learned composite scoring function.

### 2.6 Stage 3: Edge-Level Temporal Conflict Detection and Resolution (ETCDR)

ETCDR is the system's core conflict detection engine. Before any relationship extracted from the current document is written to the graph, ETCDR executes bidirectional conflict queries.

#### 2.6.1 Four-Way Relation Cardinality Ontology

The ETCDR engine operates on a four-way relation cardinality ontology that classifies every relation type by its structural exclusivity constraints. This ontology extends EVOKG's binary exclusive/non-exclusive dichotomy to capture asymmetric cases:

| Cardinality Type | Meaning | Example |
|---|---|---|
| **SUBJECT_EXCLUSIVE** | One subject can hold at most one active instance of this relation | *is\_nationality\_of* (a person has one nationality at a time) |
| **OBJECT_EXCLUSIVE** | One object can have at most one active subject for this relation | *has\_capital\_city* (a country has one capital at a time) |
| **BOTH_EXCLUSIVE** | Both sides: one-to-one constraint | *is\_CEO\_of* (one CEO per company, and one CEO role per person at a time) |
| **NON\_EXCLUSIVE** | No exclusivity constraint; history accumulated | *worked\_at*, *appeared\_in*, *co-authored* |

The BOTH_EXCLUSIVE case captures the structure that EVOKG's binary "exclusive" category misses: the CEO relation is not only exclusive for the CEO (one person, one company at a time) but also exclusive for the company (one company, one CEO at a time). BT-GraphRAG activates object-side queries for OBJECT_EXCLUSIVE and BOTH_EXCLUSIVE relations.

#### 2.6.2 Bidirectional Conflict Detection

For every candidate edge $(s_{new}, r, o_{new})$, ETCDR executes two parallel conflict sub-queries in Neo4j:

**Sub-query S (subject-side):** `MATCH (s_new)-[e:r]->(o) WHERE e.t_tx_end = inf AND e.t_valid_end = inf`
Finds currently believed, currently active edges where the same subject holds the same relation.

**Sub-query O (object-side, activated for OBJECT_EXCLUSIVE and BOTH_EXCLUSIVE):**
`MATCH (s)-[e:r]->(o_new) WHERE e.t_tx_end = inf AND e.t_valid_end = inf AND s != s_new`
Finds currently believed, currently active edges where a *different* subject holds the same exclusive relation to the same object.

The object-side query is the critical addition. Consider the canonical example:

```
Existing:   (A, is_CEO_of, X, t_valid=[t1, inf])   <- ACTIVE
Incoming:   (B, is_CEO_of, X, t_valid=[t2, inf])   <- CANDIDATE
Cardinality: BOTH_EXCLUSIVE

Sub-query S: subject=B, relation=is_CEO_of -> EMPTY (B has no prior CEO edge for X)
Sub-query O: object=X, relation=is_CEO_of, subject!=B -> FINDS (A, is_CEO_of, X)
```

Without the object-side query, $E_{conflict}$ would be empty and both A and B would remain simultaneously active as CEO of X — a systematic failure mode that no prior system in the GraphRAG lineage detects. ZEP/Graphiti would also miss this case, as its conflict detection is subject-centric only.

#### 2.6.3 The Decision Router

An LLM classifier with access to the candidate edge and its source context, all conflicting edges and their contexts, the relation's cardinality type, relative source trust scores, and temporal ordering selects one of four resolution strategies:

| Strategy | Condition | Action on Conflicting Edge | Action on Candidate |
|---|---|---|---|
| **Evolution** | World genuinely changed | Close $t_{valid\_end}$ at candidate's start time | Insert with full Temporal State Quad |
| **Correction** | Prior edge was wrong | Set $t_{tx\_end} = t_{now}$ (retroactive invalidation, preserved for audit) | Insert, inheriting old edge's valid-time |
| **Corroboration** | Candidate matches existing edge | Increment support\_count, append provenance | No new edge; update covariates only |
| **Disagreement** | No clear resolution | No change | Insert with status=disputed |

This four-way strategy map directly implements the TLINQ/VLINQ distinction between sequenced updates (Evolution: new period starts) and nonsequenced updates (Correction: fix a timestamp/value) at the knowledge graph level. It also operationalizes LOKA's insight that editing (Evolution/Correction) and unlearning (Correction/Retraction) must be kept distinct to avoid gradient conflicts in downstream LLM fine-tuning scenarios.

When classification confidence falls below threshold, the system defaults to **Disagreement** rather than risking an incorrect commit.

#### 2.6.4 The Late-Arrival Path

When a document is flagged as a late arrival ($t_{now} - t_{event} > \theta_{late}$), both conflict sub-queries are modified to query the **historical state at $t_{event}$** rather than the current state (implementing Principle 5). The Decision Router runs against this historical view, and consequences propagate forward:

A retroactive **Evolution** discovered via the object-side query may close an edge that was believed active during $[t_{event}, t_{now}]$, potentially invalidating currently-active edges downstream. A retroactive **Correction** invalidates an edge that may have been further corroborated between $t_{event}$ and $t_{now}$, requiring cascading confidence adjustments. A late-arriving edge with no conflict is inserted as Epistemic State 4 (Retroactive Correction): true in the past, but only now known.

### 2.7 Stage 4: Bitemporal Graph Store

Every edge carries a **Temporal State Quad**: $T(e) = [t_{valid\_start},\ t_{valid\_end},\ t_{tx\_start},\ t_{tx\_end}]$

This is the storage-level implementation of Principles 1 and 2. The four epistemic states derivable from this quad are:

| State | Condition | Meaning |
|---|---|---|
| **Current Truth** | $t_{valid\_end} = \infty$ and $t_{tx\_end} = \infty$ | True now, believed now |
| **Historical Truth** | $t_{valid\_end} < \infty$ and $t_{tx\_end} = \infty$ | Was true, still believed (historical fact) |
| **Retracted Error** | $t_{valid\_end} = \infty$ and $t_{tx\_end} < \infty$ | Believed true but retracted (was an error) |
| **Retroactive Correction** | $t_{valid\_end} < \infty$ and $t_{tx\_end} < \infty$ | Historical fact, now superseded in the system's record |

SCD2 (Slowly Changing Dimension Type 2) non-destructive operations over Neo4j ensure no information is ever physically deleted — superseded facts are closed by updating their transaction-time end, not erased. Composite indexes on both $(subject, relation, t_{valid\_start}, t_{tx\_end})$ and $(object, relation, t_{valid\_start}, t_{tx\_end})$ support efficient object-side conflict queries.

This is a direct implementation of the SQL:2011 bitemporal standard (ISO/IEC 9075:2011), grounded in the formal semantics of VLINQ/TLINQ and aligned with the event-time/processing-time distinction of Pathway and Aion.

### 2.8 Stages 5–6: Incremental Community Update and Selective Summarization

GraphRAG's Leiden community detection and LLM-based summarization are preserved but made incremental. Leiden re-runs only on the $k$-hop neighborhood (typically $k=2$) of modified entities after each ETCDR batch — including entities affected by object-side resolutions, which may lie far from the subject of the incoming document. Community reports are regenerated only when marked stale, and carry temporal annotations reflecting evolution over time. This selective approach directly mirrors TG-RAG's empirical demonstration of 1.6M vs. 30M tokens for incremental vs. full updates.

### 2.9 Query Pipeline

The query pipeline extends GraphRAG's original search modes with temporal semantics derived from the bitemporal store:

**Local Temporal Search:** Filters to edges valid at query time $T$ and currently believed ($t_{tx\_end} = \infty$). Temporal scoring uses DyG-RAG's exponential decay formulation $\exp(-\alpha \cdot |t_q - t_e|)$ to weight edges by temporal proximity to the query.

**Global Temporal Search:** MapReduce over community reports temporally valid at $T$.

**Temporal Audit Search:** "What did the system believe at $T_1$?" — queries by transaction-time, enabling full reconstruction of historical system states.

**Dispute Resolution Search:** Surfaces disputed edges for queries touching unresolved conflicts, presenting competing claims with provenance chains.

**Temporal Query Decomposition:** Complex temporal queries are decomposed MRAG-style and T-GRAG-style into sub-queries each with an explicit time constraint. Sub-query answers are combined using EVOREASONER's temporal alignment scoring. For long historical spans, STAR-RAG's MDL-guided rule summarization compresses the historical context.

### 2.10 Knowledge Maintenance Module

**Privacy-Compliant Removal (GDPR):** Close $t_{tx\_end}$ in the bitemporal store; apply ScaleGUN-style lazy local propagation with certified $(\varepsilon, \delta)$-differential privacy guarantees; invalidate and regenerate affected summaries. This achieves ~1,000x speedup over full retraining while maintaining privacy certification.

**Bulk Source Retraction:** Batch-invalidate all edges whose provenance traces to the retracted source; cascade confidence adjustments using EVOKG-style confidence scoring based on remaining corroborating sources.

**Temporal Archival:** Compress facts beyond a configurable horizon into STAR-RAG MDL rule summaries, reducing storage while preserving queryability of the historical record.

---

## Bibliography

- Chen, X. et al. (2024). A Survey on Temporal Knowledge Graph: Representation Learning and Applications. arXiv:2403.04782.
- Cai, L. et al. (2023). Temporal Knowledge Graph Completion: A Survey. IJCAI 2023, doi:10.24963/ijcai.2023/730.
- Edge, D. et al. (2024). From Local to Global: A Graph RAG Approach to Query-Focused Summarization. Microsoft Research. arXiv:2404.16130.
- ISO/IEC 9075:2011. Information Technology — Database Languages — SQL. (SQL:2011 Bitemporal Standard).
- Lewis, P. et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. NeurIPS 2020.
- Snodgrass, R. T. (1999). Developing Time-Oriented Database Applications in SQL. Morgan Kaufmann.
- Yang, J. et al. (2024). CRAG — Comprehensive RAG Benchmark. arXiv:2406.04744. (4,409 QA pairs, 2.6M entities, 5 domains)
- Xu, Z. et al. (2025). ZEP / Graphiti: A Temporal Knowledge Graph Architecture for Agent Memory. (94.8% DMR, 18.5% LongMemEval improvement, BGE-m3, Dynamic Label Propagation)
- Acharya, S. et al. (2025). T-GRAG: A Dynamic GraphRAG Framework for Resolving Temporal Conflicts and Redundancy in Knowledge Retrieval. (2,292 QA, Audi 2012–2023, 38–47% gain, stella-en-1.5B-v5)
- Lu, Y. et al. (2025). RAG Meets Temporal Graphs: Time-Sensitive Modeling and Retrieval for Evolving Knowledge (TG-RAG). (ECT-QA 480 transcripts, Correct=0.599, 1.6M vs 30M tokens, text-embedding-3-small)
- Zhang, Q. et al. (2025). DynaGRAG: Exploring the Topology of Information for Advancing Language Understanding. (DSA-BFS, GCN, JinaAI embeddings, podcast corpus)
- Wang, K. et al. (2024). DyG-RAG: Dynamic Graph Retrieval-Augmented Generation with Event-Centric Reasoning. (DEUs, Time-CoT, Fourier encoder, BGE-M3, ~10–18% gain on TimeQA/TempReason/ComplexTR)
- Lim, W. et al. (2024). MRAG: A Modular Retrieval Framework for Time-Sensitive Question Answering. (TempRAGEval, 33.1M Wikipedia chunks, 9.3% answer recall, ~2x overhead, Contriever)
- Sun, H. et al. (2025). Right Answer at the Right Time: Temporal Retrieval-Augmented Generation via Graph Summarization (STAR-RAG). (9.1% accuracy, 97% token reduction, MDL-guided rule graph, NV-Embed)
- Maguire, P. et al. (2025). Respecting Temporal-Causal Consistency: Entity-Event Knowledge Graphs for RAG (E2RAG). (ChronoQA 497 QA, 108s/book, text-embedding-3-small; "deduplication destroys temporal context")
- Zaharia, M. et al. (2025). KG-IRAG: Knowledge Graph-Based Iterative Retrieval-Augmented Generation Framework for Temporal Reasoning. (weatherQA-Irish 227K entities, weatherQA-Sydney 332K entities, trafficQA-TFNSW 132K entities)
- Lin, Y. et al. (2025). Temporal Reasoning over Evolving Knowledge Graphs (EVOREASONER / EVOKG). (23.3% improvement on TimeQuestions, 8B matches 671B, exclusive vs non-exclusive distinction)
- Liu, Z. et al. (2025). A Dynamic Knowledge Update-Driven Model with LLMs for Fake News Detection (DYNAMO). (MCTS h=5/9, K=5 triples, Hover 18,171 samples, Feverous 26,928 samples, NVIDIA A40)
- Jin, W. et al. (2024). DALK: Dynamic Co-Augmentation of LLMs and KG to Answer Alzheimer's Disease Questions. (ADQA, MedQA, MedMCQA, MMLU, self-aware knowledge retrieval)
- Isard, M. et al. (2022). Pathway: A Fast and Flexible Unified Stream Data Processing Framework. (COMMIT-protocol, differential dataflow, Rust engine, outperforms Flink/Spark, LiveJournal 4.8M nodes/69M edges)
- Karakaya, I. et al. (2020). Aion: Better Late than Never in Event-Time Streams. (Proactive Caching, Predictive Cleanup, dual m-bucket/p-bucket, 50% memory reduction, Apache Flink)
- Zhang, Z. et al. (2024). Scalable and Certifiable Graph Unlearning: Overcoming the Approximation Error Barrier (ScaleGUN). ((e,d)-DP, Lazy Local Propagation, 20s for 5,000 edges on 1.6B-edge graph vs 1.91h retraining, ogbn-papers100M)
- Liu, Y. et al. (2023). GIF: A General Graph Unlearning Strategy via Influence Function. WWW 2023. (15–100x speedup, Cora 0.16s vs 6.33s retraining, O(n|theta|) complexity)
- Chen, M. et al. (2022). Graph Unlearning (GraphEraser). arXiv:2103.14991. (GDPR compliance, graph partitioning, Cora/Citeseer/Pubmed/CS/Physics)
- Bowles, J. et al. (2022). Language-Integrated Query for Temporal Data (TLINQ/VLINQ). arXiv:2210.12077. (Bitemporal [start,end) intervals, sequenced/nonsequenced updates, ACID via RDBMS)
- Huang, C. et al. (2025). Resolving Editing-Unlearning Conflicts: A Knowledge Codebook Framework (LOKA). (Truth Ratio 0.80 vs 0.42, gradient conflict score, Llama3-8b/Mistral-7b, TOFU/ZsRE/PKU-SafeRLHF, 80GB A100)
- Dhulipala, L. et al. (2025). Retrieval Augmented Generation for Dynamic Graph Modeling (RAG4DyG). (exp(-lambda*dt) decay, Graph Fusion GCN, SimpleDyG backbone, UCI/Hepth/Enron/Reddit)
- Huang, X. et al. (2024). Temporal Knowledge Graph Reasoning with Dynamic Hypergraph Embedding (DHE-TKG). LREC-COLING 2024. (ICEWS14/18/05-15/WIKI/YAGO, R-GCN + hypergraph encoder, evolutionary embedding)
- Huang, Y. et al. (2023). Temporal Graph Benchmark for Machine Learning on Temporal Graphs (TGB). NeurIPS 2023. (9 datasets, filtered MRR, NDCG@10, tgbn-token 72.9M edges, tgbl-flight 67.2M edges)

---

## Appendices

### Appendix A: Formal Definitions

**Definition 1 — Temporal Origin Blindness (TOB).** A knowledge graph system $\mathcal{G}$ exhibits Temporal Origin Blindness if, for any edge $e \in \mathcal{G}$, the system stores no explicit representation of when the fact described by $e$ became true in the world, and consequently has no mechanism to determine whether a newly arriving fact $e'$ supersedes, corroborates, or conflicts with $e$.

**Definition 2 — Temporal Knowledge Drift (TKD).** Let $\mathcal{G}(t)$ denote the facts stored in the system at time $t$, and let $\mathcal{W}(t)$ denote the set of facts true in the real world at time $t$. Temporal Knowledge Drift is the symmetric difference $\text{TKD}(t) = \mathcal{G}(t) \triangle \mathcal{W}(t)$, measuring the divergence between the graph's representation and world truth. Under Temporal Origin Blindness, $|\text{TKD}(t)|$ grows monotonically with time in any domain where facts change.

**Definition 3 — Temporal State Quad.** For any edge $e$, the Temporal State Quad is the tuple $T(e) = [t_{valid\_start}, t_{valid\_end}, t_{tx\_start}, t_{tx\_end}]$ where $t_{valid\_start}, t_{valid\_end}$ define the valid-time interval (SQL:2011 APPLICATION TIME) and $t_{tx\_start}, t_{tx\_end}$ define the transaction-time interval (SQL:2011 SYSTEM TIME). The current truth state is $T(e)$ with $t_{valid\_end} = \infty$ and $t_{tx\_end} = \infty$.

**Definition 4 — Epistemic States.** Given a Temporal State Quad $T(e)$, four epistemic states are defined:
- **Current Truth**: $t_{valid\_end} = \infty \wedge t_{tx\_end} = \infty$ (true now, believed now)
- **Historical Truth**: $t_{valid\_end} < \infty \wedge t_{tx\_end} = \infty$ (was true, still believed)
- **Retracted Error**: $t_{valid\_end} = \infty \wedge t_{tx\_end} < \infty$ (believed true but retracted)
- **Retroactive Correction**: $t_{valid\_end} < \infty \wedge t_{tx\_end} < \infty$ (historical fact, now superseded in the record)

**Definition 5 — Ghost Rate.** For a set of evaluation queries $Q$ and a ground-truth bitemporal store $\mathcal{G}^*$, the Ghost Rate of a retrieval system $\mathcal{S}$ is:

$$\text{GhostRate}(\mathcal{S}, Q, \mathcal{G}^*, t_q) = \frac{|\{q \in Q : \exists f \in \text{answer}(q, \mathcal{S}) \text{ s.t. } f \notin \mathcal{W}(t_q)\}|}{|Q|}$$

where $\text{answer}(q, \mathcal{S})$ is the set of facts cited in $\mathcal{S}$'s answer to query $q$, and $\mathcal{W}(t_q)$ is world truth at query time $t_q$. A system with Ghost Rate = 0 never cites superseded facts. Standard temporal QA benchmarks (MRR, Hits@k, CRAG Truthfulness) do not measure this directly.

**Definition 6 — Role Simultaneity Rate (RSR).** For an exclusive relation $r$ and a knowledge graph $\mathcal{G}$, the Role Simultaneity Rate is:

$$\text{RSR}(r, \mathcal{G}) = \frac{|\{(o, [t_a, t_b]) : \exists s_1 \neq s_2 \text{ s.t. } (s_1, r, o) \text{ and } (s_2, r, o) \text{ are both active in } \mathcal{G} \text{ during } [t_a, t_b]\}|}{|\text{total active intervals for } r \text{ in } \mathcal{G}|}$$

A RSR of 0 means no exclusive relation is simultaneously held by multiple subjects (or objects) at any time — the correct state. RSR > 0 directly and quantitatively measures the object-side temporal interference failure mode. All systems without object-side detection will exhibit RSR > 0 on domains with leadership succession, ownership transfer, or any other OBJECT_EXCLUSIVE relation change.

### Appendix B: ETCDR Decision Router Prompt Template

[Full prompt specification to be included in final implementation]

### Appendix C: TemporalConflict Benchmark Construction Protocol

[Detailed construction methodology, annotation guidelines, inter-annotator agreement measurements, and ground-truth bitemporal labeling procedures to be included after benchmark construction]

### Appendix D: BT-GraphRAG vs. Related Systems — Quantitative Reference

| System | Key Dataset | Key Metric | Value | Temporal Model |
|---|---|---|---|---|
| CRAG | 4,409 QA, 5 domains | Truthfulness (fast-changing entities) | Significantly lower than static domains | None |
| ZEP/Graphiti | LongMemEval, DMR | Accuracy improvement / DMR Accuracy | +18.5% / 94.8% | Bitemporal (subject-centric) |
| T-GRAG | Time-LongQA (2,292 QA) | Multi-time improvement over GraphRAG | +38–47% | Valid time only |
| TG-RAG | ECT-QA (480 transcripts) | Correct / Update token cost | 0.599 / 1.6M vs 30M | Valid time only |
| DyG-RAG | TimeQA (3,159 docs) | Accuracy gain over baselines | +10–18% | Valid time + decay |
| MRAG | TempRAGEval | Answer recall / QA accuracy | +9.3% / +4.5% | Valid time + scoring |
| STAR-RAG | CronQuestion/MultiTQ | Accuracy / Token reduction | +9.1% / -97% | Valid time + rules |
| EVOKG | TimeQuestions, CRAG | Improvement over CoT/ToG | +23.3% | Valid time + exclusive/non-exclusive |
| ScaleGUN | ogbn-papers100M (1.6B edges) | Unlearning time (5,000 edges) | 20s vs 1.91h retraining | None (GNN parameters) |
| GIF | Cora | Unlearning time | 0.16s vs 6.33s retraining | None |
| LOKA | TOFU/ZsRE | Truth Ratio | 0.80 vs 0.42 baseline | None (LLM params) |
| **BT-GraphRAG** | **TemporalConflict** | **Ghost Rate / RSR** | **Target: Ghost Rate ≤50% of TG-RAG; RSR=0** | **Full bitemporal + object-side** |
