# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""LLM prompts for BT-GraphRAG temporal pipeline stages.

Contains prompts for:
- Temporal graph extraction (Stage 1)
- Relation cardinality classification (Stage 3)
- Decision Router conflict resolution (Stage 3)
- Temporal community reports (Stage 6)
- Temporal query decomposition (Stage 7)
"""

# ---------------------------------------------------------------------------
# Stage 1: Temporal Graph Extraction
# ---------------------------------------------------------------------------

TEMPORAL_GRAPH_EXTRACTION_PROMPT = """
-Goal-
Given a text document that is potentially relevant to this activity, a list of entity types, and a document date ({document_date}), identify all entities of those types from the text and all relationships among the identified entities, with special attention to TEMPORAL information.

-Steps-
1. Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity, capitalized
- entity_type: One of the following types: [{entity_types}]
- entity_description: Comprehensive description of the entity's attributes and activities

Format each entity as ("entity"<|><entity_name><|><entity_type><|><entity_description>)

2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relationship_description: explanation as to why you think the source entity and the target entity are related to each other
- relation_type: A short, canonical predicate label in UPPER_SNAKE_CASE that describes the nature of the relationship between the source and target entities. This MUST be a generic, reusable predicate — it must NOT contain entity names, proper nouns, or instance-specific details. Think of it as the edge label in a knowledge graph ontology. Good examples: IS_CEO_OF, ACQUIRED, HEADQUARTERED_IN, FOUNDED, INVESTED_IN, EMPLOYED_AT, COLLABORATED_WITH, REPORTED_ON. Bad examples: JOHN_SMITH_IS_CEO (contains entity name), TECHCORP_ACQUIRED_DATASOFT (contains entity names), SERVES_AS_CEO_OF_TECHCORP (contains entity name).
- relationship_strength: a numeric score indicating strength of the relationship between the source entity and target entity (1-10)
- valid_time_start: When this relationship started being true. Use ISO date format (YYYY-MM-DD) when possible. If the text says "since 2020", use "2020-01-01". If the text says "As of Q3 2023", use "2023-07-01". Use "UNKNOWN" only if truly unknowable.
- valid_time_end: When this relationship stopped being true. Use ISO date format. Use "ONGOING" if the relationship is still active. Use "UNKNOWN" only if truly unknowable.

IMPORTANT temporal rules:
- The document was written/published on {document_date}. Use this as the reference point for relative temporal expressions.
- "last year" means the year before {document_date}.
- "recently", "currently", "now" → valid_time_start around {document_date}, valid_time_end = ONGOING
- "from X to Y" → valid_time_start = X, valid_time_end = Y
- "since X" → valid_time_start = X, valid_time_end = ONGOING
- "until X" → valid_time_end = X
- "former", "ex-", "previously" → the relationship has ended, valid_time_end should be before {document_date}
- If no temporal info is available, set valid_time_start to {document_date} and valid_time_end to ONGOING.

Format each relationship as ("relationship"<|><source_entity><|><target_entity><|><relationship_description><|><relation_type><|><relationship_strength><|><valid_time_start><|><valid_time_end>)

3. Return output in English as a single list of all the entities and relationships identified in steps 1 and 2. Use **##** as the list delimiter.

4. When finished, output <|COMPLETE|>

######################
-Examples-
######################
Example 1:
Entity_types: ORGANIZATION,PERSON
Document_date: 2024-01-15
Text:
As of January 2024, John Smith serves as CEO of TechCorp, a position he has held since March 2021. He previously led DataSoft from 2015 to 2020 before it was acquired by GlobalTech.
######################
Output:
("entity"<|>JOHN SMITH<|>PERSON<|>John Smith is a business executive who currently serves as CEO of TechCorp and previously led DataSoft)
##
("entity"<|>TECHCORP<|>ORGANIZATION<|>TechCorp is a technology company where John Smith currently serves as CEO)
##
("entity"<|>DATASOFT<|>ORGANIZATION<|>DataSoft is a company that was previously led by John Smith before being acquired by GlobalTech)
##
("entity"<|>GLOBALTECH<|>ORGANIZATION<|>GlobalTech is a company that acquired DataSoft)
##
("relationship"<|>JOHN SMITH<|>TECHCORP<|>John Smith serves as CEO of TechCorp since March 2021<|>IS_CEO_OF<|>9<|>2021-03-01<|>ONGOING)
##
("relationship"<|>JOHN SMITH<|>DATASOFT<|>John Smith previously led DataSoft from 2015 to 2020<|>LED<|>7<|>2015-01-01<|>2020-12-31)
##
("relationship"<|>GLOBALTECH<|>DATASOFT<|>GlobalTech acquired DataSoft<|>ACQUIRED<|>8<|>2020-01-01<|>ONGOING)
<|COMPLETE|>

######################
Example 2:
Entity_types: ORGANIZATION,GEO,PERSON
Document_date: 2023-06-01
Text:
During Q2 2023, Nexon relocated its headquarters from Berlin to Munich. The move was overseen by CFO Maria Garcia, who joined the company last year.
######################
Output:
("entity"<|>NEXON<|>ORGANIZATION<|>Nexon is a company that relocated its headquarters from Berlin to Munich in Q2 2023)
##
("entity"<|>BERLIN<|>GEO<|>Berlin was the former headquarters location of Nexon)
##
("entity"<|>MUNICH<|>GEO<|>Munich is the current headquarters location of Nexon since Q2 2023)
##
("entity"<|>MARIA GARCIA<|>PERSON<|>Maria Garcia is the CFO of Nexon who oversaw the headquarters relocation)
##
("relationship"<|>NEXON<|>BERLIN<|>Nexon was headquartered in Berlin before relocating<|>HEADQUARTERED_IN<|>6<|>UNKNOWN<|>2023-04-01)
##
("relationship"<|>NEXON<|>MUNICH<|>Nexon relocated its headquarters to Munich during Q2 2023<|>HEADQUARTERED_IN<|>8<|>2023-04-01<|>ONGOING)
##
("relationship"<|>MARIA GARCIA<|>NEXON<|>Maria Garcia serves as CFO of Nexon and oversaw the headquarters relocation<|>IS_CFO_OF<|>8<|>2022-01-01<|>ONGOING)
<|COMPLETE|>

######################
-Real Data-
######################
Entity_types: [{entity_types}]
Document_date: {document_date}
Text: {input_text}
######################
Output:
"""

TEMPORAL_CONTINUE_PROMPT = "MANY entities and relationships were missed in the last extraction. Pay particular attention to temporal details — dates, periods, and changes over time. Add them below using the same format:\n"
TEMPORAL_LOOP_PROMPT = "It appears some entities and relationships with temporal details may have still been missed. Answer YES | NO\n"


# ---------------------------------------------------------------------------
# Stage 3: Relation Cardinality Classification
# ---------------------------------------------------------------------------

CARDINALITY_CLASSIFICATION_PROMPT = """You are a knowledge graph ontology expert. Given a relationship type, classify its structural cardinality constraint.

Relationship Type: {relation_type}
Context examples from the graph:
{context_examples}

The four cardinality types are:

1. **SUBJECT_EXCLUSIVE**: One subject can hold at most one active instance of this relation at a time.
   Examples: is_nationality_of (a person has one nationality at a time), is_headquartered_in (a company has one HQ at a time)

2. **OBJECT_EXCLUSIVE**: One object can have at most one active subject for this relation at a time.
   Examples: has_capital_city (a country has one capital at a time), is_primary_language_of (one primary language per country)

3. **BOTH_EXCLUSIVE**: One-to-one constraint on both sides simultaneously.
   Examples: is_CEO_of (one CEO per company AND one CEO role per person at a time), is_married_to (one spouse per person at a time)

4. **NON_EXCLUSIVE**: No exclusivity constraint; multiple instances can coexist.
   Examples: worked_at, appeared_in, co-authored, invested_in, collaborated_with

Respond with ONLY one of: SUBJECT_EXCLUSIVE, OBJECT_EXCLUSIVE, BOTH_EXCLUSIVE, NON_EXCLUSIVE
"""


# ---------------------------------------------------------------------------
# Stage 6: Temporal Community Reports
# ---------------------------------------------------------------------------

TEMPORAL_COMMUNITY_REPORT_PROMPT = """You are a helpful assistant responsible for generating a comprehensive temporal report about a community within a knowledge graph.

The community has the following entities and relationships, each with temporal annotations showing when facts were true:

{input_text}

The report should include the following sections:
- TITLE: A name for the community that reflects its key entities and theme
- SUMMARY: An executive overview of the community's structure and how it has evolved over time
- TEMPORAL_EVOLUTION: A chronological description of how key relationships changed, including:
  - When relationships started and ended
  - Major transitions (e.g., leadership changes, organizational restructuring)
  - Currently disputed or uncertain facts
- IMPACT_RATING: A single float score (0-10) indicating the importance and temporal complexity of this community
- RATING_EXPLANATION: A short rationale for the rating
- KEY_FINDINGS: 5-10 insights about the community, including temporal patterns

Format as JSON:
{{
    "title": "<title>",
    "summary": "<summary>",
    "temporal_evolution": "<chronological narrative>",
    "rating": <float>,
    "rating_explanation": "<explanation>",
    "findings": [
        {{"summary": "<finding_summary>", "explanation": "<finding_explanation> [Data: Entities (<ids>); Relationships (<ids>)]"}}
    ]
}}
"""


# ---------------------------------------------------------------------------
# Stage 7: Temporal Query Decomposition
# ---------------------------------------------------------------------------

TEMPORAL_QUERY_DECOMPOSITION_PROMPT = """You are a temporal reasoning expert. Given a complex query that involves temporal aspects, decompose it into simpler sub-queries that each target a specific time point or interval.

Original Query: {query}
Current Date: {current_date}

For each sub-query, specify:
1. The sub-query text
2. The temporal constraint (a specific date, date range, or "current")
3. The query type: POINT_IN_TIME, RANGE, EVOLUTION, or COMPARISON

Format as JSON array:
[
    {{
        "sub_query": "<sub-query text>",
        "temporal_constraint": "<date or date range>",
        "query_type": "<POINT_IN_TIME|RANGE|EVOLUTION|COMPARISON>"
    }}
]

If the query has no temporal aspect, return a single sub-query with temporal_constraint="current" and query_type="POINT_IN_TIME".
"""


# ---------------------------------------------------------------------------
# Stage 7: Temporal Answer Synthesis
# ---------------------------------------------------------------------------

TEMPORAL_ANSWER_SYNTHESIS_PROMPT = """You are a temporal knowledge synthesis expert. Given sub-query results from different time periods, synthesize a coherent answer to the original question.

Original Question: {query}

Sub-query Results:
{sub_results}

Instructions:
- Integrate information across time periods into a coherent narrative
- Highlight temporal changes and transitions explicitly
- When information conflicts across time periods, explain the evolution
- Note any disputed or uncertain facts
- Use specific dates and time periods in your answer
- If some sub-queries returned no results, note what information is missing

Provide a comprehensive answer:
"""


# ---------------------------------------------------------------------------
# Stage 7: Dispute Resolution
# ---------------------------------------------------------------------------

DISPUTE_RESOLUTION_PROMPT = """The following relationships in the knowledge graph are marked as disputed — multiple sources provide conflicting information:

{disputed_edges}

For each dispute, evaluate:
1. The temporal context — which claim is more recent?
2. Source reliability — which source has a higher trust score?
3. Corroboration — does either claim have more supporting evidence?
4. Consistency — which claim is more consistent with other known facts?

Query context: {query}

Provide an analysis of each dispute and your assessment of which claim is most likely correct, along with your confidence level (0-1).
"""
