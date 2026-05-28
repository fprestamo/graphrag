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
You are a knowledge graph extraction engine. Given a text document, a list of entity types, and a document date ({document_date}), your job is to extract every meaningful entity and every meaningful relationship between entities, producing structured output suitable for loading into a knowledge graph.

Pay special attention to:
- TEMPORAL information: when relationships started, ended, or changed.
- COMPLETENESS: extract ALL entities and relationships present in the text, not just the most prominent ones.
- PRECISION: only extract what the text actually states or strongly implies. Do not hallucinate entities or relationships.

-Steps-

##############################################
STEP 1: ENTITY EXTRACTION
##############################################

Read the entire text carefully. Identify every entity that matches one of the allowed types: [{entity_types}]

For each entity, extract:
- entity_name: The canonical, most widely recognized name for this entity, IN ALL CAPS. Prefer short common names over full legal/formal names (e.g., "GOOGLE" not "ALPHABET INC. SUBSIDIARY GOOGLE LLC"; "ELON MUSK" not "ELON REEVE MUSK"). If the text uses an abbreviation and the full name, use whichever is more recognizable. SPELL THE NAME EXACTLY AS IT APPEARS IN THE TEXT — copy character-by-character (then uppercase). Never insert, drop, or substitute letters; never anglicize, transliterate, or "correct" the spelling. If the text says "Kromkamp" the entity_name is "KROMKAMP" — never "KROMPKAMP", "KROMKAM", or any other variant.
- entity_type: Exactly one type from the allowed list above. Each entity MUST have exactly one type — do not assign the same entity to multiple types.
- entity_description: A concise but comprehensive description (1–3 sentences) covering the entity's key attributes, roles, and relevance as described in the text. Include context that would help a reader understand why this entity matters in the document.

Format: ("entity"<|><entity_name><|><entity_type><|><entity_description>)

ENTITY EXTRACTION RULES:
1. De-duplicate aggressively: the same real-world entity must appear exactly ONCE, regardless of how the text refers to it. Collapse all of the following into a single entity:
   - Pronouns ("he", "she", "they", "it") referring back to a previously named entity.
   - Bare surname / last-name references after a full name was given (e.g. "Tabetha S. Boyajian" introduced in sentence 1, "Boyajian" used in later sentences → ONE entity "TABETHA BOYAJIAN", never two).
   - First-name-only references after a full name was given.
   - Definite-noun references ("the company", "the team", "the president") that point to a previously named entity.
   - Abbreviations and acronyms when the full form is also given (e.g. "PSV" and "PSV Eindhoven" → one entity).
   Choose the most complete form attested in the text as the canonical entity_name; do NOT mint a separate entity for every surface form.
2. One type per entity: Never create the same entity under two different types.
3. Be inclusive: Extract entities even if they appear only once, as long as they participate in a relationship.
4. Named entities only — NEVER extract bare generic terms. Do NOT extract common nouns like "HIGH SCHOOL", "CORPORATION", "PUBLISHING COMPANY", "ELECTION CAMPAIGN", "MULTINATIONAL INSURANCE COMPANY", "FOOD WRITER", "THE ECONOMY", or "TECHNOLOGY" on their own. Extract only the specific named instance (e.g. "SOUTHWEST HIGH SCHOOL", "COLCHESTER CORPORATION", "AXA"). If the text only refers to the entity by a generic noun and never names it, do not extract it.
5. Disambiguate look-alikes with different referents — when two names share words but refer to different real-world things, keep them as separate entities and reflect the distinction in the name and description. Examples that MUST stay separate:
   - A sports club named after a sponsor vs the sponsor itself (e.g. "KRUNG THAI BANK F.C." [club] vs "KRUNG THAI BANK" [bank]).
   - A government body that shares words with a club (e.g. "PORT AUTHORITY OF THAILAND" vs "THAI PORT FC").
   - A country vs its language/demonym (e.g. "ENGLAND" vs "ENGLISH", "FRANCE" vs "FRENCH").
6. Include year/edition in event names — for seasons, terms, elections, tournaments, and other recurring events, ALWAYS keep the year or edition qualifier as part of the entity_name (e.g. "2003-04 THAI LEAGUE T1 SEASON", "1996 NFL SEASON", "1989 HOLIDAY BOWL"). Different editions are different entities and must not collapse into the bare event name.

##############################################
STEP 2: RELATIONSHIP EXTRACTION
##############################################

From the entities identified in Step 1, identify every pair of (source_entity, target_entity) that the text states or strongly implies are related.

For each relationship, extract:
- source_entity: Name of the source entity (must exactly match an entity_name from Step 1).
- target_entity: Name of the target entity (must exactly match a DIFFERENT entity_name from Step 1). Self-loops are NEVER allowed.
- relationship_description: A plain-language explanation of how and why these two entities are related, including any relevant context, conditions, or qualifications mentioned in the text.
- relation_type: A short, canonical edge label in UPPER_SNAKE_CASE (see rules and canonical list below).
- relationship_strength: An integer from 1 to 10 indicating how central, direct, and well-evidenced this relationship is:
    10 = defining relationship (e.g., founder of a company)
     7 = strong, clearly stated relationship
     4 = moderate or indirect relationship
     1 = weak, implied, or tangential connection
- valid_time_start: When this relationship began (see temporal rules below).
- valid_time_end: When this relationship ended (see temporal rules below).

Format: ("relationship"<|><source_entity><|><target_entity><|><relationship_description><|><relation_type><|><relationship_strength><|><valid_time_start><|><valid_time_end>)

### RELATION TYPE RULES ###

The relation_type is a reusable, generic predicate — think of it as an edge label in a knowledge graph schema. It must NEVER contain entity names, proper nouns, specific dates, or sentence fragments.

### DIRECTION CONVENTION (MANDATORY) ###

Every relationship is a DIRECTED edge: source_entity → target_entity.

The source is the ACTOR / DOER / SUBJECT that performs the action.
The target is the RECIPIENT / OBJECT that the action is performed upon.

Examples of correct direction:
  PERSON  → ORGANIZATION : IS_CEO_OF   (person holds the role AT the org)
  COMPANY → COMPANY      : ACQUIRED    (acquirer → acquired)
  ORG     → PRODUCT      : DEVELOPED   (developer → thing developed)
  FIELD   → FIELD        : SUBFIELD_OF (child field → parent field, e.g. ML → AI)
  ORG     → TECHNOLOGY   : USES        (user → thing used)
  ORG     → GEO          : HEADQUARTERED_IN (org → location)
  PERSON  → ORG          : EMPLOYED_AT (employee → employer)

NEVER USE PASSIVE-VOICE RELATION TYPES. Always use the active form and place the actor/doer as source:
  ❌ USED_BY      → ✅ USES        (swap source↔target, user is source)
  ❌ FOUNDED_BY   → ✅ FOUNDED     (swap source↔target, founder is source)
  ❌ OWNED_BY     → ✅ OWNS        (swap source↔target, owner is source)
  ❌ POWERED_BY   → ✅ POWERS      (swap source↔target)
  ❌ FUNDED_BY    → ✅ FUNDS       (swap source↔target)
  ❌ DEVELOPED_BY → ✅ DEVELOPED   (swap source↔target)
  ❌ CREATED_BY   → ✅ CREATED     (swap source↔target)
  ❌ DESIGNED_BY  → ✅ DESIGNED    (swap source↔target)
  ❌ AUTHORED_BY  → ✅ AUTHORED    (swap source↔target)
  ❌ REGULATED_BY → ✅ REGULATES   (swap source↔target)
  ❌ ENFORCED_BY  → ✅ ENFORCES    (swap source↔target)
  ❌ CONTRACTED_BY→ ✅ CONTRACTED  (swap source↔target)
  ❌ ACQUIRED_BY  → ✅ ACQUIRED    (swap source↔target)

PREFER types from this canonical list whenever they fit:

  Leadership & Roles:
    IS_CEO_OF, IS_PRESIDENT_OF, IS_CHAIRMAN_OF, IS_CFO_OF, IS_CTO_OF, IS_COO_OF,
    IS_DIRECTOR_OF, IS_FOUNDER_OF, LEADS, APPOINTED_TO, MANAGES
    Direction: person → organization

  Employment & Membership:
    EMPLOYED_AT, WORKS_FOR, SERVES_ON, MEMBER_OF, RESIGNED_FROM,
    SUCCEEDED_BY, PRECEDED_BY, ADVISOR_TO
    Direction: person → organization/body

  Founding & Creation:
    FOUNDED, CO_FOUNDED, CREATED, ESTABLISHED
    Direction: creator → thing created

  Corporate & Financial:
    ACQUIRED, MERGED_WITH, INVESTED_IN, PARTNERED_WITH,
    SUBSIDIARY_OF, PARENT_OF, SPUN_OFF, LISTED_ON, SUPPLIES, CLIENT_OF,
    LICENSED_TO, COMPETES_WITH, FUNDS
    Direction: actor → target (acquirer→acquired, investor→investee, funder→funded)

  Location & Geography:
    HEADQUARTERED_IN, LOCATED_IN, OPERATES_IN, RELOCATED_TO, BASED_IN,
    BORDERS, ORIGINATED_FROM
    Direction: entity → place

  Products & Technology:
    DEVELOPED, RELEASED, PRODUCES, MANUFACTURES, LAUNCHED, USES, BUILT_ON,
    POWERS, INTEGRATES_WITH
    Direction: developer/user → product/technology

  Governance & Law:
    ENACTED, REGULATES, SIGNED, RATIFIED, PROPOSED, VETOED, ENFORCES,
    GOVERNS, AUTHORED, AMENDED, REPEALED, VIOLATES, COMPLIES_WITH
    Direction: actor/authority → subject/law

  Events & Activities:
    PARTICIPATED_IN, HOSTED, ORGANIZED, ATTENDED, ANNOUNCED, CAUSED,
    RESULTED_IN, TRIGGERED, OCCURRED_IN, PRESENTED_AT
    Direction: participant/cause → event/effect

  Affiliation & Social:
    AFFILIATED_WITH, COLLABORATED_WITH, SPONSORED, SUPPORTED, ENDORSED,
    OPPOSED, CRITICIZED, INFLUENCED, MENTORED, RELATED_TO
    Direction: actor → target

  Geopolitical:
    REPRESENTS, CITIZEN_OF, SANCTIONED, ALLIED_WITH, DECLARED_WAR_ON,
    NEGOTIATED_WITH, RECOGNIZED

  Education & Research:
    STUDIED_AT, GRADUATED_FROM, RESEARCHED, PUBLISHED, TEACHES_AT,
    AWARDED, CITED_BY, SUBFIELD_OF
    Direction: student/researcher → institution; child_field → parent_field

  Ownership & Attribution:
    OWNS, AUTHORED, NAMED_AFTER, DESIGNED
    Direction: owner/author/designer → thing owned/written/designed

If no canonical type fits, you may create a new one — but it MUST be:
  - Short (2–4 words max)
  - Generic and reusable (would apply to other entity pairs of the same kind)
  - In UPPER_SNAKE_CASE

✅ GOOD relation_types: IS_CEO_OF, ACQUIRED, HEADQUARTERED_IN, INVESTED_IN, PUBLISHED
❌ BAD relation_types (NEVER do these):
  - AMAZON_ACQUIRED_WHOLE_FOODS → contains entity names → use ACQUIRED
  - SERVES_AS_CEO_AND_CHAIRMAN_OF → too specific/verbose → use IS_CEO_OF (create separate relationship for IS_CHAIRMAN_OF)
  - LED_THE_DEVELOPMENT_OF_THE_NEW_PRODUCT → too long, has filler words → use DEVELOPED
  - LOCATED_IN_THE_NORTHEASTERN_PART_OF → contains descriptive detail → use LOCATED_IN (put detail in relationship_description)
  - USED_BY, OWNED_BY, FOUNDED_BY, POWERED_BY → passive voice → use USES, OWNS, FOUNDED, POWERS (swap source↔target)

### TEMPORAL RULES ###

The document was written/published on {document_date}. Use this as the anchor for all relative time expressions.

Mapping relative expressions to dates:
  "currently", "now", "as of today", "presently"  → valid_time_start = {document_date}, valid_time_end = ONGOING
  "recently"                                       → valid_time_start = approximate date near {document_date}, valid_time_end = ONGOING
  "last year"                                      → the calendar year before {document_date}
  "last month"                                     → the calendar month before {document_date}
  "since X" / "from X"                             → valid_time_start = X, valid_time_end = ONGOING
  "from X to Y" / "between X and Y"                → valid_time_start = X, valid_time_end = Y
  "until X" / "through X"                          → valid_time_end = X
  "in [year/month]"                                → valid_time_start = start of that period, valid_time_end = end of that period (or ONGOING if the relationship is continuing)
  "former", "ex-", "previously", "once"            → relationship has ended; valid_time_end should be BEFORE {document_date}
  "upcoming", "planned", "will"                    → valid_time_start = future date if given, else {document_date}; valid_time_end = UNKNOWN
  "Q1/Q2/Q3/Q4 [year]"                            → Q1=Jan-Mar, Q2=Apr-Jun, Q3=Jul-Sep, Q4=Oct-Dec

Date formatting:
  - Always use ISO format: YYYY-MM-DD
  - If only a year is known: use YYYY-01-01 for start, YYYY-12-31 for end
  - If only year and month are known: use YYYY-MM-01 for start, YYYY-MM-28/30/31 for end
  - Use "ONGOING" if the relationship is still active at the time of the document
  - Use "UNKNOWN" only when the text provides absolutely no temporal signal — but note that if no temporal information exists at all, default to valid_time_start = {document_date} and valid_time_end = ONGOING (i.e., assume the relationship holds at the time the document was written)

##############################################
STEP 3: OUTPUT FORMAT
##############################################

Return ALL entities and relationships as a single flat list, using **##** as the delimiter between items. Do not group entities and relationships separately — interleaving is fine, but listing all entities first then all relationships is also fine.

After the final item, output the completion marker: <|COMPLETE|>

##############################################
CRITICAL CONSTRAINTS (review before outputting)
##############################################
□ Every entity_name is IN ALL CAPS
□ Every entity has exactly ONE entity_type from the allowed list
□ No duplicate entities (same real-world thing listed twice)
□ Every relationship connects TWO DIFFERENT entities (no self-loops)
□ source_entity is the ACTOR/DOER, target_entity is the RECIPIENT/OBJECT — never reversed
□ NEVER use passive relation types (USED_BY, OWNED_BY, FOUNDED_BY, etc.) — always active voice
□ Every source_entity and target_entity in relationships exactly matches an entity_name from the entity list
□ Every relation_type is short UPPER_SNAKE_CASE with no entity names or proper nouns embedded
□ Every relationship has valid temporal fields (valid_time_start and valid_time_end)
□ Relationship strength is an integer 1–10
□ Output uses the correct delimiters: <|> between fields, ## between items

######################
-Examples-
######################

Example 1: Corporate & Technology
Entity_types: ORGANIZATION, PERSON, PRODUCT, GEO
Document_date: 2024-03-10
Text:
Microsoft completed its $69 billion acquisition of Activision Blizzard in October 2023 after receiving regulatory approval from the UK's Competition and Markets Authority. CEO Satya Nadella called it a "landmark moment for gaming." Activision's former CEO Bobby Kotick stepped down in December 2023. The combined gaming division, now based in Redmond, Washington, oversees franchises including Call of Duty and World of Warcraft.
######################
Output:
("entity"<|>MICROSOFT<|>ORGANIZATION<|>Microsoft is a major technology corporation that acquired Activision Blizzard in a $69 billion deal completed in October 2023)
##
("entity"<|>ACTIVISION BLIZZARD<|>ORGANIZATION<|>Activision Blizzard is a video game publisher acquired by Microsoft in October 2023, known for franchises like Call of Duty and World of Warcraft)
##
("entity"<|>COMPETITION AND MARKETS AUTHORITY<|>ORGANIZATION<|>The UK's Competition and Markets Authority (CMA) is the regulatory body that granted approval for the Microsoft-Activision deal)
##
("entity"<|>SATYA NADELLA<|>PERSON<|>Satya Nadella is the CEO of Microsoft who described the Activision acquisition as a landmark moment for gaming)
##
("entity"<|>BOBBY KOTICK<|>PERSON<|>Bobby Kotick is the former CEO of Activision Blizzard who stepped down in December 2023 following the Microsoft acquisition)
##
("entity"<|>CALL OF DUTY<|>PRODUCT<|>Call of Duty is a major gaming franchise owned by Activision Blizzard, now under Microsoft's gaming division)
##
("entity"<|>WORLD OF WARCRAFT<|>PRODUCT<|>World of Warcraft is a major gaming franchise owned by Activision Blizzard, now under Microsoft's gaming division)
##
("entity"<|>REDMOND<|>GEO<|>Redmond, Washington is the location of Microsoft's combined gaming division headquarters)
##
("relationship"<|>MICROSOFT<|>ACTIVISION BLIZZARD<|>Microsoft completed a $69 billion acquisition of Activision Blizzard in October 2023<|>ACQUIRED<|>10<|>2023-10-01<|>ONGOING)
##
("relationship"<|>COMPETITION AND MARKETS AUTHORITY<|>MICROSOFT<|>The CMA granted regulatory approval for Microsoft's acquisition of Activision Blizzard<|>REGULATED_BY<|>7<|>2023-10-01<|>2023-10-31)
##
("relationship"<|>SATYA NADELLA<|>MICROSOFT<|>Satya Nadella serves as CEO of Microsoft<|>IS_CEO_OF<|>10<|>2014-02-04<|>ONGOING)
##
("relationship"<|>BOBBY KOTICK<|>ACTIVISION BLIZZARD<|>Bobby Kotick served as CEO of Activision Blizzard before stepping down in December 2023<|>IS_CEO_OF<|>9<|>UNKNOWN<|>2023-12-31)
##
("relationship"<|>ACTIVISION BLIZZARD<|>CALL OF DUTY<|>Activision Blizzard owns and publishes the Call of Duty franchise<|>PRODUCES<|>9<|>UNKNOWN<|>ONGOING)
##
("relationship"<|>ACTIVISION BLIZZARD<|>WORLD OF WARCRAFT<|>Activision Blizzard owns and publishes the World of Warcraft franchise<|>PRODUCES<|>9<|>UNKNOWN<|>ONGOING)
##
("relationship"<|>MICROSOFT<|>REDMOND<|>Microsoft's combined gaming division is based in Redmond, Washington<|>HEADQUARTERED_IN<|>7<|>2023-10-01<|>ONGOING)
<|COMPLETE|>

######################
Example 2: Geopolitics & Policy
Entity_types: ORGANIZATION, PERSON, GEO, EVENT, LAW
Document_date: 2024-07-15
Text:
In June 2024, the African Union convened the Nairobi Climate Summit to address the impact of drought across the Horn of Africa. Ethiopian Prime Minister Abiy Ahmed and Kenyan President William Ruto co-chaired the event. The summit resulted in the Nairobi Green Compact, a binding agreement requiring member nations to reduce emissions 30% by 2035. China pledged $2 billion in green infrastructure funding during the summit, while the United States sent a delegation led by Climate Envoy John Podesta but made no financial commitments.
######################
Output:
("entity"<|>AFRICAN UNION<|>ORGANIZATION<|>The African Union is a continental body that convened the Nairobi Climate Summit in June 2024 to address drought and emissions across the Horn of Africa)
##
("entity"<|>NAIROBI CLIMATE SUMMIT<|>EVENT<|>The Nairobi Climate Summit was a major climate conference held in June 2024, co-chaired by the leaders of Ethiopia and Kenya, resulting in the Nairobi Green Compact)
##
("entity"<|>NAIROBI<|>GEO<|>Nairobi is the capital of Kenya and the host city of the June 2024 climate summit)
##
("entity"<|>HORN OF AFRICA<|>GEO<|>The Horn of Africa is a region affected by drought, which was a central topic of the Nairobi Climate Summit)
##
("entity"<|>ETHIOPIA<|>GEO<|>Ethiopia is an African nation whose Prime Minister Abiy Ahmed co-chaired the Nairobi Climate Summit)
##
("entity"<|>KENYA<|>GEO<|>Kenya is an African nation whose President William Ruto co-chaired the Nairobi Climate Summit)
##
("entity"<|>ABIY AHMED<|>PERSON<|>Abiy Ahmed is the Prime Minister of Ethiopia who co-chaired the Nairobi Climate Summit in June 2024)
##
("entity"<|>WILLIAM RUTO<|>PERSON<|>William Ruto is the President of Kenya who co-chaired the Nairobi Climate Summit in June 2024)
##
("entity"<|>NAIROBI GREEN COMPACT<|>LAW<|>The Nairobi Green Compact is a binding agreement resulting from the 2024 Nairobi Climate Summit, requiring a 30% emissions reduction by 2035)
##
("entity"<|>CHINA<|>GEO<|>China pledged $2 billion in green infrastructure funding at the Nairobi Climate Summit)
##
("entity"<|>UNITED STATES<|>GEO<|>The United States sent a delegation to the Nairobi Climate Summit but made no financial commitments)
##
("entity"<|>JOHN PODESTA<|>PERSON<|>John Podesta is the US Climate Envoy who led the American delegation to the Nairobi Climate Summit)
##
("relationship"<|>AFRICAN UNION<|>NAIROBI CLIMATE SUMMIT<|>The African Union organized and convened the Nairobi Climate Summit in June 2024<|>ORGANIZED<|>10<|>2024-06-01<|>2024-06-30)
##
("relationship"<|>NAIROBI CLIMATE SUMMIT<|>NAIROBI<|>The Nairobi Climate Summit was held in Nairobi<|>OCCURRED_IN<|>8<|>2024-06-01<|>2024-06-30)
##
("relationship"<|>ABIY AHMED<|>NAIROBI CLIMATE SUMMIT<|>Abiy Ahmed co-chaired the Nairobi Climate Summit<|>LEADS<|>9<|>2024-06-01<|>2024-06-30)
##
("relationship"<|>WILLIAM RUTO<|>NAIROBI CLIMATE SUMMIT<|>William Ruto co-chaired the Nairobi Climate Summit<|>LEADS<|>9<|>2024-06-01<|>2024-06-30)
##
("relationship"<|>ABIY AHMED<|>ETHIOPIA<|>Abiy Ahmed serves as Prime Minister of Ethiopia<|>IS_PRESIDENT_OF<|>9<|>2018-04-02<|>ONGOING)
##
("relationship"<|>WILLIAM RUTO<|>KENYA<|>William Ruto serves as President of Kenya<|>IS_PRESIDENT_OF<|>9<|>2022-09-13<|>ONGOING)
##
("relationship"<|>NAIROBI CLIMATE SUMMIT<|>NAIROBI GREEN COMPACT<|>The Nairobi Climate Summit resulted in the Nairobi Green Compact agreement<|>RESULTED_IN<|>10<|>2024-06-01<|>ONGOING)
##
("relationship"<|>CHINA<|>NAIROBI CLIMATE SUMMIT<|>China participated in the summit and pledged $2 billion in green infrastructure funding<|>PARTICIPATED_IN<|>8<|>2024-06-01<|>2024-06-30)
##
("relationship"<|>UNITED STATES<|>NAIROBI CLIMATE SUMMIT<|>The United States sent a delegation to the summit but made no financial commitments<|>PARTICIPATED_IN<|>5<|>2024-06-01<|>2024-06-30)
##
("relationship"<|>JOHN PODESTA<|>UNITED STATES<|>John Podesta serves as Climate Envoy representing the United States<|>REPRESENTS<|>8<|>2024-07-15<|>ONGOING)
##
("relationship"<|>JOHN PODESTA<|>NAIROBI CLIMATE SUMMIT<|>John Podesta led the US delegation at the Nairobi Climate Summit<|>ATTENDED<|>7<|>2024-06-01<|>2024-06-30)
<|COMPLETE|>

######################
Example 3: Science, Education & Multi-sector
Entity_types: ORGANIZATION, PERSON, GEO, EVENT, PRODUCT
Document_date: 2025-01-20
Text:
Dr. Amara Osei, a former professor at MIT, joined DeepMind in London last year to lead a new protein-folding research lab. The lab builds on AlphaFold, which DeepMind released in 2020. Her team recently published a breakthrough paper in Nature showing a 40% improvement in prediction accuracy. Osei previously collaborated with the Max Planck Institute in Munich from 2018 to 2022 on related structural biology research.
######################
Output:
("entity"<|>AMARA OSEI<|>PERSON<|>Dr. Amara Osei is a scientist and former MIT professor who joined DeepMind in 2024 to lead a protein-folding research lab, recently publishing a breakthrough in Nature)
##
("entity"<|>MIT<|>ORGANIZATION<|>MIT (Massachusetts Institute of Technology) is a leading research university where Amara Osei previously served as a professor)
##
("entity"<|>DEEPMIND<|>ORGANIZATION<|>DeepMind is an AI research lab based in London, known for AlphaFold, where Amara Osei now leads a protein-folding research lab)
##
("entity"<|>LONDON<|>GEO<|>London is the city where DeepMind is based and where Amara Osei's new lab is located)
##
("entity"<|>ALPHAFOLD<|>PRODUCT<|>AlphaFold is DeepMind's protein structure prediction system, released in 2020, which forms the foundation of Osei's new research lab)
##
("entity"<|>NATURE<|>ORGANIZATION<|>Nature is a leading scientific journal where Osei's team published a breakthrough paper on protein-folding prediction accuracy)
##
("entity"<|>MAX PLANCK INSTITUTE<|>ORGANIZATION<|>The Max Planck Institute is a research organization in Munich that collaborated with Amara Osei on structural biology research from 2018 to 2022)
##
("entity"<|>MUNICH<|>GEO<|>Munich is the city where the Max Planck Institute is located)
##
("relationship"<|>AMARA OSEI<|>MIT<|>Amara Osei was formerly a professor at MIT before joining DeepMind<|>TEACHES_AT<|>7<|>UNKNOWN<|>2024-01-01)
##
("relationship"<|>AMARA OSEI<|>DEEPMIND<|>Amara Osei joined DeepMind in 2024 to lead a new protein-folding research lab<|>EMPLOYED_AT<|>9<|>2024-01-01<|>ONGOING)
##
("relationship"<|>DEEPMIND<|>LONDON<|>DeepMind is based in London<|>HEADQUARTERED_IN<|>7<|>UNKNOWN<|>ONGOING)
##
("relationship"<|>DEEPMIND<|>ALPHAFOLD<|>DeepMind developed and released AlphaFold in 2020<|>DEVELOPED<|>10<|>2020-01-01<|>ONGOING)
##
("relationship"<|>AMARA OSEI<|>NATURE<|>Amara Osei's team recently published a breakthrough protein-folding paper in Nature<|>PUBLISHED<|>8<|>2025-01-01<|>2025-01-20)
##
("relationship"<|>AMARA OSEI<|>MAX PLANCK INSTITUTE<|>Amara Osei collaborated with the Max Planck Institute on structural biology research from 2018 to 2022<|>COLLABORATED_WITH<|>7<|>2018-01-01<|>2022-12-31)
##
("relationship"<|>MAX PLANCK INSTITUTE<|>MUNICH<|>The Max Planck Institute is located in Munich<|>LOCATED_IN<|>6<|>UNKNOWN<|>ONGOING)
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

CARDINALITY_CLASSIFICATION_PROMPT = """You are a knowledge graph ontology expert. Given a relationship type and example edges from the graph, classify its structural cardinality constraint.

Relationship Type: {relation_type}
Context examples from the graph:
{context_examples}

The four cardinality types are:

1. **SUBJECT_EXCLUSIVE**: One subject can hold at most one active instance of this relation at a time.
   Examples:
   - HEADQUARTERED_IN (an organization has one HQ at a time — e.g., Google DeepMind is headquartered in one city)
   - LEADS (a research lab has one primary lead at a time)
   - BASED_IN (an org is based in one location at a time)
   - IS_PRIMARY_QUERY_LANGUAGE_OF (a database system has one primary query language — e.g., SPARQL for RDF stores)
   - RELOCATED_TO (one active relocation destination at a time)

2. **OBJECT_EXCLUSIVE**: One object can have at most one active subject for this relation at a time.
   Examples:
   - IS_CEO_OF (one CEO per company at a time, but a person can be CEO of multiple companies simultaneously)
   - IS_CHIEF_AI_SCIENTIST_OF (one chief AI scientist per org at a time — e.g., Yann LeCun at Meta)
   - IS_DIRECTOR_OF (one director per lab at a time, but a person can direct multiple labs)
   - IS_PRIMARY_LANGUAGE_OF (one primary language per framework)

3. **BOTH_EXCLUSIVE**: One-to-one constraint on BOTH sides simultaneously. One subject → one object AND one object → one subject at any point in time.
   Examples:
   - IS_PRESIDENT_OF (one president per country AND one presidency per person at a time)
   - IS_LEAD_RESEARCHER_OF (one lead researcher per project AND one lead project per researcher at a time)
   - IS_GOVERNOR_OF (one governor per state AND one governorship per person at a time)

4. **NON_EXCLUSIVE**: No exclusivity constraint; multiple instances can coexist freely.
   Examples:
   - COLLABORATED_WITH, PARTICIPATED_IN, MEMBER_OF, AFFILIATED_WITH
   - FOUNDED, CO_FOUNDED, ACQUIRED, PARTNERED_WITH
   - DEVELOPED, PUBLISHED, AUTHORED, RELEASED
   - SUBFIELD_OF, BUILT_ON, USES, INTEGRATES_WITH
   - EMPLOYED_AT, WORKS_FOR, SERVES_ON
   - RESEARCHED, STUDIED_AT, GRADUATED_FROM, TEACHES_AT

IMPORTANT: Most relationship types are NON_EXCLUSIVE. Only classify as exclusive if there is a clear real-world constraint that prevents multiple active instances. When in doubt, choose NON_EXCLUSIVE.

## Classification Examples

Example 1:
Relationship Type: IS_LEAD_RESEARCHER_OF
Context examples from the graph:
  (Geoffrey Hinton) -[IS_LEAD_RESEARCHER_OF]-> (Google Brain): Geoffrey Hinton led deep learning research at Google Brain
  (Demis Hassabis) -[IS_LEAD_RESEARCHER_OF]-> (DeepMind): Demis Hassabis leads research at DeepMind
Answer: BOTH_EXCLUSIVE
Reason: A research lab has one lead researcher at a time (object-side), and a person leads one lab at a time (subject-side).

Example 2:
Relationship Type: DEVELOPED
Context examples from the graph:
  (Google DeepMind) -[DEVELOPED]-> (AlphaGo): Google DeepMind developed the AlphaGo system
  (Google DeepMind) -[DEVELOPED]-> (Gemini): Google DeepMind developed the Gemini LLM
  (OpenAI) -[DEVELOPED]-> (GPT-4): OpenAI developed GPT-4
Answer: NON_EXCLUSIVE
Reason: An organization can develop multiple products simultaneously; no exclusivity constraint.

Example 3:
Relationship Type: HEADQUARTERED_IN
Context examples from the graph:
  (OpenAI) -[HEADQUARTERED_IN]-> (San Francisco): OpenAI is headquartered in San Francisco
  (Anthropic) -[HEADQUARTERED_IN]-> (San Francisco): Anthropic is headquartered in San Francisco
Answer: SUBJECT_EXCLUSIVE
Reason: An organization has one headquarters at a time (subject-side), but multiple organizations can share the same HQ city (e.g., OpenAI and Anthropic are both in San Francisco).

Example 4:
Relationship Type: IS_CHIEF_AI_SCIENTIST_OF
Context examples from the graph:
  (Yann LeCun) -[IS_CHIEF_AI_SCIENTIST_OF]-> (Meta): Yann LeCun serves as VP & Chief AI Scientist at Meta
Answer: OBJECT_EXCLUSIVE
Reason: An organization has one Chief AI Scientist at a time (object-side), but a person could theoretically hold such a title at multiple organizations.

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

TEMPORAL_QUERY_ANALYSIS_PROMPT = """You are a temporal reasoning expert. Your task is to analyse a question against a bitemporal knowledge graph and prepare it for retrieval.

Original Query: {query}
Current Date: {current_date}

Do TWO things and return a single JSON object:

1. ENTITY EXTRACTION
   List every named entity the question is asking ABOUT or that must be located in the graph to answer it (people, organisations, places, events, products, laws, etc.). Use the entity's most recognisable surface form (e.g. "Attaphol Buspakom", "Apple", "Nairobi Climate Summit"). Do NOT include relation types, dates, or generic nouns.

2. TEMPORAL DECOMPOSITION
   Decompose the question into one or more sub-queries that each carry an explicit temporal constraint. Most questions only need ONE sub-query — only emit multiple when the question references DIFFERENT independent time points or periods (e.g. "who led X when it merged with Y" → one sub-query about the merger date, one about leadership at that date).

   For each sub-query, give:
   - "sub_query": natural-language sub-question
   - "entities": entities relevant to THIS sub-query (subset of the global list)
   - "query_type": one of
       * POINT_IN_TIME — about a single instant ("in 1992", "in March 2014", "when X resigned")
       * RANGE         — about an interval ("between Apr 1987 and Nov 1988", "during the 1990s")
       * EVOLUTION     — about a sequence of changes over time ("how did X evolve", "list every CEO of X")
       * COMPARISON    — contrasts two different time points ("compare X in 2010 vs 2020")
   - "t_start": ISO date (YYYY-MM-DD) for the start of the constraint, or "CURRENT" if the question is about now, or "UNKNOWN" if no temporal anchor is present.
   - "t_end": ISO date (YYYY-MM-DD) for the end, or "CURRENT", or same as t_start for POINT_IN_TIME, or "UNKNOWN".

OUTPUT FORMAT (strict JSON, no markdown, no commentary):
{{
    "entities": ["<entity 1>", "<entity 2>"],
    "sub_queries": [
        {{
            "sub_query": "<sub-query text>",
            "entities": ["<entity 1>"],
            "query_type": "POINT_IN_TIME|RANGE|EVOLUTION|COMPARISON",
            "t_start": "YYYY-MM-DD|CURRENT|UNKNOWN",
            "t_end":   "YYYY-MM-DD|CURRENT|UNKNOWN"
        }}
    ]
}}

## Examples

Q: "Which team did Attaphol Buspakom play for between Apr 1987 and Nov 1988?"
A:
{{
    "entities": ["Attaphol Buspakom"],
    "sub_queries": [
        {{
            "sub_query": "Which team did Attaphol Buspakom play for between Apr 1987 and Nov 1988?",
            "entities": ["Attaphol Buspakom"],
            "query_type": "RANGE",
            "t_start": "1987-04-01",
            "t_end":   "1988-11-30"
        }}
    ]
}}

Q: "Who was the CEO of Apple when Steve Jobs returned in 1997?"
A:
{{
    "entities": ["Apple", "Steve Jobs"],
    "sub_queries": [
        {{
            "sub_query": "Who was the CEO of Apple in 1997?",
            "entities": ["Apple"],
            "query_type": "POINT_IN_TIME",
            "t_start": "1997-01-01",
            "t_end":   "1997-12-31"
        }}
    ]
}}

Q: "List every chairman of Microsoft."
A:
{{
    "entities": ["Microsoft"],
    "sub_queries": [
        {{
            "sub_query": "List every chairman of Microsoft throughout its history.",
            "entities": ["Microsoft"],
            "query_type": "EVOLUTION",
            "t_start": "UNKNOWN",
            "t_end":   "CURRENT"
        }}
    ]
}}

Q: "Who was the occupant of Anmer Hall before Mar 1897?"
A:
{{
    "entities": ["Anmer Hall"],
    "sub_queries": [
        {{
            "sub_query": "Who was the occupant of Anmer Hall before March 1897?",
            "entities": ["Anmer Hall"],
            "query_type": "RANGE",
            "t_start": "UNKNOWN",
            "t_end":   "1897-03-01"
        }}
    ]
}}

Q: "Tabetha S. Boyajian went to which school after Aug 2005?"
A:
{{
    "entities": ["Tabetha S. Boyajian"],
    "sub_queries": [
        {{
            "sub_query": "Which school did Tabetha S. Boyajian attend after August 2005?",
            "entities": ["Tabetha S. Boyajian"],
            "query_type": "RANGE",
            "t_start": "2005-08-01",
            "t_end":   "CURRENT"
        }}
    ]
}}

Now analyse the question above and return the JSON object.
"""

# Backward-compat alias — older code paths may still import the old name.
TEMPORAL_QUERY_DECOMPOSITION_PROMPT = TEMPORAL_QUERY_ANALYSIS_PROMPT


# ---------------------------------------------------------------------------
# Stage 7: Temporal Answer Synthesis
# ---------------------------------------------------------------------------

TEMPORAL_ANSWER_SYNTHESIS_PROMPT = """You answer temporal questions over a bitemporal knowledge graph, with help from a non-temporal text retriever.

Edge format:
  (SOURCE) -[RELATION]-> (TARGET) [valid_start → valid_end] {{status, support_count, confidence}}: description

Notes: dates were extracted from text and are approximate. UNKNOWN = earlier than recorded; ONGOING = still true. Ignore retracted edges unless the question is about history. The description often carries the precise role / position / place that answers the question — read it; it overrides a generic relation name.

Per sub-query you receive:
- **PRIMARY** edges — validity overlaps the sub-query's window, pre-ranked (anchor contained / start near anchor for POINT_IN_TIME / narrower / closer in time / more reliable).
- **BACKGROUND** edges — out of window, for disambiguation.
- **BASELINE EVIDENCE** (when present) — a text-grounded answer the non-temporal retriever produced after reading the source passage directly. No date filter.

Question: {query}

{community_context}{sub_results}

{baseline_block}First **judge whether each source actually answers the question** — not just whether it mentions the entity. Then write the best single-entity answer.

Reconciling sources:
- PRIMARY is strongest for **WHEN** something was true (only PRIMARY filters by the window).
- BASELINE is strongest for **WHO / WHAT / WHICH** (it reads the source text; PRIMARY's extracted dates can be off by months or years and pull in a wrong-but-in-window edge).
- When PRIMARY and BASELINE name different entities for the same slot, prefer BASELINE. Override only if *multiple* PRIMARY edges agree on the alternative.
- When PRIMARY is empty / out of window, answer from BASELINE. Don't refuse if BASELINE has a confident answer.

Reply in one short sentence. For atomic questions ("which team / what position / who / where / which employer"), name exactly ONE entity — drop extras PRIMARY may list. Quote dates verbatim. Only say "the information is missing" if every source genuinely fails.

ANSWER:
"""


# ---------------------------------------------------------------------------
# Stage 7: Dispute Resolution
# ---------------------------------------------------------------------------

DISPUTE_RESOLUTION_PROMPT = """The following relationships in the knowledge graph are marked as `disputed` — the indexing pipeline (ETCDR) could not commit to a single resolution, so multiple competing versions co-exist:

{disputed_edges}

Each edge carries:
- a valid period [t_valid_start, t_valid_end),
- a transaction timestamp t_tx_start (when the system came to believe it),
- a confidence value in [0, 1],
- a support_count (how many sources corroborate it).

Query context: {query}

For each dispute, decide which claim is most likely correct by weighting:
  score = confidence * log(1 + support_count) * recency(t_tx_start)
The most recent transaction breaks ties.

Output, for each dispute group, JSON of the form:
{{
    "winner_edge_id": "<id>",
    "loser_edge_ids": ["<id>", ...],
    "rationale": "<one sentence>",
    "confidence": <float in [0,1]>
}}
"""


# ---------------------------------------------------------------------------
# Stage 6(b) — Agent loop
# ---------------------------------------------------------------------------
#
# The temporal_query pipeline is implemented as a small agent: the LLM is
# given a toolbox and a question, and decides — one step at a time — which
# tool to call next, until it has enough evidence to answer. No native
# function-calling: every turn the model emits a JSON object describing
# either a tool invocation or the final answer.

TEMPORAL_AGENT_SYSTEM_PROMPT = """You answer temporal questions about people, organisations, events, places, etc., using a bitemporal knowledge graph and a non-temporal text retriever.

Today's date: {current_date}.

You operate as a step-by-step agent. At each turn you MUST output a single JSON object with ONE of these shapes:

  {{"tool": "<tool_name>", "args": {{...}}}}     ← invoke a tool
  {{"answer": "<one short sentence>"}}            ← finish and reply to the user

No prose, no markdown, no commentary. Only the JSON object.

You have at most {max_iters} tool turns. Use them. Repeated identical tool calls are wasted turns — vary the arguments or try a different tool.

────────────────────────────  TOOLS  ────────────────────────────

1) resolve_entities — Look up entities by name (title match + embedding NN over entity descriptions).
   args:
     {{"names": ["<string>", ...],            // 1-3 candidate names from the question
       "top_k": <int, default 3>}}
   returns: list of {{title, type, description}}. Use the returned `title` (uppercased) for later tool calls.

2) time_window_search — Bitemporal as-of subgraph: edges around the given entities whose validity *overlaps* the requested window.
   args:
     {{"entity_titles": ["<TITLE>", ...],
       "t_start":  "YYYY-MM-DD" | null,        // null = open past
       "t_end":    "YYYY-MM-DD" | null,        // null = today
       "k_hop":    <int, default 1, max 2>,
       "limit":    <int, default 12>}}
   returns: edges formatted as
     (SOURCE) -[RELATION]-> (TARGET) [valid_start → valid_end] {{status, support, conf}}: description

3) wide_search — Same as (2) but WITHOUT the temporal filter. Use when the time window misses (extracted bounds are often noisy) or to map the entity's full neighbourhood.
   args:
     {{"entity_titles": ["<TITLE>", ...],
       "k_hop":  <int, default 1, max 2>,
       "limit":  <int, default 15>}}

4) search_edges_by_description — **Semantic search over relationship descriptions.** Embeds your description text and returns the top-K edges whose own description is closest. Bypasses entity-resolution entirely: useful when the question wording mentions the role, position, or event directly (e.g. "chair of Norwegian Association in 1898", "broadcaster of Beugró", "joined the Royal Society in 1908"). Often surfaces the exact edge that answers the question even when resolve_entities + time_window_search returned nothing.
   args:
     {{"description": "<natural-language phrase summarising the edge you want>",
       "top_k": <int, default 10, max 20>}}
   returns: same edge format as (2).

5) text_search — Ask a non-temporal text retriever that reads the source passages directly. Returns one sentence/paragraph grounded in the underlying text. **Authoritative for WHO / WHAT / WHICH identity questions.** Has no temporal filter, so its dates may be loose — but its named entities (person, team, place, organisation) come straight from the text and are usually right when graph extraction was noisy.
   args:
     {{"question": "<self-contained question, full phrasing>"}}
   returns: string. (Available at most twice per question.)

6) final_answer — Done. Stop the loop.
   args:
     {{"answer": "<one short sentence>"}}

────────────────────────  HOW TO REASON  ────────────────────────

The graph's extracted dates AND extracted entity names are *approximate*. Multiple graph edges can be consistently wrong because they all came from the same noisy extraction batch. Apparent consensus across graph edges is therefore NOT proof — you must confirm identity with `text_search` (which reads the source passage) before answering atomic identity questions.

Default strategy:

  Step 1. `resolve_entities` on the explicit name(s) in the question.

  Step 2. `time_window_search` over the resolved titles with the question's time bounds.
          This gives you a candidate answer from the graph.

  Step 3. `search_edges_by_description` with a phrase paraphrasing what you're looking for
          (e.g. "chair of Norwegian Association for Women's Rights in 1898",
                "team Steve Guppy played for in April 2006",
                "official name of VP-5 in August 1946").
          Often surfaces the exact edge directly when entity-resolution missed.

  Step 4. **`text_search` is MANDATORY before answering any atomic identity question**
          (which team / what position / who / where / which employer / which broadcaster /
           which capital / which spouse / which rank / which party / which organisation / which name).
          Phrase the call as a complete self-contained question.
          The text retriever reads the source passage and is the authoritative source for
          WHO / WHAT / WHICH identity. Apparent consensus across graph edges does NOT replace it.

  Step 5. Decide:
          • If `text_search` and the graph agree → answer with that entity.
          • If they disagree → **prefer `text_search`'s identity** unless TWO OR MORE INDEPENDENT
            graph edges (different relation_types or different descriptions, not duplicates from
            the same extraction) clearly name the alternative.
          • If `text_search` returned empty AND the graph has a clear candidate → answer with
            the graph candidate. If both empty/unclear → use remaining turns: try
            `search_edges_by_description` with a different phrasing, or `wide_search` to scan
            the entity's full neighbourhood, or `text_search` with a sharper rewording.

Hard rules (apply at every step):

  • **NEVER refuse or hedge.** Your final answer must commit to a SINGLE concrete entity
    (a name / role / place / organisation), never phrases like "the information is not
    available", "no record found", "data does not provide", "did not hold any position",
    "cannot be determined". Even when every source is sparse or contradictory, look at ALL
    the evidence you have gathered (resolved entities, time_window_search edges,
    search_edges_by_description edges, text_search responses including their *content* even
    when they hedge) and pick the best-supported single entity. The text retriever
    occasionally hedges in its prose ("there is no data...") while still mentioning the
    answer in the same sentence — extract the named entity, don't echo the hedge.
  • **For atomic identity questions: ALWAYS call `text_search` before answering**, even when
    `time_window_search` and `search_edges_by_description` look consistent. Multiple graph edges
    naming the same entity can all be wrong from the same extraction error. `text_search` is the
    independent check.
  • Don't verify graph with more graph: re-querying `time_window_search` on the candidate answer
    entity is not independent confirmation — it draws from the same extraction batch.
  • For atomic identity questions, name exactly ONE entity in the final answer; drop extras the
    edges may list (parent companies, secondary roles, transitions). Pick the most specific name
    that matches the question's frame (e.g. the subsidiary, not the parent corporation).
  • Quote dates verbatim as they appear in the source.

Commitment over caution: a wrong guess and a hedged refusal score the same, but a wrong guess
that happens to be a partial match can still be judged correct. Always commit.
"""


TEMPORAL_AGENT_USER_PROMPT = """QUESTION: {query}

Begin. Output your first JSON action now."""


TEMPORAL_AGENT_FORCE_ANSWER_PROMPT = """You have used all your tool turns. Output your best final answer now as JSON:
  {{"answer": "<one short sentence>"}}
"""
