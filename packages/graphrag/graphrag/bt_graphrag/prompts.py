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
- entity_name: The canonical, most widely recognized name for this entity, IN ALL CAPS. Prefer short common names over full legal/formal names (e.g., "GOOGLE" not "ALPHABET INC. SUBSIDIARY GOOGLE LLC"; "ELON MUSK" not "ELON REEVE MUSK"). If the text uses an abbreviation and the full name, use whichever is more recognizable.
- entity_type: Exactly one type from the allowed list above. Each entity MUST have exactly one type — do not assign the same entity to multiple types.
- entity_description: A concise but comprehensive description (1–3 sentences) covering the entity's key attributes, roles, and relevance as described in the text. Include context that would help a reader understand why this entity matters in the document.

Format: ("entity"<|><entity_name><|><entity_type><|><entity_description>)

ENTITY EXTRACTION RULES:
1. De-duplicate: If the same entity is mentioned by different names or aliases (e.g., "the company" referring to "Acme Corp"), extract it only once under its canonical name.
2. One type per entity: Never create the same entity under two different types.
3. Be inclusive: Extract entities even if they appear only once, as long as they participate in a relationship.
4. Named entities only: Do not extract generic concepts ("the economy", "technology") unless they are a named, specific thing (e.g., "THE INFLATION REDUCTION ACT", "BITCOIN").

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
