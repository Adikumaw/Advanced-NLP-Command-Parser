"""
This script provides an advanced natural language command parser. It uses the
Stanza NLP library to perform deep linguistic analysis, including dependency
parsing and named entity recognition (NER). The goal is to convert a user's
command into a structured JSON object that details the primary action, its
parameters, and their semantic roles.

The script features an interactive shell for real-time command processing.
"""

import stanza
import sys
import json
from datetime import datetime, timedelta, time
from dateparser.search import search_dates

# --- Configuration ---
# Use 'gpu=False' if you don't have a compatible GPU or CUDA installed.
# The 'tokenize,pos,lemma,depparse,ner' processors are essential for the parsing logic.
PROCESSORS = 'tokenize,pos,lemma,depparse,ner'
NLP_PIPELINE = None # We will initialize this once, globally

def initialize_nlp():
    """Initializes the Stanza pipeline. To be called once on startup."""
    global NLP_PIPELINE
    if NLP_PIPELINE is None:
        print("Initializing Stanza NLP Pipeline... (This may take a moment)")
        NLP_PIPELINE = stanza.Pipeline('en', processors=PROCESSORS, use_gpu=False, verbose=False)
        print("NLP Pipeline Ready.")

# This will be the main entry point function for our service
def process_text(text: str):
    """
    Takes a raw text string, processes it with the NLP pipeline,
    and returns the structured JSON.
    """
    if NLP_PIPELINE is None:
        # This is a fallback in case initialization failed, but it shouldn't happen.
        return {"error": "NLP Pipeline not initialized."}
    
    doc = NLP_PIPELINE(text)
    structured_output = parse_query_to_structured_json(doc)
    
    return structured_output

# --- Helper Functions to Build JSON Parts ---

def extract_metadata(doc):
    """
    Extracts high-level metadata from the processed Stanza document.

    Args:
        doc (stanza.Document): The processed Stanza document object.

    Returns:
        dict: A dictionary containing metadata such as the original text,
              sentence count, and a timestamp.
    """
    metadata = {
        "source_text": doc.text,
        "num_sentences": len(doc.sentences),
        "timestamp": datetime.now().isoformat(),
    }
    return metadata

def extract_pragmatics(doc):
    """
    Analyzes the overall intent and style of the query.

    This is a simplified, rule-based implementation. A production system might
    use a dedicated intent classification model. It infers intent (Question,
    Request, Command) and modality based on root verbs and auxiliary words.

    Args:
        doc (stanza.Document): The processed Stanza document object.

    Returns:
        dict: A dictionary containing pragmatic analysis, including intent,
              sentiment, and style.
    """
    pragmatics = {
        "intent_primary": "UNKNOWN",
        "sentiment": "neutral",
        "emotion": None,
        "style": {
            "formality": "neutral",
            "politeness": 5,
            "modality": None
        }
    }

    # Simple rule-based logic for intent and modality
    if not doc.sentences:
        return pragmatics
    sentence = doc.sentences[0]
    root_verb = find_root(sentence)
    if not root_verb:
        return pragmatics

    # Check for question words
    first_word_lemma = sentence.words[0].lemma.lower()
    if first_word_lemma in ["who", "what", "where", "when", "why", "how"]:
        pragmatics["intent_primary"] = "QUESTION"
    elif root_verb.upos == "VERB":
        # A command involving "you" is often a request.
        pragmatics["intent_primary"] = "REQUEST" if "you" in [w.lemma for w in sentence.words] else "COMMAND"

    # Check for modality (e.g., "can", "will", "should")
    for word in sentence.words:
        if word.deprel == 'aux' and word.head == root_verb.id:
            if word.lemma == 'can':
                pragmatics['style']['modality'] = 'possibility_challenge'
                pragmatics['style']['politeness'] = 8
            # Add more rules for 'will', 'should', etc.
            break  # Assume one main modal auxiliary

    return pragmatics

def find_root(sentence):
    """
    Finds the root word of a sentence, which typically represents the main action.

    Args:
        sentence (stanza.Sentence): The Stanza sentence object.

    Returns:
        stanza.Word: The root word object, or None if not found.
    """
    for word in sentence.words:
        if word.deprel == 'root':
            return word
    return None

def extract_word_preposition(word, sentence):
    """
    Finds the preposition attached to a word (its 'case').

    This is crucial for determining semantic roles like 'source' (from),
    'target' (to), or 'location' (in, on, at).

    Args:
        word (stanza.Word): The word to check for an associated preposition.
        sentence (stanza.Sentence): The sentence containing the word.

    Returns:
        str: The text of the preposition, or None if not found.
    """
    for w in sentence.words:
        if w.head == word.id and w.deprel == 'case':
            return w.text
    return None

def extract_word_modifiers_and_embedded_params(word, sentence, stanza_entity=None):
    """
    Extracts descriptive modifiers and identifies embedded phrasal parameters.

    For a given noun, this function distinguishes between:
    1.  **Modifiers**: Words that describe the noun and should remain part of its
        entity (e.g., "financial" in "financial report").
    2.  **Embedded Parameters**: Prepositional phrases that act as separate
        parameters (e.g., "from Microsoft" in "a report from Microsoft").

    Args:
        word (stanza.Word): The head word of the entity.
        sentence (stanza.Sentence): The sentence object.
        stanza_entity (stanza.Entity, optional): The full Stanza NER entity,
            if one exists, to avoid re-adding its own words as modifiers.

    Returns:
        tuple[list, list]: A tuple containing:
            - A list of modifier dictionaries.
            - A list of embedded parameter dictionaries to be processed separately.
    """
    modifiers = []
    embedded_params = []

    entity_word_ids = {w.id for w in stanza_entity.words} if stanza_entity else set()

    for w in sentence.words:
        if w.head != word.id:
            continue
        # Don't add a word as a modifier if it's already part of the main entity text
        if w.id in entity_word_ids:
            continue

        preposition = extract_word_preposition(w, sentence)

        # Identify dependents that are likely separate parameters
        if w.deprel in ['nmod', 'obl'] and preposition in [
            'from', 'by', 'to', 'with', 'in', 'on', 'at', 'for', 'before', 'after', 'during'
        ]:
            embedded_params.append({
                "word": w,
                "preposition": preposition,
                "deprel": w.deprel
            })
            continue

        # Identify valid modifiers to keep with the entity
        if w.deprel in ['amod', 'nummod', 'compound', 'neg', 'advmod', 'conj']:
            modifiers.append({
                "text": w.text, "lemma": w.lemma, "pos": w.upos, "deprel": w.deprel
            })

    return modifiers, embedded_params

def normalize_time_and_date(entity_dict):
    """
    Normalizes a date/time expression to an ISO 8601 string using dateparser.

    This function leverages the `dateparser` library, which is highly effective
    at interpreting ambiguous, human-readable date/time strings.

    Args:
        entity_dict (dict): A dictionary representing the entity. It must have
                          a 'text' key containing the string to parse
                          (e.g., "tomorrow morning", "in 5 minutes").

    Returns:
        str: An ISO 8601 formatted string (e.g., "2023-11-03T09:00:00"),
             or None if parsing fails.
    """
    text_to_parse = entity_dict.get('text', '').strip()
    if not text_to_parse:
        return None

    # 'PREFER_DATES_FROM': 'future' resolves ambiguities like "Friday at 5pm"
    # to the upcoming Friday, not the one in the past.
    settings = {'PREFER_DATES_FROM': 'future', 'RETURN_AS_TIMEZONE_AWARE': False}
    parsed_date = search_dates(text_to_parse, settings=settings)

    if parsed_date:
        # search_dates returns a list of (text, datetime_obj) tuples. We use the first.
        return parsed_date[0][1].isoformat()
    return None

def normalized_time(entity_dict):
    """
    Normalizes time-related words based on their text and modifiers.
    Accepts an entity dictionary as input.
    """
    now = datetime.now()
    target_date = now.date()
    target_time = time(0, 0, 0)

    # --- Step 1: Consolidate all text for easy searching (FIXED) ---
    main_text = entity_dict.get("text", "").lower()
    modifiers = entity_dict.get("modifiers", []) # Use .get() for safety
    modifier_texts = [mod.get("text", "").lower() for mod in modifiers]
    all_texts = [main_text] + modifier_texts
    
    # Helper to find the first number in a list of texts
    def find_first_number(texts):
        for text in texts:
            if text.isdigit():
                return int(text)
        return None

    # --- Step 2: Handle special cases like "in 2 hours" (FIXED) ---
    if entity_dict.get("preposition") and entity_dict.get("preposition").lower() == 'in':
        num = find_first_number(modifier_texts)
        if num is not None:
            delta = None
            if any(s in all_texts for s in ['hour', 'hours']):
                delta = timedelta(hours=num)
            elif any(s in all_texts for s in ['minute', 'minutes']):
                delta = timedelta(minutes=num)
            elif any(s in all_texts for s in ['second', 'seconds']):
                delta = timedelta(seconds=num)
            
            if delta:
                final_dt = now + delta
                return final_dt.strftime("%Y-%m-%dT%H:%M:%S")

    # --- Step 3: Parse the Date ---
    date_is_set = False

    # Condition 4: "today", "tomorrow", "yesterday"
    if "tomorrow" in all_texts:
        target_date = now.date() + timedelta(days=1)
        date_is_set = True
    elif "yesterday" in all_texts:
        target_date = now.date() - timedelta(days=1)
        date_is_set = True
    elif "today" in all_texts:
        target_date = now.date()
        date_is_set = True

    # Condition 2: "week", "monday", "tuesday", etc.
    weekdays = {"monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3, 
                "friday": 4, "saturday": 5, "sunday": 6}
    day_word_found = next((day for day in weekdays if day in all_texts), None)

    if main_text == 'week' and not date_is_set:
        days_to_last_monday = now.weekday()
        if 'next' in all_texts:
            target_date = now.date() - timedelta(days=days_to_last_monday) + timedelta(weeks=1)
        elif 'last' in all_texts or 'past' in all_texts:
            target_date = now.date() - timedelta(days=days_to_last_monday) - timedelta(weeks=1)

    elif day_word_found and not date_is_set:
        today_weekday = now.weekday()
        target_weekday = weekdays[day_word_found]
        day_diff = target_weekday - today_weekday
        
        if 'next' in all_texts:
            if day_diff <= 0: day_diff += 7
        elif 'last' in all_texts or 'past' in all_texts:
            if day_diff >= 0: day_diff -= 7
        elif day_diff < 0:
            day_diff += 7
            
        target_date = now.date() + timedelta(days=day_diff)

    # --- Step 4: Parse the Time ---
    time_is_set = False
    hour = 0
    found_hour = find_first_number(all_texts)
    
    if 'morning' in all_texts:
        hour = found_hour if found_hour is not None else 9
        time_is_set = True
    elif 'noon' in all_texts:
        hour = 12
        time_is_set = True
    elif 'afternoon' in all_texts:
        hour = found_hour if found_hour is not None else 15
        if hour < 12: hour += 12
        time_is_set = True
    elif 'evening' in all_texts:
        hour = found_hour if found_hour is not None else 18
        if hour < 12: hour += 12
        time_is_set = True

    if found_hour is not None and not time_is_set:
        hour = found_hour
        if any(t in all_texts for t in ["pm", "p.m."]):
            if hour < 12: hour += 12
        elif any(t in all_texts for t in ["am", "a.m."]):
            if hour == 12: hour = 0
        time_is_set = True

    if time_is_set:
        target_time = time(hour, 0, 0)
        
    final_datetime = datetime.combine(target_date, target_time)
    return final_datetime.strftime("%Y-%m-%dT%H:%M:%S")

def extract_word_entity(word, sentence):
    """
    Creates a structured dictionary for a word, prioritizing Stanza's NER.

    This function implements a "NER-first" strategy. It first checks if the
    word is part of a multi-word named entity (e.g., "Google LLC", "next Tuesday").
    If so, it uses the full entity text and type. If not, it falls back to
    using the word's part-of-speech tag to infer a generic type.

    Args:
        word (stanza.Word): The word to analyze.
        sentence (stanza.Sentence): The sentence containing the word.

    Returns:
        tuple[dict, list, stanza.Entity]: A tuple containing:
            - The structured entity dictionary.
            - A list of any embedded parameters found.
            - The original Stanza entity object, if one was found.
    """
    # Check if the word is part of a larger named entity
    found_entity = next((ent for ent in sentence.ents if word in ent.words), None)

    entity = {
        "lemma": word.lemma,
        "pos": word.upos,
        "stanza_word_obj": word  # Temporary key for sorting; removed before output
    }

    if found_entity:
        # Use the full text and type from the identified NER entity
        entity["text"] = found_entity.text
        entity["ner_type"] = found_entity.type
    else:
        # Fallback to single-word analysis based on part-of-speech
        entity["text"] = word.text
        if word.upos == "NOUN": entity["ner_type"] = "THING"
        elif word.upos == "PROPN": entity["ner_type"] = "PROPN_UNKNOWN"
        elif word.upos == "PRON": entity["ner_type"] = "PRONOUN"
        elif word.upos == "NUM": entity["ner_type"] = "NUMBER"
        else: entity["ner_type"] = "OTHER"

    # --- Normalization ---
    normalized_value = entity["lemma"]  # Default to lemma
    if entity["ner_type"] in ["DATE", "TIME"]:
        normalized_value = normalize_time_and_date(entity) or entity["lemma"]
    elif entity["ner_type"] == "NUMBER":
        normalized_value = word.lemma
    elif found_entity:
        normalized_value = found_entity.text  # Use full text for proper names

    entity["normalized"] = normalized_value

    # --- Extract Modifiers and Prepositions ---
    preposition = extract_word_preposition(word, sentence)
    if preposition:
        entity["preposition"] = preposition

    modifiers, embedded = extract_word_modifiers_and_embedded_params(word, sentence, found_entity)
    if modifiers:
        entity["modifiers"] = modifiers

    return entity, embedded, found_entity

    # old method for custom NER Recognition
    # --- Named Entity Recognition Heuristics ---
    lower_text = word.text.lower()
    lemma_lower = word.lemma.lower()
    
    TIME_KEYWORDS = {
        "morning", "noon", "afternoon", "evening", "week", "sunday", "monday", 
        "tuesday", "wednesday", "thursday", "friday", "saturday", 
        "pm", "am", "p.m.", "a.m.", "o'clock", "hour", "hours", "minute", "minutes", "second", "seconds", 
        "today", "tomorrow", "yesterday"
    }
    DATE_KEYWORDS = {"day", "month", "year", "date", "decade", "century"}
    GPE_SUFFIXES = {"city", "country", "state", "province", "village"}
    ORG_HINTS = {"inc", "corp", "corporation", "ltd", "company", "group", "organization", "university", "institute"}
    PERSON_PRONOUNS = {"he", "she", "him", "her", "mr", "mrs", "ms"}


    ner_type = None
    normalized = None

    # --- TIME entities ---
    if lower_text in TIME_KEYWORDS:
        ner_type = "TIME"
        normalized = normalized_time(entity)

    # --- DATE entities ---
    elif lemma_lower in DATE_KEYWORDS:
        ner_type = "DATE"

    # --- NUMERIC ---
    elif word.upos == "NUM":
        ner_type = "NUMBER"
        normalized = word.lemma

    # --- PROPER NOUNs (can be ORG, GPE, PERSON) ---
    elif word.upos == "PROPN":
        compound_parts = [mod['text'] for mod in modifiers if mod['deprel'] == 'compound']
        full_name = ' '.join(compound_parts + [word.text])
        normalized = full_name

        # Basic heuristics
        if any(hint in lower_text for hint in ORG_HINTS):
            ner_type = "ORG"
        elif any(sfx in lower_text for sfx in GPE_SUFFIXES):
            ner_type = "GPE"
        elif any(mod['text'].lower() in PERSON_PRONOUNS for mod in modifiers):
            ner_type = "PERSON"
        else:
            ner_type = "PROPN"

    # --- Common NOUNs (like 'report', 'knife') ---
    elif word.upos == "NOUN":
        normalized = word.lemma
        ner_type = "THING"

    # --- PRONOUN ---
    elif word.upos == "PRON":
        ner_type = "PRONOUN"
        normalized = word.lemma

    # --- Default ---
    else:
        normalized = word.lemma
        ner_type = "ENTITY"

    entity["normalized"] = normalized
    entity["ner_type"] = ner_type

    return entity, embedded

def infer_role_from_preposition(preposition):
    """
    Maps common prepositions to semantic roles.

    Args:
        preposition (str): The preposition text (e.g., "from", "with").

    Returns:
        str: The inferred semantic role (e.g., "source", "instrument"),
             or None.
    """
    if not preposition:
        return None
    role_map = {
        "from": "source", "by": "agent", "with": "instrument", "for": "beneficiary",
        "to": "target", "in": "location", "on": "location", "at": "location",
        "before": "time_reference", "after": "time_reference", "during": "time_reference"
    }
    return role_map.get(preposition.lower())

def merge_time_entities(parameters):
    """
    Finds and merges separate but related time entities into a single entity.

    For example, if "at 9" and "tomorrow morning" are parsed as two separate
    parameters, this function combines them into a single parameter with the
    text "at 9 tomorrow morning" and re-parses it for a more accurate result.

    Args:
        parameters (list): The list of all extracted parameter dictionaries.

    Returns:
        list: The updated list of parameters with time entities merged.
    """
    time_params = [p for p in parameters if p.get("role") == "time_reference"]
    other_params = [p for p in parameters if p.get("role") != "time_reference"]

    if len(time_params) <= 1:
        return parameters # No merge needed
    
    # Sort time entities by their position in the sentence to maintain correct order
    time_params.sort(key=lambda p: p['entity']['stanza_word_obj'].id)
    
    combined_text = " ".join([p['entity']['text'] for p in time_params])
    
    # Use dateparser on the new, complete text string
    settings = {'PREFER_DATES_FROM': 'future', 'RETURN_AS_TIMEZONE_AWARE': False}
    parsed_results = search_dates(combined_text, settings=settings)

    if not parsed_results:
        return other_params + time_params # Return original parts if merge fails

    normalized_datetime = parsed_results[0][1].isoformat()

    # Create a single new parameter to replace the old ones
    merged_param = {
        "role": "time_reference",
        "entity": {
            "text": combined_text,
            "pos": "TIME",
            "lemma": combined_text.lower(),
            "ner_type": "TIME",
            "normalized": normalized_datetime
        }
    }
    return other_params + [merged_param]

def extract_parameters(parent_word, sentence):
    """
    Extracts all parameters for a given action word using dependency relations.

    This function iterates through the grammatical children of the action verb
    and assigns them semantic roles based on their dependency relation (`deprel`)
    and other contextual clues like prepositions.

    Args:
        parent_word (stanza.Word): The action word (e.g., the root verb).
        sentence (stanza.Sentence): The sentence object.

    Returns:
        list: A list of structured parameter dictionaries.
    """
    parameters = []
    processed_word_ids = set() # Track words handled by multi-word NER
    
    children = [word for word in sentence.words if word.head == parent_word.id]
    for child in children:
        if child.id in processed_word_ids:
            continue

        param = None
        entity, embedded_params, stanza_entity = extract_word_entity(child, sentence)

        if stanza_entity:
            for w in stanza_entity.words:
                processed_word_ids.add(w.id)

        # Main role assignment logic based on dependency relation
        deprel = child.deprel
        ner_type = entity.get("ner_type")
        prep = entity.get("preposition")

        # 1. Direct Object
        if deprel == 'obj':
            param = {
                "role": "direct_object",
                "entity": entity
            }
        # 2. Indirect Object
        elif deprel == 'iobj':
            param = {
                "role": "recipient",
                "entity": entity
            }
        # 3. Secondary Action
        elif deprel == 'xcomp':
            param = {
                "role": "secondary_action",
                "entity": extract_secondary_entity(child, sentence)
            }
        # 4. Clausal Complement
        elif deprel == 'ccomp':
            param = {
                "role": "embedded_clause",
                "entity": extract_secondary_entity(child, sentence)
            }
        # 5. Adverbial Clause
        elif deprel == 'advcl':
            param = {
                "role": "condition_or_cause",
                "entity": extract_secondary_entity(child, sentence)
            }
        # 6. Oblique (obl or obl:unmarked) and nmod
        elif deprel in ['obl', 'obl:unmarked', 'nmod']:
            if ner_type in ['TIME', 'DATE']:
                param = {
                    "role": "time_reference",
                    "entity": entity
                }
            elif entity.get("ner_type") == "NUMBER" and prep == "at":
                param = {
                    "role": "time_reference",
                    "entity": entity
                }
            elif ner_type == "GPE" or (prep in ["in", "on", "at"]):
                param = {
                    "role": "location",
                    "entity": entity
                }
            # Source
            elif prep == "from":
                param = {
                    "role": "source",
                    "entity": entity
                }
            # Target
            elif prep == "to":
                param = {
                    "role": "target",
                    "entity": entity
                }
            # Beneficiary
            elif prep == "for":
                # If it's a pronoun or proper noun → likely a beneficiary
                if child.upos in ["PRON", "PROPN"]:
                    param = {
                        "role": "beneficiary",
                        "entity": entity
                    }
                # If it's a noun like "meeting", "event", etc. → subject/topic/purpose
                elif child.upos == "NOUN":
                    param = {
                        "role": "subject",  # or "topic", if you prefer
                        "entity": entity
                    }
            # Instrument or Companion
            elif prep == "with":
                if child.upos in ["PROPN", "PRON"]:
                    inferred = "companion"
                else:
                    inferred = "instrument"
                param = {
                    "role": inferred,
                    "entity": entity
                }
            # Agent (by)
            elif prep == "by" or deprel in ['nmod:agent', 'obl:agent']:
                param = {
                    "role": "agent",
                    "entity": entity
                }
            # Fallback from preposition or ner_type
            elif prep:
                param = {
                    "role": infer_role_from_preposition(prep) or "oblique",
                    "entity": entity
                }
            else:
                param = {
                    "role": "modifier",
                    "entity": entity
                }
        # 7. Possessor (e.g., "John's book")
        elif deprel == 'nmod:poss':
            param = {
                "role": "possessor",
                "entity": entity
            }
        # 8. Other modifiers (amod, advmod)
        elif deprel in ['amod', 'advmod']:
            param = {
                "role": "modifier",
                "entity": entity
            }

        # ➕ Add main parameter
        if param:
            parameters.append(param)

        # Process any embedded parameters that were extracted
        for ep in embedded_params:
            ep_entity, _, _ = extract_word_entity(ep["word"], sentence)
            ep_ner_type = ep_entity.get("ner_type")
            ep_prep = ep.get("preposition")

            # Determine role for embedded entity
            if ep_ner_type in ["TIME", "DATE"]:
                role = "time_reference"
            elif ep_ner_type == "NUMBER" and ep_prep == "at":
                role = "time_reference"
            elif ep_ner_type == "GPE":
                role = "location"
            else:
                role = infer_role_from_preposition(ep_prep) or ep.get("deprel") or "parameter"

            parameters.append({
                "role": role,
                "entity": ep_entity
            })
            # print(f"Extracted embedded parameter: {role} -> {ep_entity}")

    # Final cleanup step to merge related time entities
    parameters = merge_time_entities(parameters)
    return parameters

def extract_secondary_entity(action_word, sentence):
    action = {
        'text': action_word.text,
        'lemma': action_word.lemma,
        'pos': action_word.upos
    }
    # agent = None
    # for word in sentence.words:
    #     if word.head == action_word.id and word.deprel == 'nsubj':
    #         agent = {
    #             "text": word.text,
    #             "pos": word.upos,
    #             "is_ai": word.lemma == 'you'
    #         }
    #         break

    entity = {
        "action": action,
        # "agent": agent,
        "parameters": extract_parameters(action_word, sentence)
    }

    return entity

def extract_tasks(action_word, sentence):
    """
    Constructs the main task object, including action, agent, and parameters.

    Args:
        action_word (stanza.Word): The root word representing the main action.
        sentence (stanza.Sentence): The sentence object.

    Returns:
        dict: A dictionary representing the primary task.
    """
    action = {'lemma': action_word.lemma, 'text': action_word.text, 'pos': action_word.upos}
    
    # Find the agent (subject of the action)
    agent_word = next((w for w in sentence.words if w.head == action_word.id and w.deprel == 'nsubj'), None)
    agent = None
    if agent_word:
        agent = {
            "text": agent_word.text,
            "pos": agent_word.upos,
            "is_ai": agent_word.lemma == 'you' # Flag if the AI is the agent
        }

    task = {
        "action": action,
        "agent": agent,
        "parameters": extract_parameters(action_word, sentence)
    }

    return task

def recursively_clean_word_objects(obj):
    """
    Recursively removes temporary 'stanza_word_obj' keys from the final output.

    This function cleans the final dictionary/list structure before JSON
    serialization, ensuring the output is clean and contains no un-serializable
    Python objects.

    Args:
        obj (dict or list): The object to clean.
    """
    if isinstance(obj, dict):
        if 'stanza_word_obj' in obj:
            del obj['stanza_word_obj']
        for value in obj.values():
            recursively_clean_word_objects(value)
    elif isinstance(obj, list):
        for item in obj:
            recursively_clean_word_objects(item)


# --- The Main Parser Function ---

def parse_query_to_structured_json(doc):
    """
    Orchestrates the entire parsing pipeline for a given document.

    Args:
        doc (stanza.Document): The processed Stanza document.

    Returns:
        dict: The final, structured JSON representation of the command.
    """
    if not doc.sentences:
        return {"error": "No sentences found."}

    sentence = doc.sentences[0]
    root = find_root(sentence)

    if not root:
        return {"error": "Could not determine the main action of the sentence."}
    
    structured_output = {
        "metadata": extract_metadata(doc),
        "pragmatics": extract_pragmatics(doc),
        "tasks": extract_tasks(root, sentence)
    }
    
    recursively_clean_word_objects(structured_output)
    return structured_output


# --- Main Application Logic ---

def main():
    """
    The main function to set up Stanza and run the interactive command shell.
    """
    print("Initializing Stanza NLP Pipeline... (This may take a moment)")
    try:
        nlp = stanza.Pipeline('en', processors=PROCESSORS, use_gpu=False)
        print("Pipeline ready. Type a sentence and press Enter.")
        print("Type 'exit' or 'quit' to close the shell.")
    except Exception as e:
        print(f"Error initializing Stanza pipeline: {e}")
        print("\nHint: If this is your first time, you may need to download the models.")
        print("Run this in a Python shell: \nimport stanza\nstanza.download('en')")
        sys.exit(1)

    while True:
        try:
            user_input = input("\n> ")
            if user_input.lower() in ['exit', 'quit']:
                print("Exiting. Goodbye!")
                break
            
            if not user_input.strip():
                continue

            doc = nlp(user_input)
            
            # A document can have multiple sentences. Let's work with the first one.
            sentence = doc.sentences[0]
            print("--- Word Attributes ---")
            for word in sentence.words:
                # The word's index in the sentence (1-based)
                print(f"ID: {word.id}") 
                # The text of the word
                print(f"Text: {word.text}")
                # The lemma (base form) of the word
                print(f"Lemma: {word.lemma}")
                # The universal POS tag (VERB, NOUN, ADJ...)
                print(f"UPOS: {word.upos}")
                # The more specific treebank tag (VB, NN, JJ...)
                print(f"XPOS: {word.xpos}")
                # The ID of the word this word is attached to (its parent/head)
                print(f"Head ID: {word.head}")
                # The dependency relation label (e.g., 'nsubj', 'obj')
                print(f"Dependency Relation: {word.deprel}")
                print("-" * 20)
            
            result = parse_query_to_structured_json(doc)
            
            print("\n--- Structured JSON Output ---")
            print(json.dumps(result, indent=2))

        except KeyboardInterrupt:
            # Allow clean exit with Ctrl+C
            print("\nExiting. Goodbye!")
            break
        except Exception as e:
            print(f"An error occurred: {e}")
            import traceback
            traceback.print_exc() # Print full error for easier debugging


if __name__ == '__main__':
    main()