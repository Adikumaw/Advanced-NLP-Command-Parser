import stanza
import sys
import json
from datetime import datetime, timedelta, time
from dateparser.search import search_dates

# --- Configuration ---
# Use 'gpu=False' if you don't have a compatible GPU or CUDA installed.
# The 'tokenize,pos,lemma,depparse' processors are needed for our tasks.
PROCESSORS = 'tokenize,pos,lemma,depparse,ner'

# --- Helper Functions to Build JSON Parts ---

def extract_metadata(doc):
    """
    Extracts metadata from the document, such as the number of sentences and words.
    """
    metadata = {
        "source_text": doc.text,
        "num_sentences": len(doc.sentences),
        "timestamp": datetime.now().isoformat(),
    }
    return metadata

def extract_pragmatics(doc):
    """
    Analyzes the overall feel of the query.
    This is a simplified version; a real one would use more complex models.
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
    root_verb = find_root(doc.sentences[0])
    if not root_verb:
        return pragmatics

    # Check for question words
    first_word = doc.sentences[0].words[0].lemma.lower()
    if first_word in ["who", "what", "where", "when", "why", "how"]:
        pragmatics["intent_primary"] = "QUESTION"
    elif root_verb.upos == "VERB":
        pragmatics["intent_primary"] = "REQUEST" if "you" in [w.lemma for w in doc.sentences[0].words] else "COMMAND"

    # Check for modality (like "can", "will", "should")
    for word in doc.sentences[0].words:
        if word.deprel == 'aux' and word.head == root_verb.id:
            if word.lemma == 'can':
                pragmatics['style']['modality'] = 'possibility_challenge'
                pragmatics['style']['politeness'] = 8
            # Add more rules for 'will', 'should', etc.
            break # Assume one main modal auxiliary
            
    return pragmatics

def find_root(sentence):
    """Finds the root word of a sentence."""
    for word in sentence.words:
        if word.deprel == 'root':
            return word
    return None

def extract_word_preposition(word, sentence):
    """
    Extracts the preposition associated with a word in a sentence.
    This is useful for understanding the context of the word.
    """
    preposition = None
    for w in sentence.words:
        if w.head == word.id and w.deprel == 'case':
            preposition = w.text
            break
    return preposition

def extract_word_modifiers_and_embedded_params(word, sentence, stanza_entity=None):
    """
    Extracts modifiers (e.g., adjectives, compounds) and flags embedded parameters
    like sources or locations (e.g., 'from Microsoft') attached to this word.
    Returns:
        - modifiers: list of descriptive modifiers to attach to entity
        - embedded_params: list of words that should be treated as separate parameters
    """
    modifiers = []
    embedded_params = []

    entity_word_ids = []
    if stanza_entity:
        entity_word_ids = {w.id for w in stanza_entity.words}

    for w in sentence.words:
        if w.head != word.id:
            continue

        # --- NEW: Don't add a word as a modifier if it's already part of the main entity text ---
        if w.id in entity_word_ids:
            continue

        # Extract preposition if it's a prepositional phrase (e.g., 'from Microsoft')
        case_word = next((cw for cw in sentence.words if cw.head == w.id and cw.deprel == 'case'), None)
        preposition = case_word.text if case_word else None

        # Parameter-worthy dependents
        if w.deprel in ['nmod', 'obl', 'obl:agent', 'nmod:agent']:
            if preposition in ['from', 'by', 'to', 'with', 'in', 'on', 'at', 'for', 'before', 'after', 'during']:
                # Tag this as a candidate for separate parameter
                embedded_params.append({
                    "word": w,
                    "preposition": preposition,
                    "deprel": w.deprel
                })
                continue  # skip adding to modifiers

        # Valid modifiers to keep inside entity
        if w.deprel in ['amod', 'nummod', 'compound', 'neg', 'advmod', 'conj']:
            modifiers.append({
                "text": w.text,
                "lemma": w.lemma,
                "pos": w.upos,
                "deprel": w.deprel
            })

    return modifiers, embedded_params

def normalize_time_and_date(entity_dict):
    """
    Uses the dateparser library to normalize any time or date expression.
    
    Args:
        entity_dict: A dictionary representing the entity, which must have a 'text' key
                     and can have 'modifiers'.
    
    Returns:
        An ISO 8601 formatted string if parsing is successful, otherwise None.
    """
    # Combine the main entity text with its modifiers for full context.
    # e.g., for "next" and "week", the full text is "next week".
    full_text = entity_dict.get('text', '')
    # NOTE: No need for modifiers here, as Stanza's NER gives us the full text.
    text_to_parse = full_text.strip()

    if not text_to_parse:
        return None

    # Use dateparser. It's smart about context.
    # 'PREFER_DATES_FROM': 'future' tells it to resolve ambiguities like "Friday at 5pm"
    # to the upcoming Friday, not the one in the past.
    settings = {'PREFER_DATES_FROM': 'future', 'RETURN_AS_TIMEZONE_AWARE': False}
    parsed_date = search_dates(text_to_parse, settings=settings)

    if parsed_date:
        # dateparser returns a list of (text, datetime_obj) tuples. We take the first.
        return parsed_date[0][1].isoformat()
    
    return None # Return None if parsing fails

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
    Extracts a structured entity, now prioritizing Stanza's NER.
    It correctly handles multi-word entities (e.g., "financial report").
    """
    # --- Stanza NER Integration ---
    # Check if the current word is part of a larger named entity identified by Stanza.
    # `sentence.ents` contains all entities like ("Microsoft", "ORG"), ("tomorrow morning", "DATE").
    found_entity = None
    for ent in sentence.ents:
        # Check if the current word is one of the words that make up this entity.
        if word in ent.words:
            found_entity = ent
            break

    entity = {
        "lemma": word.lemma,
        "pos": word.upos,
        "stanza_word_obj": word # --- NEW: Store the word object for sorting later ---
    }

    if found_entity:
        # This is a named entity (e.g., ORG, PERSON, DATE). Use its properties.
        entity["text"] = found_entity.text
        entity["ner_type"] = found_entity.type
    else:
        # Not a named entity. Fall back to POS-based heuristics.
        entity["text"] = word.text
        # Fallback NER type based on the Part-of-Speech tag.
        if word.upos == "NOUN":
            entity["ner_type"] = "THING"
        elif word.upos == "PROPN":
            entity["ner_type"] = "PROPN_UNKNOWN" # Proper noun, but not a known entity type
        elif word.upos == "PRON":
            entity["ner_type"] = "PRONOUN"
        elif word.upos == "NUM":
            entity["ner_type"] = "NUMBER"
        else:
            entity["ner_type"] = "OTHER"

    # --- Normalization Step ---
    normalized_value = entity["lemma"] # Default normalization
    if entity["ner_type"] in ["DATE", "TIME"]:
        # Use our new, powerful dateparser function
        normalized_value = normalize_time_and_date(entity) or entity["lemma"]
    elif entity["ner_type"] == "NUMBER" or word.upos == "NUM":
        normalized_value = word.lemma
    elif found_entity: # For entities like ORG, PERSON, GPE
        normalized_value = found_entity.text

    entity["normalized"] = normalized_value

    # --- Extract Modifiers and Prepositions
    preposition = extract_word_preposition(word, sentence)
    if preposition:
        entity["preposition"] = preposition
    
    modifiers, embedded = extract_word_modifiers_and_embedded_params(word, sentence)
    if modifiers:
        entity["modifiers"] = modifiers
    
    return entity, embedded, found_entity

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
    if not preposition:
        return None
    return {
        "from": "source",
        "by": "agent",
        "with": "instrument",
        "for": "beneficiary",
        "to": "target",
        "in": "location",
        "on": "location",
        "at": "location",
        "before": "time_reference",
        "after": "time_reference",
        "during": "time_reference"
    }.get(preposition.lower(), None)

def merge_time_entities(parameters):
    """
    Finds and merges separate but related time entities into a single,
    more complete entity. This is the final cleanup step.
    
    Example: Merges ["9 tomorrow"] and ["morning"] into ["9 tomorrow morning"].
    """
    # 1. Collect all time-related parameters and all other parameters.
    time_params = [p for p in parameters if p.get("role") == "time_reference"]
    other_params = [p for p in parameters if p.get("role") != "time_reference"]

    # 2. Handle the case where there is 0 or 1 time entity (no merge needed).
    if len(time_params) <= 1:
        # Even if we don't merge, we MUST clean the temporary key before returning.
        if time_params:  # Check if the list isn't empty
            if 'stanza_word_obj' in time_params[0]['entity']:
                del time_params[0]['entity']['stanza_word_obj']
        # Return the RECONSTRUCTED list from the clean parts. This is the key fix.
        return other_params + time_params
    
    # 3. Handle the case where a merge is needed (more than 1 time entity).
    # We sort them by their first word's position in the sentence to keep the order correct.
    # This is a robust way to handle the word order.
    time_params.sort(key=lambda p: p['entity']['stanza_word_obj'].id)

    # Correctly build the combined text *from the sorted list*.
    combined_text = " ".join([p['entity']['text'] for p in time_params])

    # Now, clean the temporary keys from the original list parts.
    for p in time_params:
        if 'stanza_word_obj' in p['entity']:
            del p['entity']['stanza_word_obj']


    # 3. Use dateparser on the new, full text string.
    settings = {'PREFER_DATES_FROM': 'future', 'RETURN_AS_TIMEZONE_AWARE': False}
    parsed_results = search_dates(combined_text, settings=settings)

    # If merging fails, we must still return a clean list.
    if not parsed_results:
        return other_params + time_params

    normalized_datetime = parsed_results[0][1].isoformat()

    # 4. Create a single new parameter to replace the old ones.
    merged_param = {
        "role": "time_reference",
        "entity": {
            "text": combined_text,
            "pos": "TIME", # Generic POS for the merged entity
            "lemma": combined_text,
            "ner_type": "TIME",
            "normalized": normalized_datetime
        }
    }

    # 5. Return the other parameters plus our new, single merged time parameter.
    return other_params + [merged_param]

def extract_parameters(parent_word, sentence):
    """
    Recursively finds all parameters (obj, iobj, xcomp, obl, etc.) for a given action word.
    """
    parameters = []
    # --- NEW: A set to track word IDs that have already been processed ---
    processed_word_ids = set()
    children = [word for word in sentence.words if word.head == parent_word.id]
    
    for child in children:
        # --- NEW: If we've already handled this word, skip it ---
        if child.id in processed_word_ids:
            continue

        param = None
        entity, embedded_params, stanza_entity = extract_word_entity(child, sentence)

        # --- NEW: If this was part of a multi-word entity, mark all its words as processed ---
        if stanza_entity:
            for w in stanza_entity.words:
                processed_word_ids.add(w.id)

        ner_type = entity.get("ner_type")
        prep = entity.get("preposition")

        # 1. Direct Object
        if child.deprel == 'obj':
            param = {
                "role": "direct_object",
                "entity": entity
            }

        # 2. Indirect Object
        elif child.deprel == 'iobj':
            param = {
                "role": "recipient",
                "entity": entity
            }

        # 3. Secondary Action
        elif child.deprel == 'xcomp':
            param = {
                "role": "secondary_action",
                "entity": extract_secondary_entity(child, sentence)
            }

        # 4. Clausal Complement
        elif child.deprel == 'ccomp':
            param = {
                "role": "embedded_clause",
                "entity": extract_secondary_entity(child, sentence)
            }

        # 5. Adverbial Clause
        elif child.deprel == 'advcl':
            param = {
                "role": "condition_or_cause",
                "entity": extract_secondary_entity(child, sentence)
            }

        # 6. Oblique (obl or obl:unmarked)
        elif child.deprel in ['obl', 'obl:unmarked', 'nmod']:

            # Time-based ner_type
            if ner_type in ['TIME', 'DATE']:
                param = {
                    "role": "time_reference",
                    "entity": entity
                }
            
            elif entity.get("ner_type") == "NUMBER" and entity.get("preposition") == "at":
                param = {
                    "role": "time_reference",
                    "entity": entity
                }

            # Location
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
            elif prep == "by" or child.deprel in ['nmod:agent', 'obl:agent']:
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
        elif child.deprel == 'nmod:poss':
            param = {
                "role": "possessor",
                "entity": entity
            }

        # 8. Other modifiers (amod, advmod)
        elif child.deprel in ['amod', 'advmod']:
            param = {
                "role": "modifier",
                "entity": entity
            }

        # ➕ Add main parameter
        if param:
            parameters.append(param)
            # print(f"Extracted parameter: {param['role']} -> {param['entity']}")

        # ➕ Add embedded parameters extracted inside the entity (from modifiers)
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
    action = {
        'lemma': action_word.lemma,
        'text': action_word.text,
        'pos': action_word.upos
    }
    agent = None
    for word in sentence.words:
        if word.head == action_word.id and word.deprel == 'nsubj':
            agent = {
                "text": word.text,
                "pos": word.upos,
                "is_ai": word.lemma == 'you'
            }
            break

    task = {
        # "id": "task_1",
        "action": action,
        "agent": agent,
        # "confidence": 0.95,
        "parameters": extract_parameters(action_word, sentence)
    }

    return task

def recursively_clean_word_objects(obj):
    """
    Recursively walks through a dictionary or list and removes any key
    named 'stanza_word_obj'. This is the final cleanup step before
    converting to JSON.
    """
    if isinstance(obj, dict):
        # Create a list of keys to remove to avoid modifying dict while iterating
        keys_to_remove = [key for key in obj if key == 'stanza_word_obj']
        for key in keys_to_remove:
            del obj[key]
        # Recursively clean the values
        for key, value in obj.items():
            recursively_clean_word_objects(value)
    elif isinstance(obj, list):
        # Recursively clean each item in the list
        for item in obj:
            recursively_clean_word_objects(item)

# --- The Main Parser Function ---

def parse_query_to_structured_json(doc):
    if not doc.sentences:
        return {"error": "No sentences found."}

    sentence = doc.sentences[0]
    # --- NEW: It's good practice to print the entities Stanza found for debugging ---
    print("\n--- Stanza's Named Entities ---")
    if not sentence.ents:
        print("None found.")
    for ent in sentence.ents:
        print(f'-> text: "{ent.text}", type: {ent.type}')
    print("-" * 20)

    root = find_root(sentence)

    if not root:
        return {"error": "Could not determine the main action of the sentence."}
    
    structured_output = {
        "metadata": extract_metadata(doc),
        "pragmatics": extract_pragmatics(doc),
        "tasks": extract_tasks(root, sentence)
    }
    # Clean up any temporary word objects before returning
    recursively_clean_word_objects(structured_output)
    return structured_output


# --- Main Application Logic ---

def main():
    """The main function to set up Stanza and run the interactive shell."""
    print("Initializing Stanza NLP Pipeline... (This may take a moment)")
    try:
        nlp = stanza.Pipeline('en', processors=PROCESSORS, use_gpu=False)
        print("Pipeline ready. Type a sentence and press Enter.")
        print("Type 'exit' or 'quit' to close the shell.")
    except Exception as e:
        print(f"Error initializing Stanza pipeline: {e}")
        # --- NEW: Added a helpful hint to download the models ---
        print("\nHint: If this is your first time, you may need to download the models.")
        print("Run this in a Python shell: ")
        print("import stanza")
        print("stanza.download('en')")
        sys.exit(1)

    # The interactive loop
    while True:
        try:
            user_input = input("\n> ")
            if user_input.lower() in ['exit', 'quit']:
                print("Exiting. Goodbye!")
                break
            
            if not user_input:
                continue

            # Process the user's text
            doc = nlp(user_input)
            
            # Print the three different analyses
            # print_dependency_tree(doc)
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
            # Pretty print the JSON result
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