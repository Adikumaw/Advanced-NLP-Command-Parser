# Advanced NLP Command Parser

## 1. Overview

This project is a sophisticated Python script designed to parse natural language commands into a structured, machine-readable JSON format. It leverages the **Stanza NLP** library for deep linguistic analysis and **dateparser** for robust time and date normalization.

The primary goal is to deconstruct a user's command (e.g., _"Remind me to send the financial report from ACME to John tomorrow morning"_) into its core components:

- The main action
- The agent performing the action
- A detailed list of parameters with assigned semantic roles (like direct object, source, target, and time)

This structured output can be used to power a wide range of applications, including:

- Intelligent personal assistants and chatbots
- Task automation and scheduling systems
- Data extraction from unstructured text
- Interfacing with APIs using natural language

---

## 2. Core Features

- **Dependency Parsing**  
  Identifies the main action (root verb) of the sentence and the grammatical relationships between words.

- **Named Entity Recognition (NER)**  
  Uses Stanza's pre-trained models to accurately identify entities like `PERSON`, `ORGANIZATION`, `DATE`, and `TIME`.

- **Semantic Role Labeling**  
  Assigns meaningful roles to parameters based on their grammatical function (e.g., `direct_object`, `source`, `location`, `beneficiary`).

- **Advanced Time/Date Normalization**

  - Uses the `dateparser` library to convert natural language expressions (_"next Friday at 9am"_, _"in 2 hours"_) into standardized **ISO 8601** format.
  - Intelligently merges fragmented time expressions (e.g., _"tomorrow"_ and _"morning"_) into a single, cohesive entity.

- **Pragmatic Analysis**  
  Performs a basic analysis of the user's intent (e.g., `QUESTION`, `REQUEST`) and modality.

- **Structured JSON Output**  
  Produces a clean, predictable, and hierarchical JSON object representing the parsed command.

---

## 3. How It Works: The Parsing Pipeline

The script processes text through a multi-stage pipeline:

### Initialization

- The `main()` function initializes the Stanza pipeline with the necessary processors:  
  `tokenize`, `pos`, `lemma`, `depparse`, `ner`.

### Linguistic Processing

- Input text is passed to the Stanza pipeline, producing a `Document` object containing:
  - Tokens
  - Part-of-speech tags
  - Lemmas
  - Dependency relations
  - Named entities

### Root Identification

- Identifies the root of the sentence (typically the main verb).

### High-Level Analysis

- `extract_metadata()` — Gathers basic info like source text and timestamp.
- `extract_pragmatics()` — Analyzes the structure to infer intent.
- `extract_tasks()` — Core analysis:
  - Identifies the main action (root word)
  - Finds the agent (e.g., "You" or "I")
  - Calls `extract_parameters()` to find all associated parameters

### Parameter & Entity Extraction

Functions: `extract_parameters`, `extract_word_entity`

- Iterates through dependent words of the main action
- Uses a **NER-first** approach:
  - If part of a named entity (e.g., _"San Francisco"_, _"tomorrow morning"_) → treat as single unit
  - Else → fallback to POS tagging (e.g., `NOUN`, `PROPN`)
- Extracts modifiers (e.g., _"financial report"_)
- Separates embedded parameters (e.g., _"from ACME"_)

### Role Assignment

- Based on `deprel` (dependency relation) and prepositions:
  - `obj` → `direct_object`
  - `obl` with “from” → `source`
  - etc.

### Normalization (`normalize_time_and_date`)

- Passes `DATE` or `TIME` entities to `dateparser`
- Resolves into **ISO 8601**, future-aware, timezone-naive format

### Post-Processing & Cleanup

- `merge_time_entities()` — Combines related fragments like _"at 9"_ and _"tomorrow"_
- `recursively_clean_word_objects()` — Removes temp fields from output for clean JSON

---

## 4. Prerequisites

- **Python 3.6+**
- Required libraries:
  - `stanza`
  - `dateparser`

---

## 5. Installation & Setup

Clone the repo (optional) and install required libraries:

```bash
pip install stanza dateparser
```

Download Stanza English Models (one-time setup):

```bash
python stanza_model_download.py
```

## 6. Usage

Run the script from terminal:

```bash
python core.py
```

You'll get an interactive prompt (>). Type your sentence and press Enter to see structured JSON.
To exit, type exit or quit.

### Example Interaction

Command:

```CLI
> Send the weekly report from accounting to marketing next Friday.
```

Output:

```JSON
{
  "metadata": {
    "source_text": "Send the weekly report from accounting to marketing next Friday.",
    "num_sentences": 1,
    "timestamp": "2025-07-14T13:33:39.678989"
  },
  "pragmatics": {
    "intent_primary": "COMMAND",
    "sentiment": "neutral",
    "emotion": null,
    "style": {
      "formality": "neutral",
      "politeness": 5,
      "modality": null
    }
  },
  "tasks": {
    "action": {
      "lemma": "send",
      "text": "Send",
      "pos": "VERB"
    },
    "agent": null,
    "parameters": [
      {
        "role": "direct_object",
        "entity": {
          "lemma": "report",
          "pos": "NOUN",
          "text": "report",
          "ner_type": "THING",
          "normalized": "report",
          "modifiers": [
            {
              "text": "weekly",
              "lemma": "weekly",
              "pos": "ADJ",
              "deprel": "amod"
            }
          ]
        }
      },
      {
        "role": "source",
        "entity": {
          "lemma": "accounting",
          "pos": "NOUN",
          "text": "accounting",
          "ner_type": "THING",
          "normalized": "accounting",
          "preposition": "from"
        }
      },
      {
        "role": "target",
        "entity": {
          "lemma": "marketing",
          "pos": "NOUN",
          "text": "marketing",
          "ner_type": "THING",
          "normalized": "marketing",
          "preposition": "to"
        }
      },
      {
        "role": "time_reference",
        "entity": {
          "lemma": "Friday",
          "pos": "PROPN",
          "text": "next Friday",
          "ner_type": "DATE",
          "normalized": "2025-07-18T00:00:00"
        }
      }
    ]
  }
}
```
