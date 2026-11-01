"""Recover one unambiguous choice, including A..Z, AA..BF inventories."""

import json
import re


def parse_answer(raw_output, choices):
    """Return a dict with valid, option_id and reason.

    An explicit option identifier takes priority over a handshape name such as
    'c'. Conflicting answer declarations are invalid. Ordinary explanations
    are not searched for isolated letters (the article 'a' is not an answer).
    """
    allowed = {}
    for choice in choices:
        allowed[choice["option_id"].upper()] = choice

    candidates = []
    saw_answer = False
    invalid_declaration = False
    for match in re.finditer(r"\{[^{}]*\}", raw_output, re.DOTALL):
        try:
            # Keep duplicate keys visible: answer A followed by answer B is
            # conflicting output, rather than a JSON last-key-wins answer.
            payload = json.loads(match.group(), object_pairs_hook=list)
        except json.JSONDecodeError:
            continue
        for key, raw_value in payload:
            if key not in ("answer", "option", "choice"):
                continue
            saw_answer = True
            if raw_value is None or isinstance(raw_value, (list, dict)):
                invalid_declaration = True
                continue
            value = str(raw_value).strip()
            option = resolve_value(value, allowed)
            if option is None:
                invalid_declaration = True
            elif option not in candidates:
                candidates.append(option)

    # Check explicit text declarations too, so JSON A + 'Answer: B' conflicts.
    pattern = r"\b(?:answer|option|choice)\s*(?:is\s+|[:=]\s*)?\(?([A-Za-z]{1,3})\)?(?=[\s.,;:!\]}]|$)"
    for match in re.finditer(pattern, raw_output, re.IGNORECASE):
        token = match.group(1).upper()
        if token in allowed and token not in candidates:
            candidates.append(token)
            saw_answer = True

    # Explicit alternatives do not identify a unique answer.
    alternatives = r"\b([A-Z]{1,3})\s*(?:or|and|/)\s*([A-Z]{1,3})\b"
    for match in re.finditer(alternatives, raw_output, re.IGNORECASE):
        left, right = match.group(1).upper(), match.group(2).upper()
        if left in allowed and right in allowed:
            saw_answer = True
            for option in [left, right]:
                if option not in candidates:
                    candidates.append(option)

    if not saw_answer:
        text = raw_output.strip().strip("`").strip()
        # A single bare answer, optionally enclosed in parentheses or ending
        # in punctuation, is allowed. Do not mine arbitrary prose for letters.
        text = text.strip("(). \n")
        option = resolve_value(text, allowed)
        if option is not None:
            candidates.append(option)

    if invalid_declaration:
        return {"valid": False, "option_id": None, "reason": "invalid_answer_declaration"}
    if len(candidates) == 1:
        return {"valid": True, "option_id": candidates[0], "reason": "unique_answer"}
    reason = "conflicting_answers" if candidates else "no_allowed_answer"
    return {"valid": False, "option_id": None, "reason": reason}


def resolve_value(value, allowed):
    token = value.upper()
    if token.startswith("OPTION "):
        token = token[7:].strip()
    if token in allowed:
        return token
    matches = []
    for option, choice in allowed.items():
        text = str(choice["text"]).strip().lower()
        semantic = str(choice["semantic"]).strip().lower()
        if value.lower() == text or value.lower() == semantic:
            matches.append(option)
    if len(matches) == 1:
        return matches[0]
    return None


def semantic_for_option(choices, option):
    for choice in choices:
        if choice["option_id"] == option:
            return choice["semantic"]
    raise ValueError(f"Unknown option {option!r}")
