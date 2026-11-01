"""Execute the ten formal tasks. Gold answers never depend on video pixels."""

from .graph import delta, is_one_feature_pair


def execute_program(program, vectors, labels=None):
    """Return a bool, count, field name, or item key; reject ambiguous selections."""
    if labels is None:
        labels = {}
        for key in vectors:
            labels[key] = key
    op = program["op"]

    if op == "same_sign":
        left, right = program["items"]
        return labels[left] == labels[right]

    if op == "one_feature_pair":
        left, right = program["items"]
        return is_one_feature_pair(labels[left], labels[right], vectors[left], vectors[right])

    if op == "distinguishing_feature":
        left, right = program["items"]
        changed = delta(vectors[left], vectors[right])
        if labels[left] == labels[right] or len(changed) != 1:
            raise ValueError("Distinguishing-feature inputs must form a one-feature pair")
        return changed[0]

    if op == "exists_one_feature_pair":
        items = program["items"]
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                left, right = items[i], items[j]
                if is_one_feature_pair(labels[left], labels[right], vectors[left], vectors[right]):
                    return True
        return False

    if op == "universal_shared_feature":
        items = program["items"]
        if not items:
            raise ValueError("Universal task requires a nonempty set")
        field = program["field"]
        value = vectors[items[0]][field]
        for key in items[1:]:
            if vectors[key][field] != value:
                return False
        return True

    if op == "feature_value_counting":
        count = 0
        for key in program["items"]:
            if vectors[key][program["field"]] == str(program["value"]):
                count += 1
        return count

    valid = []
    if op == "constraint_satisfaction":
        for key in program["candidates"]:
            a = delta(vectors[program["anchor_a"]], vectors[key])
            b = delta(vectors[program["anchor_b"]], vectors[key])
            if a == [program["field_a"]] and b == [program["field_b"]]:
                valid.append(key)

    elif op == "feature_transformation":
        base = program["base"]
        field = program["field"]
        if "source_value" in program and vectors[base][field] != str(program["source_value"]):
            raise ValueError("Transformation base does not have the supplied source value")
        for key in program["candidates"]:
            if delta(vectors[base], vectors[key]) == [field]:
                if vectors[key][field] == str(program["target_value"]):
                    valid.append(key)

    elif op == "common_neighbor":
        left, right = program["anchor_left"], program["anchor_right"]
        for key in program["candidates"]:
            a = is_one_feature_pair(labels[left], labels[key], vectors[left], vectors[key])
            b = is_one_feature_pair(labels[right], labels[key], vectors[right], vectors[key])
            if a and b:
                valid.append(key)

    elif op == "isolate_detection":
        for key in program["items"]:
            isolated = True
            for other in program["items"]:
                if key == other:
                    continue
                if is_one_feature_pair(labels[key], labels[other], vectors[key], vectors[other]):
                    isolated = False
                    break
            if isolated:
                valid.append(key)
    else:
        raise ValueError("Unknown formal task: " + str(op))

    if len(valid) != 1:
        raise ValueError("Expected one valid selection for " + op + ", got " + str(valid))
    return valid[0]
