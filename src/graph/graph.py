"""An undirected edge joins distinct labels that differ in exactly one field."""

import statistics

from .fields import FEATURE_FIELDS


def delta(left, right):
    """Return the changed fields; fail on incomplete input instead of guessing."""
    changed = []
    for field in FEATURE_FIELDS:
        if left[field] != right[field]:
            changed.append(field)
    return changed


def is_one_feature_pair(left_label, right_label, left, right):
    return left_label != right_label and len(delta(left, right)) == 1


def build_graph(nodes):
    """Group vectors by their other 16 values to avoid a quadratic full scan."""
    neighbors = {}
    for label in nodes:
        neighbors[label] = {}
    edges = []
    for field in FEATURE_FIELDS:
        buckets = {}
        for label in sorted(nodes):
            vector = nodes[label]["vector"]
            signature = []
            for other in FEATURE_FIELDS:
                if other != field:
                    signature.append(vector[other])
            key = tuple(signature)
            buckets.setdefault(key, []).append(label)
        for labels in buckets.values():
            for i in range(len(labels)):
                for j in range(i + 1, len(labels)):
                    left, right = labels[i], labels[j]
                    if nodes[left]["vector"][field] == nodes[right]["vector"][field]:
                        continue
                    edges.append({"left": left, "right": right, "field": field})
                    neighbors[left][right] = field
                    neighbors[right][left] = field
    return {"nodes": nodes, "edges": edges, "neighbors": neighbors}


def graph_audit(graph):
    degree = []
    isolated = 0
    for label in graph["nodes"]:
        count = len(graph["neighbors"][label])
        degree.append(count)
        if count == 0:
            isolated += 1
    per_field = {}
    for field in FEATURE_FIELDS:
        field_degrees = {}
        values = []
        count = 0
        for edge in graph["edges"]:
            if edge["field"] != field:
                continue
            count += 1
            for label in [edge["left"], edge["right"]]:
                field_degrees[label] = field_degrees.get(label, 0) + 1
                value = graph["nodes"][label]["vector"][field]
                if value not in values:
                    values.append(value)
        per_field[field] = {
            "edges": count, "participating_nodes": len(field_degrees),
            "values_on_edges": len(values),
            "median_degree": statistics.median(field_degrees.values()) if field_degrees else 0,
            "maximum_degree": max(field_degrees.values(), default=0),
        }
    return {
        "nodes": len(graph["nodes"]), "edges": len(graph["edges"]),
        "isolated_nodes": isolated,
        "mean_degree": statistics.mean(degree) if degree else 0,
        "median_degree": statistics.median(degree) if degree else 0,
        "maximum_degree": max(degree, default=0), "per_field": per_field,
    }
