"""Conflict detection, hypothesis branching, gap finding and fact selection.

Pure Python, no models. Conflicts are preserved, never merged away: each
hypothesis is one internally consistent choice for every conflict.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from graph import SceneGraph
from matching import values_agree
from schema import GAP_ATTRIBUTES, LOCATION_ID, MULTI_VALUED, Fact, origin_of

ROLE_ORDER = {"suspect": 0, "victim": 1, "bystander": 3, "none": 4}


def cluster_values(facts: List[Fact]) -> List[List[Fact]]:
    """Group facts whose values agree. The strongest fact leads each cluster."""
    clusters: List[List[Fact]] = []
    for f in sorted(facts, key=lambda x: -x.confidence):
        for c in clusters:
            if values_agree(c[0].value, f.value):
                c.append(f)
                break
        else:
            clusters.append([f])
    return clusters


def find_conflicts(graph: SceneGraph) -> List[dict]:
    conflicts = []
    for (entity, attribute), facts in graph.groups().items():
        if attribute in MULTI_VALUED:
            continue
        clusters = cluster_values(facts)
        origins = {origin_of(f.source) for f in facts}
        if len(clusters) < 2 or len(origins) < 2:
            continue
        options = []
        for c in clusters:
            options.append({
                "value": c[0].value,
                "fact_ids": [f.fact_id for f in c],
                "sources": sorted({f.source for f in c}),
                "confidence": max(f.confidence for f in c),
            })
        options.sort(key=lambda o: -o["confidence"])
        conflicts.append({
            "id": f"c{len(conflicts) + 1}", "entity": entity,
            "entity_type": facts[0].entity_type, "role": facts[0].role,
            "attribute": attribute, "options": options,
        })
    return conflicts


def find_role_conflicts(graph: SceneGraph) -> List[dict]:
    """Sources that named different roles for one merged person. Kept as conflicts (kind "role")
    so they are visible, but roles are soft: they don't branch hypotheses or change what is drawn."""
    out = []
    entities = graph.entities()
    for entity, votes in sorted(graph.role_votes.items()):
        if entity not in entities or len(set(votes.values())) < 2:
            continue
        by_role: Dict[str, List[str]] = {}
        for origin, role in sorted(votes.items()):
            by_role.setdefault(role, []).append(origin)
        winner = entities[entity]["role"]
        options = [{"value": r, "fact_ids": [], "sources": srcs, "confidence": round(len(srcs) / len(votes), 3)}
                   for r, srcs in by_role.items()]
        options.sort(key=lambda o: (o["value"] != winner, -o["confidence"]))
        out.append({"id": f"r{len(out) + 1}", "kind": "role", "entity": entity, "entity_type": "person",
                    "role": winner, "attribute": "role", "options": options})
    return out


def build_hypotheses(conflicts: List[dict], max_hypotheses: int = 2) -> List[dict]:
    """H1 = most supported option for every conflict; H2 = the runner-up for every conflict.
    Kept to 2 so the number of images stays small; raise max_hypotheses for more.
    Role conflicts don't branch: the voted role is used in every hypothesis."""
    conflicts = [c for c in conflicts if c.get("kind") != "role"]
    if not conflicts:
        return [{"id": "H1", "name": "Main reconstruction", "choices": {},
                 "description": "No conflicting evidence."}]
    hyps = []
    n = min(max_hypotheses, max(len(c["options"]) for c in conflicts))
    for k in range(n):
        choices = {c["id"]: min(k, len(c["options"]) - 1) for c in conflicts}
        desc = "; ".join(
            f"{c['entity']}.{c['attribute']} = '{c['options'][choices[c['id']]]['value']}' "
            f"({', '.join(c['options'][choices[c['id']]]['sources'])})" for c in conflicts)
        name = "Most supported version" if k == 0 else f"Alternative version {k}"
        hyps.append({"id": f"H{k + 1}", "name": name, "choices": choices, "description": desc})
    return hyps


def find_gaps(graph: SceneGraph) -> List[dict]:
    """Attributes in GAP_ATTRIBUTES that no source has mentioned."""
    gaps = []
    entities = graph.entities()
    if LOCATION_ID not in entities:
        entities = dict(entities, **{LOCATION_ID: {"entity_type": "location", "role": "none"}})
    known = {(f.entity, f.attribute) for f in graph.facts}
    for eid, e in entities.items():
        for attr in GAP_ATTRIBUTES[e["entity_type"]]:
            if (eid, attr) not in known:
                gaps.append({"entity": eid, "entity_type": e["entity_type"],
                             "role": e["role"], "attribute": attr})
    gaps.sort(key=lambda g: (ROLE_ORDER.get(g["role"], 2) if g["entity"] != LOCATION_ID else 2,
                             g["entity"]))
    return gaps


def select_facts(graph: SceneGraph, conflicts: List[dict], hypothesis: dict,
                 render_threshold: float, hypothesis_threshold: float
                 ) -> Tuple[List[Fact], List[Fact], set]:
    """Decide what gets drawn for one hypothesis.

    Returns (included, withheld, contested_fact_ids). One representative fact
    per distinct value. Unsupported detail is withheld, never drawn.
    """
    chosen: Dict[tuple, dict] = {}
    for c in conflicts:
        chosen[(c["entity"], c["attribute"])] = c["options"][hypothesis["choices"].get(c["id"], 0)]

    included: List[Fact] = []
    withheld: List[Fact] = []
    contested: set = set()
    for key, facts in graph.groups().items():
        if key in chosen:
            opt = chosen[key]
            rep = max((f for f in facts if f.fact_id in opt["fact_ids"]),
                      key=lambda f: f.confidence)
            if opt["confidence"] >= hypothesis_threshold:
                included.append(rep)
                contested.add(rep.fact_id)
            else:
                withheld.append(rep)
            continue
        for cluster in cluster_values(facts):
            rep = cluster[0]
            (included if rep.confidence >= render_threshold else withheld).append(rep)
    return included, withheld, contested
