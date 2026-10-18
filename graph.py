"""The provenance scene graph.

All reasoning runs on the in-memory SceneGraph (plain Python, easy to test).
Neo4jStore mirrors it into Neo4j so you can browse and query it at
http://localhost:7474 during the demo.

Neo4j model
  (:Entity {id, type, role, scene})
  (:Fact {id, attribute, value, confidence, base_confidence, hedged, quote, scene})
  (:Source {id, type, viewpoint, scene})
  (Entity)-[:HAS_FACT]->(Fact)<-[:ASSERTS]-(Source)
  (Fact)-[:CONFLICTS_WITH]-(Fact)
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Dict, List, Optional

from matching import values_agree
from schema import Fact, origin_of


class SceneGraph:
    def __init__(self, scene_id: str, facts: Optional[List[Fact]] = None,
                 confidence_cfg: Optional[dict] = None):
        self.scene_id = scene_id
        self.facts: List[Fact] = []
        self.confidence_cfg = confidence_cfg or {"corroboration_bonus": 0.2}
        # {entity: {origin: role}} for people whose sources named different roles (extraction).
        self.role_votes: Dict[str, Dict[str, str]] = {}
        if facts:
            self.add(facts)

    # ------------------------------------------------------------ editing
    def add(self, facts: List[Fact]) -> None:
        self.facts.extend(facts)
        self.recompute_confidence()

    def next_fact_id(self) -> str:
        nums = [int(f.fact_id[1:]) for f in self.facts if f.fact_id[1:].isdigit()]
        return f"f{(max(nums) + 1 if nums else 1):03d}"

    def recompute_confidence(self) -> None:
        """confidence = base + bonus for every OTHER independent source that agrees."""
        bonus = self.confidence_cfg["corroboration_bonus"]
        for group in self.groups().values():
            for f in group:
                agreeing = {origin_of(o.source) for o in group if values_agree(o.value, f.value)}
                f.support = sorted(agreeing)
                extra = len(agreeing - {origin_of(f.source)})
                f.confidence = round(min(0.99, f.base_confidence + bonus * extra), 3)

    # ------------------------------------------------------------ queries
    def groups(self) -> Dict[tuple, List[Fact]]:
        g: Dict[tuple, List[Fact]] = defaultdict(list)
        for f in self.facts:
            g[(f.entity, f.attribute)].append(f)
        return g

    def entities(self) -> Dict[str, dict]:
        out: Dict[str, dict] = {}
        for f in self.facts:
            out.setdefault(f.entity, {"entity_type": f.entity_type, "role": f.role})
        return out

    def facts_by_source(self, source_prefix: str) -> List[Fact]:
        return [f for f in self.facts if origin_of(f.source) == source_prefix]

    def get(self, fact_id: str) -> Fact:
        return next(f for f in self.facts if f.fact_id == fact_id)

    @staticmethod
    def support_level(f: Fact) -> str:
        """confirmed = CCTV or 2+ independent sources; single = one witness only."""
        if len(f.support) >= 2 or "cctv" in f.support:
            return "confirmed"
        return "single"

    # ------------------------------------------------------------ io
    def to_rows(self) -> List[dict]:
        return [{
            "id": f.fact_id, "entity": f.entity, "role": f.role, "attribute": f.attribute,
            "value": f.value, "source": f.source, "confidence": f.confidence,
            "support": ", ".join(f.support), "hedged": f.hedged,
        } for f in self.facts]

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as fh:
            json.dump({"scene_id": self.scene_id, "facts": [f.to_dict() for f in self.facts],
                       "role_votes": self.role_votes}, fh, indent=2)

    @staticmethod
    def load(path: str, confidence_cfg: Optional[dict] = None) -> "SceneGraph":
        with open(path) as fh:
            data = json.load(fh)
        graph = SceneGraph(data["scene_id"], [Fact.from_dict(d) for d in data["facts"]],
                           confidence_cfg)
        graph.role_votes = data.get("role_votes", {})
        return graph


class Neo4jStore:
    def __init__(self, uri: str, user: str, password: str):
        from neo4j import GraphDatabase

        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.driver.verify_connectivity()

    def close(self) -> None:
        self.driver.close()

    def sync(self, graph: SceneGraph, conflicts: Optional[list] = None) -> None:
        """Replace this scene's subgraph with the current state."""
        facts = [f.to_dict() for f in graph.facts]
        pairs = []
        for c in conflicts or []:
            ids = [opt["fact_ids"] for opt in c["options"]]
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    pairs += [{"a": a, "b": b} for a in ids[i] for b in ids[j]]

        with self.driver.session() as s:
            s.run("MATCH (n {scene: $scene}) DETACH DELETE n", scene=graph.scene_id)
            s.run("""
                UNWIND $facts AS f
                MERGE (e:Entity {id: f.entity, scene: $scene})
                  SET e.type = f.entity_type, e.role = f.role
                MERGE (src:Source {id: f.source, scene: $scene})
                  SET src.type = f.source_type, src.viewpoint = f.viewpoint
                CREATE (x:Fact {id: f.fact_id, scene: $scene, attribute: f.attribute,
                                value: f.value, confidence: f.confidence,
                                base_confidence: f.base_confidence, hedged: f.hedged,
                                quote: f.quote, support: f.support})
                CREATE (e)-[:HAS_FACT]->(x)
                CREATE (src)-[:ASSERTS]->(x)
            """, facts=facts, scene=graph.scene_id)
            if pairs:
                s.run("""
                    UNWIND $pairs AS p
                    MATCH (a:Fact {id: p.a, scene: $scene}), (b:Fact {id: p.b, scene: $scene})
                    MERGE (a)-[:CONFLICTS_WITH]->(b)
                """, pairs=pairs, scene=graph.scene_id)


def draw_graph_png(graph: SceneGraph, conflict_fact_ids: Optional[set] = None):
    """Small picture of the graph for the UI: entities in the middle, facts around them.
    Green = confirmed, yellow = one witness, red = in conflict."""
    import io

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import networkx as nx
    from PIL import Image

    conflict_fact_ids = conflict_fact_ids or set()
    G = nx.Graph()
    colors, labels, sizes = {}, {}, {}
    for eid, e in graph.entities().items():
        G.add_node(eid)
        colors[eid], sizes[eid] = "#6b8fd6", 1400
        labels[eid] = f"{eid}\n({e['role']})" if e["role"] != "none" else eid
    for f in graph.facts:
        G.add_node(f.fact_id)
        G.add_edge(f.entity, f.fact_id)
        if f.fact_id in conflict_fact_ids:
            colors[f.fact_id] = "#e06666"
        elif graph.support_level(f) == "confirmed":
            colors[f.fact_id] = "#6aa84f"
        else:
            colors[f.fact_id] = "#f1c232"
        sizes[f.fact_id] = 300
        labels[f.fact_id] = f"{f.attribute}:\n{f.value}\n[{f.source}]"

    # Entities on a big circle, each entity's facts on a small ring around it.
    import math
    pos = {}
    ents = list(graph.entities())
    for i, eid in enumerate(ents):
        a = 2 * math.pi * i / max(1, len(ents))
        cx, cy = 4 * math.cos(a), 4 * math.sin(a)
        pos[eid] = (cx, cy)
        own = [f for f in graph.facts if f.entity == eid]
        for j, f in enumerate(own):
            b = 2 * math.pi * j / max(1, len(own)) + a / 2
            pos[f.fact_id] = (cx + 1.6 * math.cos(b), cy + 1.6 * math.sin(b))

    fig, ax = plt.subplots(figsize=(15, 12))
    nodes = list(G.nodes)
    nx.draw_networkx(G, pos, ax=ax, nodelist=nodes, node_color=[colors[n] for n in nodes],
                     node_size=[sizes[n] for n in nodes], labels=labels, font_size=6,
                     edge_color="#bbbbbb")
    ax.axis("off")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)
