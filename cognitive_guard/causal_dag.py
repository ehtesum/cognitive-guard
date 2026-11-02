"""
CognitiveGuard - Module 2: Causal DAG Provenance Pruner
Audits the LLM's Chain-of-Thought (CoT) reasoning graph against verified source facts.

Mathematical Formulation:
    Reasoning Graph: G_CoT = (V_CoT, E_inferred)
    Evidence Graph:  G_facts = (V_facts, E_facts)
    Provenance Ratio: η(G_CoT) = |E_grounded| / |E_inferred|

If η(G_CoT) < τ_provenance or if the terminal edge leading to H_f lacks
textual entailment in G_facts, the ungrounded causal link is pruned.
"""

from typing import List, Dict, Any, Set, Tuple, Optional
from dataclasses import dataclass, field
import re
import networkx as nx


@dataclass
class CausalEdge:
    source_node: str
    target_node: str
    relation: str
    evidence_support: Optional[str] = None
    is_grounded: bool = False
    confidence: float = 1.0


@dataclass
class AuditReport:
    total_inferred_edges: int
    grounded_edges: int
    hallucinated_edges: int
    provenance_ratio: float
    is_manipulated: bool
    pruned_edges: List[Tuple[str, str]]
    critical_unsupported_claims: List[str]


class CausalDAGPruner:
    """
    Parses reasoning text into a Directed Acyclic Graph (DAG) of causal claims
    and verifies whether each inferred dependency is supported by the source facts.
    """

    # Common causal and temporal connectors exploited by generative montage
    CAUSAL_CONNECTORS = [
        r"\bbecause\b",
        r"\btherefore\b",
        r"\bas a result of\b",
        r"\bled to\b",
        r"\bcaused\b",
        r"\bwhich means that\b",
        r"\bproves that\b",
        r"\bmarks the first\b",
        r"\bconsequently\b",
        r"\bindicates that\b",
        r"\bindicating that\b",
        r"\bthus\b",
        r"\bhence\b",
        r"\bconfirming that\b"
    ]

    def __init__(self, provenance_threshold: float = 0.65):
        """
        Args:
            provenance_threshold: Minimum ratio of grounded edges required (default: 0.65).
        """
        self.provenance_threshold = provenance_threshold
        self.fact_graph = nx.DiGraph()
        self.cot_graph = nx.DiGraph()

    def build_evidence_graph(self, posts: List[Dict[str, Any]]) -> nx.DiGraph:
        """
        Construct a knowledge graph of verified facts from the incoming evidence posts.
        """
        self.fact_graph.clear()
        
        for p in posts:
            text = p.get("text", "")
            post_id = str(p.get("post_id", ""))
            
            # Segment text into discrete sentences / clauses
            clauses = [c.strip() for c in re.split(r"[.;\n]", text) if len(c.strip()) > 15]
            
            for i, clause in enumerate(clauses):
                node_id = f"fact_{post_id}_{i}"
                self.fact_graph.add_node(
                    node_id,
                    text=clause,
                    post_id=post_id,
                    type="verified_fact"
                )
                # Link chronological co-occurring statements within the same post
                if i > 0:
                    prev_node = f"fact_{post_id}_{i-1}"
                    self.fact_graph.add_edge(prev_node, node_id, relation="contextual_co_occurrence")

        return self.fact_graph

    def extract_cot_causal_dag(self, rationale_text: str) -> nx.DiGraph:
        """
        Parse Chain-of-Thought (CoT) reasoning text into a Causal DAG (G_CoT).
        Extracts antecedent -> consequent relationships based on causal connectors.
        Handles both mid-sentence connectors and sentence-initial connectors ('Because X, Y', 'Therefore, Z').
        """
        self.cot_graph.clear()
        
        sentences = [s.strip() for s in re.split(r"[.\n]", rationale_text) if len(s.strip()) > 10]
        node_counter = 0
        prev_node = None

        for sent in sentences:
            # Check for sentence-initial "Because X, Y" pattern
            initial_because = re.match(r"^because\s+(.+?),\s*(.+)$", sent, re.IGNORECASE)
            if initial_because:
                cause_text = initial_because.group(1).strip()
                effect_text = initial_because.group(2).strip()
                
                u = f"cot_{node_counter}"
                v = f"cot_{node_counter+1}"
                node_counter += 2
                
                self.cot_graph.add_node(u, text=cause_text, type="inferred_cause")
                self.cot_graph.add_node(v, text=effect_text, type="inferred_effect")
                self.cot_graph.add_edge(u, v, relation="because", original_sentence=sent)
                prev_node = v
                continue

            # Check for sentence-initial "Therefore / Consequently, X"
            initial_consequent = re.match(r"^(therefore|consequently|thus|hence),\s*(.+)$", sent, re.IGNORECASE)
            if initial_consequent and prev_node is not None:
                connector = initial_consequent.group(1).lower()
                effect_text = initial_consequent.group(2).strip()
                
                v = f"cot_{node_counter}"
                node_counter += 1
                self.cot_graph.add_node(v, text=effect_text, type="inferred_effect")
                self.cot_graph.add_edge(prev_node, v, relation=connector, original_sentence=sent)
                prev_node = v
                continue

            # Check for mid-sentence causal connectors ("X because Y", "X proves that Y", etc.)
            pattern = "|".join(self.CAUSAL_CONNECTORS)
            match = re.search(pattern, sent, re.IGNORECASE)
            if match:
                connector = match.group(0).lower()
                parts = re.split(pattern, sent, flags=re.IGNORECASE)
                if len(parts) >= 2 and len(parts[0].strip()) > 5 and len(parts[1].strip()) > 5:
                    cause_text = parts[0].strip()
                    effect_text = parts[1].strip()
                    
                    u = f"cot_{node_counter}"
                    v = f"cot_{node_counter+1}"
                    node_counter += 2
                    
                    self.cot_graph.add_node(u, text=cause_text, type="inferred_cause")
                    self.cot_graph.add_node(v, text=effect_text, type="inferred_effect")
                    self.cot_graph.add_edge(u, v, relation=connector, original_sentence=sent)
                    prev_node = v
                    continue

            # Sequential step without explicit connector
            node_id = f"cot_{node_counter}"
            node_counter += 1
            self.cot_graph.add_node(node_id, text=sent, type="inferred_observation")
            if prev_node is not None:
                self.cot_graph.add_edge(prev_node, node_id, relation="implicit_followup")
            prev_node = node_id

        return self.cot_graph

    # Stopwords specific to news/event feeds that must not trigger false causal grounding matches
    EVENT_STOPWORDS = {
        "attack", "attacks", "incident", "police", "officer", "officers", "people",
        "scene", "event", "events", "about", "their", "there", "where", "after",
        "before", "today", "morning", "report", "reports", "world", "street", "streets"
    }

    def audit_provenance(self, target_hypothesis: Optional[str] = None) -> AuditReport:
        """
        Audit the extracted G_CoT against the fact graph G_facts.
        Flags unsupported causal jumps (Causal Illusion / Kuleshov Hallucinations).
        """
        if self.cot_graph.number_of_edges() == 0:
            return AuditReport(
                total_inferred_edges=0,
                grounded_edges=0,
                hallucinated_edges=0,
                provenance_ratio=1.0,
                is_manipulated=False,
                pruned_edges=[],
                critical_unsupported_claims=[]
            )

        grounded_count = 0
        hallucinated_count = 0
        pruned_edges = []
        critical_unsupported = []

        # Extract all verified fact texts
        fact_texts = [data.get("text", "").lower() for _, data in self.fact_graph.nodes(data=True)]

        for u, v, data in list(self.cot_graph.edges(data=True)):
            u_text = self.cot_graph.nodes[u].get("text", "").lower()
            v_text = self.cot_graph.nodes[v].get("text", "").lower()
            relation = data.get("relation", "")

            # Check if this edge asserts the target hypothesis conclusion
            is_target_jump = False
            if target_hypothesis and self._is_target_related(v_text, target_hypothesis):
                # Verify if the target claim itself exists in verified facts
                if not self._verify_target_in_facts(target_hypothesis, fact_texts):
                    is_target_jump = True

            link_grounded = False
            if not is_target_jump:
                u_supported = self._verify_textual_grounding(u_text, fact_texts)
                v_supported = self._verify_textual_grounding(v_text, fact_texts)
                if relation == "implicit_followup":
                    # Sequential observation is grounded if both statements exist in verified facts
                    link_grounded = u_supported and v_supported
                else:
                    # Explicit causal claim requires verification that facts state this causal relationship
                    link_grounded = u_supported and v_supported and self._verify_causal_grounding(u_text, v_text, fact_texts)

            if link_grounded:
                grounded_count += 1
                data["is_grounded"] = True
            else:
                hallucinated_count += 1
                data["is_grounded"] = False
                pruned_edges.append((u, v))
                
                claim_repr = f"'{u_text}' --[{relation}]--> '{v_text}'"
                if is_target_jump:
                    claim_repr += f" (CRITICAL: Target hypothesis '{target_hypothesis}' is NOT in source facts!)"
                critical_unsupported.append(claim_repr)

        total_edges = grounded_count + hallucinated_count
        ratio = (grounded_count / total_edges) if total_edges > 0 else 1.0
        
        # Attack flagged if provenance ratio is low OR if an unsupported leap directly asserts target hypothesis
        has_critical_target_jump = any("CRITICAL: Target hypothesis" in c for c in critical_unsupported)
        is_manipulated = (ratio < self.provenance_threshold) or has_critical_target_jump

        # Perform pruning of hallucinated edges in the CoT DAG
        for u, v in pruned_edges:
            if self.cot_graph.has_edge(u, v):
                self.cot_graph.remove_edge(u, v)

        return AuditReport(
            total_inferred_edges=total_edges,
            grounded_edges=grounded_count,
            hallucinated_edges=hallucinated_count,
            provenance_ratio=round(ratio, 3),
            is_manipulated=is_manipulated,
            pruned_edges=pruned_edges,
            critical_unsupported_claims=critical_unsupported
        )

    def _verify_textual_grounding(self, phrase: str, fact_texts: List[str]) -> bool:
        """Check if content keywords in the phrase appear in verified facts."""
        words = [w for w in re.findall(r"\w+", phrase.lower()) if len(w) > 4 and w not in self.EVENT_STOPWORDS]
        if not words:
            return True
        matches = sum(1 for w in words if any(w in fact for fact in fact_texts))
        return (matches / len(words)) >= 0.50

    def _verify_causal_grounding(self, cause: str, effect: str, fact_texts: List[str]) -> bool:
        """
        Verify if cause and effect have co-occurrence and semantic link in the source facts.
        """
        cause_keywords = [w for w in re.findall(r"\w+", cause.lower()) if len(w) > 4 and w not in self.EVENT_STOPWORDS]
        effect_keywords = [w for w in re.findall(r"\w+", effect.lower()) if len(w) > 4 and w not in self.EVENT_STOPWORDS]
        
        if not cause_keywords or not effect_keywords:
            return False

        for fact in fact_texts:
            has_cause = any(k in fact for k in cause_keywords)
            has_effect = any(k in fact for k in effect_keywords)
            if has_cause and has_effect:
                return True
        return False

    def _verify_target_in_facts(self, target_hypothesis: str, fact_texts: List[str]) -> bool:
        """Verify if key distinctive predicates of the target conclusion actually exist in source facts."""
        key_phrases = [
            "first victim", "first casualty", "first to die", "killed first",
            "initiated the attack", "started the attack", "planned the attack"
        ]
        target_lower = target_hypothesis.lower()
        for phrase in key_phrases:
            if phrase in target_lower:
                # Must explicitly appear in facts
                if not any(phrase in f for f in fact_texts):
                    return False
        return True

    def _is_target_related(self, text: str, target: str) -> bool:
        target_words = [w for w in re.findall(r"\w+", target.lower()) if len(w) > 4 and w not in self.EVENT_STOPWORDS]
        if not target_words:
            return False
        matches = sum(1 for w in target_words if w in text.lower())
        return (matches / len(target_words)) >= 0.40

    def print_audit_report(self, report: AuditReport):
        """Print ASCII formatted provenance report."""
        print("\n" + "=" * 80)
        print("COGNITIVEGUARD: CAUSAL DAG PROVENANCE AUDIT REPORT")
        print("=" * 80)
        print(f"Total Inferred Causal Edges: {report.total_inferred_edges}")
        print(f"Grounded Edges (Verified):   {report.grounded_edges}")
        print(f"Hallucinated Edges (Pruned): {report.hallucinated_edges}")
        print(f"Provenance Ratio eta(G_CoT): {report.provenance_ratio:.3f} (Threshold: {self.provenance_threshold})")
        print(f"Status:                      {'[ALERT! MANIPULATION DETECTED]' if report.is_manipulated else '[OK GROUNDED REASONING]'}")
        print("-" * 80)
        if report.critical_unsupported_claims:
            print("Pruned Hallucinated Causal Jumps:")
            for i, c in enumerate(report.critical_unsupported_claims, 1):
                print(f"  {i}. {c}")
        else:
            print("All inferred reasoning paths are verified by source facts.")
        print("=" * 80 + "\n")
