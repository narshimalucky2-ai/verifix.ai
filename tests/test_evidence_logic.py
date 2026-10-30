import urllib.parse
import re

def enrich_sources_and_calculate_strength(supporting, contradicting, grounding_chunks, conflicts=None):
    """
    Enriches sources with domain-derived publishers, clusters independence groups,
    and calculates an explainable heuristic evidence_strength_score (0-100).
    """
    conflicts = conflicts or []
    independence_groups = {}
    group_counter = 1

    def process_source_list(sources):
        nonlocal group_counter
        enriched = []
        for s in sources:
            url = s.get("url", "")
            title = s.get("title", "Web Source")
            publisher = s.get("publisher")

            # Derive publisher from URL domain if missing or generic
            if not publisher or publisher.lower() in ("unknown", "web source", ""):
                try:
                    host = urllib.parse.urlsplit(url).hostname or ""
                    publisher = host.replace("www.", "").split(".")[0].capitalize()
                except Exception:
                    publisher = "Unknown"

            # Cluster by publisher for independence group
            norm_pub = publisher.lower().strip()
            if norm_pub not in independence_groups:
                independence_groups[norm_pub] = f"group_{group_counter}"
                group_counter += 1

            ind_group = s.get("independence_group") or independence_groups[norm_pub]

            enriched.append({
                "title": title,
                "url": url,
                "publisher": publisher,
                "published_at": s.get("published_at"),
                "evidence_summary": s.get("evidence_summary", "Retrieved web evidence"),
                "source_type": s.get("source_type", "secondary"),
                "relevance": s.get("relevance", "medium"),
                "reliability_assessment": s.get("reliability_assessment", "Standard web publication; factual consistency evaluated against context."),
                "independence_group": ind_group
            })
        return enriched

    supp_enriched = process_source_list(supporting)
    contra_enriched = process_source_list(contradicting)

    # Count unique independent source groups
    unique_groups = set(s["independence_group"] for s in supp_enriched + contra_enriched)
    num_independent = len(unique_groups)
    has_primary = any(s["source_type"] == "primary" for s in supp_enriched + contra_enriched)
    has_high_relevance = any(s["relevance"] == "high" for s in supp_enriched + contra_enriched)
    has_conflicts = len(conflicts) > 0

    # Heuristic evidence strength calculation (0 to 100)
    # Base on independent evidence density
    score = 30
    score += min(30, num_independent * 10)  # up to +30 for independent sources
    if has_primary:
        score += 15
    if has_high_relevance:
        score += 15

    # Penalties for conflicts and insufficient sources
    if has_conflicts:
        score -= 25
    if num_independent == 0:
        score = 15
    elif num_independent == 1:
        score = min(score, 55)

    # Bound score between 10 and 95 (never 100% since heuristic)
    score = max(10, min(95, score))

    # Qualitative label
    if score >= 75 and num_independent >= 2 and not has_conflicts:
        strength_label = "strong"
    elif score >= 50 and num_independent >= 1:
        strength_label = "moderate"
    elif score >= 30:
        strength_label = "limited"
    else:
        strength_label = "insufficient"

    return supp_enriched, contra_enriched, strength_label, score, num_independent

print("Source & Evidence processor module ready")

