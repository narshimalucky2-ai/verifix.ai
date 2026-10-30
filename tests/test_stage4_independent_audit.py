import unittest
import os
import sys

# Ensure project root is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from main import (
    enrich_sources_and_calculate_strength,
    enforce_decision_rules,
    app
)

class TestStage4IndependentAudit(unittest.TestCase):

    def test_01_unknown_independence_status(self):
        """
        Sources from unknown or missing publishers must receive independence_status='unknown',
        and must NOT inflate the independent source count.
        """
        sources = [
            {"title": "Anonymous Blog Post", "url": "https://random-blog.xyz/post", "publisher": "Web Source"},
            {"title": "Unidentified Forum Thread", "url": "https://forum-mirror.net/thread", "publisher": "unknown"}
        ]
        supp_e, _, strength, score, num_ind = enrich_sources_and_calculate_strength(sources, [], [])

        self.assertEqual(num_ind, 0)
        self.assertEqual(supp_e[0]["independence_status"], "unknown")
        self.assertEqual(supp_e[1]["independence_status"], "unknown")
        self.assertEqual(supp_e[0]["independence_group"], "unknown_provenance")

    def test_02_unknown_sources_cannot_yield_verified(self):
        """
        A claim supported only by unknown/unprovenanced sources must NEVER yield 'verified'.
        """
        verdict, status = enforce_decision_rules(
            verdict="verified",
            evidence_status="supported",
            conflicts=[],
            num_independent=0
        )
        self.assertEqual(verdict, "unverified")
        self.assertEqual(status, "insufficient_evidence")

    def test_03_pr_distributor_clustering(self):
        """
        Verify that PR Newswire and Business Wire syndications are categorized as
        syndicated_pr and clustered into a single independent group.
        """
        sources = [
            {"title": "Company Q3 Earnings Release", "url": "https://prnewswire.com/news-releases/123", "publisher": "PR Newswire"},
            {"title": "Financial Portal: Company Q3 Earnings", "url": "https://financeportal.com/pr/123", "publisher": "PR Newswire"}
        ]
        supp_e, _, _, _, num_ind = enrich_sources_and_calculate_strength(sources, [], [])

        self.assertEqual(num_ind, 1)
        self.assertEqual(supp_e[0]["independence_status"], "syndicated_pr")
        self.assertEqual(supp_e[1]["independence_status"], "syndicated_pr")
        self.assertEqual(supp_e[0]["independence_group"], supp_e[1]["independence_group"])

    def test_04_divergence_vs_genuine_dispute(self):
        """
        Distinguish between non-contradictory cohort/definition divergences
        and genuine source disputes.
        """
        # Case A: Methodological divergence (not a factual dispute)
        divergence_conflicts = [
            "Study A surveyed adult cohort while Study B evaluated pediatric patients; differing definitions rather than a factual dispute."
        ]
        verdict_a, status_a = enforce_decision_rules(
            verdict="verified",
            evidence_status="supported",
            conflicts=divergence_conflicts,
            num_independent=2
        )
        self.assertEqual(verdict_a, "verified")
        self.assertEqual(status_a, "supported")

        # Case B: Genuine source dispute (competing contradictory findings)
        dispute_conflicts = [
            "Government archive and independent audit present competing and conflicting financial totals."
        ]
        verdict_b, status_b = enforce_decision_rules(
            verdict="verified",
            evidence_status="supported",
            conflicts=dispute_conflicts,
            num_independent=2
        )
        self.assertEqual(verdict_b, "unverified")
        self.assertEqual(status_b, "insufficient_evidence")

    def test_05_semantic_support_and_accessibility_status(self):
        """
        Verify that sources default to 'not_semantically_verified' and 'not_checked'
        for external URL accessibility, avoiding fabricated verification claims.
        """
        sources = [
            {"title": "Peer-Reviewed Study", "url": "https://nature.com/articles/123", "publisher": "Nature"}
        ]
        supp_e, _, _, _, _ = enrich_sources_and_calculate_strength(sources, [], [])

        self.assertEqual(supp_e[0]["semantic_support_status"], "not_semantically_verified")
        self.assertEqual(supp_e[0]["url_accessibility"], "not_checked")
        self.assertIn("not independently verified server-side", supp_e[0]["semantic_validation_note"])

    def test_06_health_endpoint_stage4_capabilities(self):
        """
        Verify that the /api/health endpoint advertises Stage 4 audit and independence capabilities.
        """
        client = app.test_client()
        res = client.get("/api/health")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["version"], "4.0-stage4")
        self.assertIn("semantic_citation_audit", data["capabilities"])
        self.assertIn("unknown_independence_handling", data["capabilities"])


if __name__ == "__main__":
    unittest.main()

