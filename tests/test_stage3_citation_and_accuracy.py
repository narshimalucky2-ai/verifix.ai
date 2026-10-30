import unittest
import os
import sys

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from main import (
    enrich_sources_and_calculate_strength,
    enforce_decision_rules,
    WIRE_SERVICES
)

class TestStage3CitationAndAccuracy(unittest.TestCase):

    def test_01_wire_service_clustering(self):
        """
        Verify that multiple regional or syndicated outlets reprinting wire stories
        (AP, Reuters) are clustered into shared wire groups and do not artificially
        inflate independent source counts.
        """
        sources = [
            {"title": "AP Wire: Mars Mission Launched", "url": "https://apnews.com/article/mars1", "publisher": "Associated Press"},
            {"title": "Local Herald: Mars Rocket Launches", "url": "https://localherald.com/mars1", "publisher": "Local Herald (AP Syndicated)"},
            {"title": "Reuters Dispatch: NASA Mars Launch", "url": "https://reuters.com/science/mars1", "publisher": "Reuters"}
        ]
        supp_e, _, strength, score, num_ind = enrich_sources_and_calculate_strength(
            supporting=sources,
            contradicting=[],
            grounding_chunks=[]
        )

        # 2 AP stories + 1 Reuters story -> exactly 2 independent groups (not 3)
        self.assertEqual(num_ind, 2)
        ap_groups = [s["independence_group"] for s in supp_e if "associated_press" in s["independence_group"]]
        self.assertEqual(len(ap_groups), 2)
        self.assertEqual(ap_groups[0], ap_groups[1])
        self.assertTrue(supp_e[0]["syndicated_wire"])
        self.assertTrue(supp_e[1]["syndicated_wire"])
        self.assertTrue(supp_e[2]["syndicated_wire"])

    def test_02_citation_grounding_validation(self):
        """
        Verify that sources with URLs present in grounding_chunks are marked verified_grounded,
        while hallucinated or ungrounded URLs are flagged and penalized.
        """
        grounding_chunks = [
            {"web": {"uri": "https://www.cdc.gov/flu/treatment.htm", "title": "CDC Flu Treatment"}}
        ]
        grounded_source = [
            {"title": "CDC Guidelines", "url": "https://www.cdc.gov/flu/treatment.htm", "publisher": "CDC"}
        ]
        ungrounded_source = [
            {"title": "Bogus Blog", "url": "https://unverified-miracle-cure-blog.org/cure", "publisher": "BogusBlog"}
        ]

        # Case A: Grounded URL
        supp_g, _, _, score_g, _ = enrich_sources_and_calculate_strength(
            supporting=grounded_source,
            contradicting=[],
            grounding_chunks=grounding_chunks
        )
        self.assertTrue(supp_g[0]["citation_grounded"])
        self.assertEqual(supp_g[0]["grounding_status"], "verified_grounded")

        # Case B: Ungrounded URL
        supp_u, _, _, score_u, _ = enrich_sources_and_calculate_strength(
            supporting=ungrounded_source,
            contradicting=[],
            grounding_chunks=grounding_chunks
        )
        self.assertFalse(supp_u[0]["citation_grounded"])
        self.assertEqual(supp_u[0]["grounding_status"], "ungrounded_citation")
        # Ungrounded source receives -15 penalty
        self.assertLess(score_u, score_g)

    def test_03_stale_citation_detection(self):
        """
        Verify that time-sensitive claims flag outdated citations.
        """
        stale_sources = [
            {"title": "Old Economic Survey", "url": "https://site.com/econ", "publisher": "EconLab", "published_at": "2019-05-10"}
        ]
        context = "Currently in 2026, the latest inflation rate in the region is record breaking."

        supp_e, _, _, _, _ = enrich_sources_and_calculate_strength(
            supporting=stale_sources,
            contradicting=[],
            grounding_chunks=[],
            context_text=context
        )
        self.assertTrue(supp_e[0]["is_stale_citation"])
        self.assertIn("2019", supp_e[0]["staleness_warning"])

    def test_04_decision_rule_preserves_misleading_context(self):
        """
        Ensure context exaggeration/distortion does not falsely collapse to 'unverified'.
        """
        conflicts = ["Article headline asserts breakthrough, but body text states preliminary in-vitro findings only."]
        verdict, status = enforce_decision_rules(
            verdict="misleading",
            evidence_status="misleading_context",
            conflicts=conflicts,
            num_independent=1
        )
        self.assertEqual(verdict, "misleading")
        self.assertEqual(status, "misleading_context")

    def test_05_decision_rule_disputed_claims_map_to_unverified(self):
        """
        Ensure genuine source disputes (competing credible findings) map to 'unverified'.
        """
        conflicts = ["Credible archives dispute the exact origin; historians present conflicting documentary evidence."]
        verdict, status = enforce_decision_rules(
            verdict="verified",
            evidence_status="supported",
            conflicts=conflicts,
            num_independent=2
        )
        self.assertEqual(verdict, "unverified")
        self.assertEqual(status, "insufficient_evidence")

    def test_06_decision_rule_zero_independent_sources_cannot_verify(self):
        """
        Ensure a supported claim with 0 independent source groups cannot be verified.
        """
        verdict, status = enforce_decision_rules(
            verdict="verified",
            evidence_status="supported",
            conflicts=[],
            num_independent=0
        )
        self.assertEqual(verdict, "unverified")
        self.assertEqual(status, "insufficient_evidence")


if __name__ == "__main__":
    unittest.main()
