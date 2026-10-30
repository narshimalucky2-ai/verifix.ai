import unittest
from unittest.mock import patch, MagicMock
import os
import sys
import json

# Ensure project root is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from main import get_gemini_api_keys, analyze_with_gemini

class TestGeminiApiKeyFallback(unittest.TestCase):

    def test_01_get_gemini_api_keys_order(self):
        """
        Verify that get_gemini_api_keys() extracts numbered keys in priority order.
        """
        keys = get_gemini_api_keys()
        self.assertGreaterEqual(len(keys), 3)
        # Check that keys are non-empty strings
        for k in keys:
            self.assertTrue(len(k) > 10)

    @patch("requests.post")
    def test_02_key1_rate_limited_falls_back_to_key2(self, mock_post):
        """
        If Key 1 returns HTTP 429, the system must immediately try Key 2 and succeed.
        """
        # Call 1: HTTP 429
        mock_429 = MagicMock()
        mock_429.status_code = 429
        mock_429.text = "Quota exceeded for key 1"
        mock_429.json.return_value = {"error": {"message": "Quota exceeded for key 1"}}

        # Call 2: HTTP 200 Success
        mock_200 = MagicMock()
        mock_200.status_code = 200
        mock_200.json.return_value = {
            "candidates": [{
                "content": {
                    "parts": [{
                        "text": json.dumps({
                            "claim": "Sample factual claim",
                            "verdict": "verified",
                            "evidence_status": "supported",
                            "summary": "Verified via fallback key 2.",
                            "reasoning": ["Found matching records on key 2"],
                            "evidence_strength": "strong",
                            "confidence_explanation": "Supported by 2 independent sources.",
                            "supporting_evidence": [
                                {"title": "Gov Source", "url": "https://agency.gov/doc", "publisher": "Agency", "source_type": "primary", "relevance": "high"},
                                {"title": "News Source", "url": "https://reuters.com/news", "publisher": "Reuters", "source_type": "secondary", "relevance": "high"}
                            ],
                            "contradicting_evidence": [],
                            "conflicts": [],
                            "limitations": []
                        })
                    }]
                },
                "groundingMetadata": {
                    "webSearchQueries": ["sample claim query"],
                    "groundingChunks": [
                        {"web": {"uri": "https://agency.gov/doc", "title": "Gov Source"}},
                        {"web": {"uri": "https://reuters.com/news", "title": "News Source"}}
                    ]
                }
            }]
        }

        mock_post.side_effect = [mock_429, mock_200]

        res = analyze_with_gemini("Sample factual claim", "text")

        self.assertEqual(res["verdict"], "verified")
        self.assertEqual(res["api_key_used"], "key_2")
        self.assertTrue(res["fallback_used"])
        self.assertEqual(mock_post.call_count, 2)

    @patch("requests.post")
    def test_03_key1_and_key2_rate_limited_falls_back_to_key3(self, mock_post):
        """
        If Key 1 and Key 2 both return HTTP 429, Key 3 must be invoked and succeed.
        """
        mock_429_k1 = MagicMock(status_code=429, text="Quota exceeded k1")
        mock_429_k1.json.return_value = {"error": {"message": "Quota exceeded k1"}}

        mock_429_k2 = MagicMock(status_code=429, text="Quota exceeded k2")
        mock_429_k2.json.return_value = {"error": {"message": "Quota exceeded k2"}}

        mock_200_k3 = MagicMock(status_code=200)
        mock_200_k3.json.return_value = {
            "candidates": [{
                "content": {
                    "parts": [{
                        "text": json.dumps({
                            "claim": "Claim on Key 3",
                            "verdict": "verified",
                            "evidence_status": "supported",
                            "summary": "Verified via fallback key 3.",
                            "reasoning": ["Records verified"],
                            "evidence_strength": "strong",
                            "supporting_evidence": [
                                {"title": "WHO", "url": "https://who.int/record", "publisher": "WHO"}
                            ],
                            "contradicting_evidence": [],
                            "conflicts": [],
                            "limitations": []
                        })
                    }]
                },
                "groundingMetadata": {
                    "webSearchQueries": ["claim query 3"],
                    "groundingChunks": [{"web": {"uri": "https://who.int/record", "title": "WHO"}}]
                }
            }]
        }

        mock_post.side_effect = [mock_429_k1, mock_429_k2, mock_200_k3]

        res = analyze_with_gemini("Claim on Key 3", "text")

        self.assertEqual(res["verdict"], "verified")
        self.assertEqual(res["api_key_used"], "key_3")
        self.assertTrue(res["fallback_used"])
        self.assertEqual(mock_post.call_count, 3)

    @patch("requests.post")
    def test_04_all_keys_exhausted_honest_unverified(self, mock_post):
        """
        If all configured keys return 429, system returns honest unverified state.
        """
        keys = get_gemini_api_keys()
        mock_429 = MagicMock(status_code=429, text="Quota exceeded")
        mock_429.json.return_value = {"error": {"message": "Quota exceeded"}}

        mock_post.side_effect = [mock_429] * len(keys)

        res = analyze_with_gemini("Test claim", "text")

        self.assertEqual(res["verdict"], "unverified")
        self.assertEqual(res["evidence_status"], "insufficient_evidence")
        self.assertFalse(res["search_performed"])
        self.assertIn("All", res["summary"])
        self.assertIn("failed", res["summary"])
        self.assertEqual(mock_post.call_count, len(keys))


if __name__ == "__main__":
    unittest.main()

