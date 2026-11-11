import os
import unittest
from unittest.mock import patch

import feedback_pipeline as fp


class RoutingConfigTests(unittest.TestCase):
    def test_import_does_not_require_api_key(self):
        with patch.object(fp, "_openrouter_api_key", return_value=None):
            with patch.dict(os.environ, {}, clear=True):
                with self.assertRaises(RuntimeError):
                    fp.ensure_api_key()

    def test_default_routing_uses_current_cost_aware_models(self):
        routing = fp.build_model_routing()

        self.assertEqual(routing.generation, "gpt-6.1-sol@medium")
        self.assertEqual(routing.scoring, "gpt-6.1-terra@low")
        self.assertEqual(routing.verification, "gpt-6.1-sol@low")
        self.assertEqual(routing.rewrite, "gpt-6-luna@none")
        self.assertEqual(routing.clustering, "gpt-6-luna@none")
        self.assertEqual(routing.editorial_triage, "gpt-6.1-sol@medium")
        self.assertEqual(routing.meta_review, "gpt-6.1-sol@medium")
        self.assertEqual(routing.escalation, "gpt-6.1-sol@medium")

    def test_baseline_preset_reproduces_previous_routing(self):
        routing = fp.build_model_routing(preset="baseline-gpt-5.6")

        self.assertEqual(routing.generation, "gpt-5.6-terra")
        self.assertEqual(routing.scoring, "gpt-5.6-terra")
        self.assertEqual(routing.verification, "gpt-5.6-terra")
        self.assertEqual(routing.rewrite, "gpt-5.6-luna")
        self.assertEqual(routing.clustering, "gpt-5.6-luna")
        self.assertEqual(routing.editorial_triage, "gpt-5.6-sol")
        self.assertEqual(routing.meta_review, "gpt-5.6-sol")
        self.assertEqual(routing.escalation, "gpt-5.6-sol")
        with self.assertRaises(ValueError):
            fp.build_model_routing(preset="no-such-preset")

    def test_generation_override_does_not_change_other_stages(self):
        routing = fp.build_model_routing(gen_model="gpt-5.6-sol", preset="baseline-gpt-5.6")

        self.assertEqual(routing.generation, "gpt-5.6-sol")
        self.assertEqual(routing.scoring, "gpt-5.6-terra")
        self.assertEqual(routing.rewrite, "gpt-5.6-luna")
        self.assertEqual(routing.meta_review, "gpt-5.6-sol")

    def test_default_panel_rotates_vendors_and_baseline_has_none(self):
        with patch.dict(os.environ, {"LLM_PROVIDER": "openrouter"}, clear=True):
            routing = fp.build_model_routing()
            self.assertEqual(routing.generation_panel, fp.GENERATION_PANEL)
            self.assertEqual(routing.critiques_per_reviewer, 2)
            models = fp.generation_models_for_workers(routing, 8)
            self.assertEqual(models[0], "gpt-6.1-sol@medium")
            self.assertEqual(models[1], "anthropic/claude-opus-5.5@medium")
            self.assertEqual(models[2], "google/gemini-4-carbon@medium")
            self.assertEqual(models[3], models[0])
            # Paired roles (reviewers 1-2, 3-4, 5-6) get different models.
            for first in (0, 2, 4):
                self.assertNotEqual(models[first], models[first + 1])

            baseline = fp.build_model_routing(preset="baseline-gpt-5.6")
            self.assertEqual(baseline.generation_panel, ())
            self.assertEqual(baseline.critiques_per_reviewer, 1)
            self.assertEqual(
                fp.generation_models_for_workers(baseline, 3), ["gpt-5.6-terra"] * 3
            )

    def test_panel_is_dropped_for_explicit_generator_or_openai_provider(self):
        with patch.dict(os.environ, {"LLM_PROVIDER": "openrouter"}, clear=True):
            explicit = fp.build_model_routing(gen_model="anthropic/claude-opus-5.5@high")
            self.assertEqual(explicit.generation_panel, ())
            custom = fp.build_model_routing(
                generation_panel=["gpt-6.1-sol@low", "x-ai/grok-4.7"], critiques_per_reviewer=3
            )
            self.assertEqual(custom.generation_panel, ("gpt-6.1-sol@low", "x-ai/grok-4.7"))
            self.assertEqual(custom.critiques_per_reviewer, 3)
            self.assertEqual(fp.build_model_routing(generation_panel=[]).generation_panel, ())
            with self.assertRaises(ValueError):
                fp.build_model_routing(critiques_per_reviewer=0)
        with patch.dict(os.environ, {"LLM_PROVIDER": "openai"}, clear=True):
            direct = fp.build_model_routing()
            self.assertEqual(direct.generation_panel, ())
            self.assertEqual(direct.generation, "gpt-6.1-sol@medium")

    def test_mixed_stage_is_priced_per_model(self):
        from types import SimpleNamespace

        tracker = fp.UsageTracker()
        tracker.set_stage("generation")
        usage = SimpleNamespace(prompt_tokens=1_000_000, completion_tokens=0)
        tracker.record(usage, model="gpt-6.1-sol@medium")
        tracker.record(usage, model="anthropic/claude-opus-5.5@medium")
        with patch.dict(os.environ, {"LLM_PROVIDER": "openrouter"}, clear=True):
            actual = fp.compute_actual_cost(tracker, routing=fp.build_model_routing())

        generation = actual["stages"]["generation"]
        self.assertAlmostEqual(generation["cost_usd"], 2.00 + 4.00)
        self.assertEqual(
            generation["model"], "gpt-6.1-sol@medium + anthropic/claude-opus-5.5@medium"
        )

    def test_stage_overrides_and_all_stages_precedence(self):
        routing = fp.build_model_routing(
            all_stages_model="gpt-6.1-terra@low",
            meta_model="gpt-6.1-sol@high",
        )

        self.assertEqual(routing.generation, "gpt-6.1-terra@low")
        self.assertEqual(routing.verification, "gpt-6.1-terra@low")
        self.assertEqual(routing.meta_review, "gpt-6.1-sol@high")

    def test_model_effort_suffix_is_validated_and_sent(self):
        messages = [{"role": "user", "content": "x"}]

        kwargs = fp._chat_request_kwargs(messages, "gpt-6.1-sol@high")
        self.assertEqual(kwargs["model"], "gpt-6.1-sol")
        self.assertEqual(kwargs["reasoning_effort"], "high")
        self.assertNotIn("reasoning_effort", fp._chat_request_kwargs(messages, "gpt-6.1-sol"))
        self.assertEqual(
            fp._lookup_pricing_model("gpt-6.1-sol@high"), fp.MODEL_PRICING["gpt-6.1-sol"]
        )
        with self.assertRaises(ValueError):
            fp.build_model_routing(gen_model="gpt-6.1-sol@none")
        with self.assertRaises(ValueError):
            fp.build_model_routing(gen_model="gpt-6.1-sol@extreme")

    def test_estimate_grows_with_generation_reasoning_effort(self):
        text = "Title\n\nThis paper estimates a difference-in-differences model."
        low = fp.estimate_cost_before_run(
            text, routing=fp.build_model_routing(gen_model="gpt-6.1-sol@low")
        )
        high = fp.estimate_cost_before_run(
            text, routing=fp.build_model_routing(gen_model="gpt-6.1-sol@high")
        )

        self.assertGreater(
            high["stages"]["generation"]["cost_usd"], low["stages"]["generation"]["cost_usd"]
        )

    def test_invalid_model_is_rejected(self):
        with self.assertRaises(ValueError):
            fp.build_model_routing(gen_model="not-a-model")

    def test_current_model_options_put_current_models_first(self):
        options = fp.current_model_options()

        self.assertLess(options.index("gpt-5.6-sol"), options.index("gpt-5.5"))
        self.assertLess(options.index("gpt-5.6-terra"), options.index("gpt-5.4-mini"))
        self.assertLess(options.index("gpt-5.6-luna"), options.index("gpt-5.4-nano"))

    def test_chat_request_kwargs_preserve_effective_reasoning_defaults(self):
        messages = [{"role": "user", "content": "Review this."}]

        terra = fp._chat_request_kwargs(messages, "gpt-5.6-terra")
        luna = fp._chat_request_kwargs(messages, "gpt-5.6-luna")
        sol = fp._chat_request_kwargs(messages, "gpt-5.6-sol")
        legacy = fp._chat_request_kwargs(messages, "gpt-5.4-mini")

        self.assertEqual(terra["reasoning_effort"], "none")
        self.assertEqual(luna["reasoning_effort"], "none")
        self.assertEqual(sol["reasoning_effort"], "medium")
        self.assertNotIn("reasoning_effort", legacy)


class ProviderTests(unittest.TestCase):
    def test_explicit_provider_wins_and_is_validated(self):
        with patch.dict(os.environ, {"LLM_PROVIDER": "openai"}, clear=True):
            self.assertEqual(fp.llm_provider(), "openai")
        with patch.dict(os.environ, {"LLM_PROVIDER": "other"}, clear=True):
            with self.assertRaises(ValueError):
                fp.llm_provider()

    def test_provider_falls_back_to_openai_without_openrouter_key(self):
        with patch.object(fp, "_openrouter_api_key", return_value=None):
            with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=True):
                self.assertEqual(fp.llm_provider(), "openai")
            with patch.dict(os.environ, {}, clear=True):
                self.assertEqual(fp.llm_provider(), "openrouter")
                with self.assertRaises(RuntimeError):
                    fp.ensure_api_key()

    def test_model_ids_are_translated_per_provider(self):
        self.assertEqual(
            fp.provider_model_id("gpt-5.6-terra", "openrouter"), "openai/gpt-5.6-terra"
        )
        self.assertEqual(
            fp.provider_model_id("anthropic/claude-sonnet-5.5", "openrouter"),
            "anthropic/claude-sonnet-5.5",
        )
        self.assertEqual(fp.provider_model_id("gpt-5.6-terra", "openai"), "gpt-5.6-terra")
        with self.assertRaises(ValueError):
            fp.provider_model_id("anthropic/claude-sonnet-5.5", "openai")

    def test_openrouter_only_models_need_openrouter(self):
        with patch.dict(os.environ, {"LLM_PROVIDER": "openai"}, clear=True):
            with self.assertRaises(ValueError):
                fp.build_model_routing(gen_model="anthropic/claude-sonnet-5.5")
            self.assertNotIn("anthropic/claude-sonnet-5.5", fp.current_model_options())
        with patch.dict(os.environ, {"LLM_PROVIDER": "openrouter"}, clear=True):
            routing = fp.build_model_routing(gen_model="anthropic/claude-sonnet-5.5")
            self.assertEqual(routing.generation, "anthropic/claude-sonnet-5.5")
            self.assertIn("anthropic/claude-sonnet-5.5", fp.current_model_options())

    def test_openrouter_chat_kwargs_add_routing_preferences(self):
        schema_format = fp.json_schema_response_format("s", {"type": "object"})
        base = fp._chat_request_kwargs(
            [{"role": "user", "content": "x"}], "gpt-5.6-terra", response_format=schema_format
        )
        with patch.dict(os.environ, {"LLM_PROVIDER": "openrouter"}, clear=True):
            adapted = fp._provider_chat_kwargs(base)
            plain = fp._provider_chat_kwargs(
                fp._chat_request_kwargs([{"role": "user", "content": "x"}], "gpt-5.6-sol")
            )
        self.assertEqual(adapted["model"], "openai/gpt-5.6-terra")
        self.assertNotIn("reasoning_effort", adapted)
        self.assertEqual(adapted["extra_body"]["reasoning"], {"effort": "none"})
        self.assertEqual(
            adapted["extra_body"]["provider"],
            {"data_collection": "deny", "require_parameters": True},
        )
        self.assertEqual(base["reasoning_effort"], "none")
        self.assertEqual(plain["extra_body"]["provider"], {"data_collection": "deny"})
        self.assertEqual(base["model"], "gpt-5.6-terra")
        self.assertNotIn("extra_body", base)

        with patch.dict(os.environ, {"LLM_PROVIDER": "openai"}, clear=True):
            direct = fp._provider_chat_kwargs(base)
        self.assertEqual(direct["model"], "gpt-5.6-terra")
        self.assertEqual(direct["reasoning_effort"], "none")
        self.assertNotIn("extra_body", direct)

    def test_batch_mode_rejects_openrouter_only_models(self):
        fp._require_batch_model("gpt-5.6-terra")
        with self.assertRaises(ValueError):
            fp._require_batch_model("anthropic/claude-sonnet-5.5")

    def test_actual_cost_prefers_provider_reported_cost(self):
        from types import SimpleNamespace

        tracker = fp.UsageTracker()
        tracker.set_stage("generation")
        tracker.record(SimpleNamespace(prompt_tokens=1000, completion_tokens=100, cost=0.5))
        tracker.record(SimpleNamespace(prompt_tokens=1000, completion_tokens=100, cost=0.25))
        tracker.set_stage("scoring")
        tracker.record(SimpleNamespace(prompt_tokens=1000, completion_tokens=100, cost=0.5))
        tracker.record(SimpleNamespace(prompt_tokens=1000, completion_tokens=100))

        actual = fp.compute_actual_cost(
            tracker, routing=fp.build_model_routing(preset="baseline-gpt-5.6")
        )

        generation = actual["stages"]["generation"]
        self.assertEqual(generation["cost_source"], "provider_reported")
        self.assertAlmostEqual(generation["cost_usd"], 0.75)
        scoring = actual["stages"]["scoring"]
        self.assertEqual(scoring["cost_source"], "registry_prices")
        self.assertAlmostEqual(scoring["cost_usd"], 2000 * 2.00 / 1e6 + 200 * 12.00 / 1e6)


class StructuredOutputTests(unittest.TestCase):
    def test_json_schema_response_format(self):
        schema = {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
            "additionalProperties": False,
        }

        payload = fp.json_schema_response_format("answer_schema", schema)

        self.assertEqual(payload["type"], "json_schema")
        self.assertEqual(payload["json_schema"]["name"], "answer_schema")
        self.assertIs(payload["json_schema"]["schema"], schema)
        self.assertTrue(payload["json_schema"]["strict"])


class CostEstimateTests(unittest.TestCase):
    def test_current_model_pricing_matches_official_standard_rates(self):
        expected_per_million = {
            "gpt-5.6-sol": {"input": 5.00, "cached_input": 0.50, "output": 30.00},
            "gpt-5.6-terra": {"input": 2.00, "cached_input": 0.20, "output": 12.00},
            "gpt-5.6-luna": {"input": 0.20, "cached_input": 0.02, "output": 1.20},
        }

        for model, expected in expected_per_million.items():
            with self.subTest(model=model):
                actual = fp.MODEL_PRICING[model]
                self.assertEqual(actual["input"] * 1e6, expected["input"])
                self.assertEqual(actual["cached_input"] * 1e6, expected["cached_input"])
                self.assertEqual(actual["output"] * 1e6, expected["output"])

    def test_estimate_uses_routing_models_without_api_key(self):
        with patch.dict(os.environ, {}, clear=True):
            estimate = fp.estimate_cost_before_run(
                "Title\n\nThis paper estimates a difference-in-differences model.",
                num_agents=8,
                top_k=3,
                routing=fp.build_model_routing(preset="baseline-gpt-5.6"),
            )

        stages = estimate["stages"]
        self.assertEqual(stages["generation"]["model"], "gpt-5.6-terra")
        self.assertEqual(stages["scoring"]["model"], "gpt-5.6-terra")
        self.assertEqual(stages["score_escalation"]["model"], "gpt-5.6-sol")
        self.assertEqual(stages["verification"]["model"], "gpt-5.6-terra")
        self.assertEqual(stages["rewrite"]["model"], "gpt-5.6-luna")
        self.assertNotIn("critique", stages)
        self.assertNotIn("re_scoring", stages)
        self.assertEqual(stages["clustering"]["model"], "gpt-5.6-luna")
        self.assertEqual(stages["meta_review"]["model"], "gpt-5.6-sol")
        self.assertGreater(estimate["estimated_total_cost_usd"], 0)


if __name__ == "__main__":
    unittest.main()
