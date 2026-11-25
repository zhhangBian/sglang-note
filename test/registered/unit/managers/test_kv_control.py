# SPDX-License-Identifier: Apache-2.0
"""PIN wire contracts and shared result, claim and cache salt vectors."""

import json
import unittest
from pathlib import Path

import msgspec

from sglang.srt.managers.kv_control import KVControl
from sglang.srt.managers.kv_control.contracts import (
    PUBLIC_ACTIONS,
    ControlError,
    Pin,
    Status,
    is_native_action,
    parse_action,
)
from sglang.srt.managers.kv_hints import KvHintAction, decode_kv_hints_envelope
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


def wire(value):
    return msgspec.json.decode(msgspec.json.encode(value))


class TestPublicKVProtocol(CustomTestCase):
    fixtures = (
        Path(__file__).resolve().parents[4] / "sgl-router/sgl-kv-orchestrator/fixtures"
    )

    def test_result_and_claim_vectors_remain_separate(self):
        from sglang.srt.managers.kv_control.contracts import (
            ClaimState,
            PublicResult,
            validated,
        )

        fixtures = json.loads((self.fixtures / "results.json").read_text())
        for key, schema in (("results", PublicResult), ("claims", ClaimState)):
            for value in fixtures[key]:
                self.assertEqual(wire(validated(value, schema)), value)
        self.assertTrue(fixtures["claims"][0]["tombstone"])
        self.assertTrue(all(r["claim"] is None for r in fixtures["results"]))

    def test_salt_vectors(self):
        from sglang.srt.managers.kv_control.key_namespace import encode_cache_salt

        for case in json.loads((self.fixtures / "salt.json").read_text()):
            self.assertEqual(
                encode_cache_salt(case["namespace"], case["salt"]), case["encoded"]
            )

    def test_pin_wire_contract(self):
        action = KvHintAction(
            action_id="pin",
            action_type="sglang.pin",
            action_version="1.0",
            payload={
                "target": {"kind": "tokens", "namespace": "ns", "token_ids": [1, 2]},
                "location": {"tier": "host"},
                "mode": "acquire",
                "claim_id": "claim",
                "revision": 1,
                "ttl_seconds": 30.0,
                "priority": -2,
            },
        )
        envelope = decode_kv_hints_envelope(
            {"protocol_version": "0.1", "message_id": "pin", "actions": [wire(action)]}
        )
        self.assertEqual(envelope.actions, [action])
        self.assertEqual(PUBLIC_ACTIONS, ("sglang.pin",))
        self.assertTrue(is_native_action(envelope.actions[0]))
        parsed = parse_action(envelope.actions[0])
        self.assertIsInstance(parsed, Pin)
        self.assertEqual(parsed.target.token_ids, (1, 2))
        self.assertEqual(parsed.ttl_seconds, 30.0)
        self.assertEqual(parsed.priority, -2)
        for fields, code in (
            ({"location": {"tier": "gpu"}}, "unsupported"),
            ({"location": {"tier": "host", "backend": "file"}}, "invalid_payload"),
            ({"ttl_seconds": None, "priority": None}, "invalid_payload"),
            ({"mode": "release"}, "invalid_payload"),
        ):
            with self.subTest(fields=fields), self.assertRaises(ControlError) as error:
                parse_action(
                    msgspec.structs.replace(action, payload=action.payload | fields)
                )
            self.assertEqual(error.exception.code, code)

    def test_unowned_sglang_actions_are_unsupported_independently(self):
        actions = [
            {
                "action_id": name,
                "action_type": f"sglang.{name}",
                "action_version": version,
            }
            for name, version in (
                ("prefetch", "1.0"),
                ("demote", "1.0"),
                ("unknown", "1.0"),
                ("pin", "2.0"),
            )
        ]
        envelope = decode_kv_hints_envelope(
            {"protocol_version": "0.1", "message_id": "unowned", "actions": actions}
        )
        owner = KVControl("engine", "ns", enabled=True)
        result = owner.submit(envelope, owner.context("caller"))
        self.assertEqual(result.unhandled_actions, ())
        self.assertEqual(
            tuple(item.status for item in result.results),
            (Status.UNSUPPORTED,) * len(actions),
        )
        self.assertTrue(
            all(item.update.code == "unsupported" for item in result.results)
        )


if __name__ == "__main__":
    unittest.main()
