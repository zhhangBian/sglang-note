# SPDX-License-Identifier: Apache-2.0
"""Public capabilities and results for native KV cache control."""

import msgspec

from sglang.srt.managers.kv_control.contracts import (
    ACTION_VERSION,
    HASH_SCHEME,
    PROTOCOL_VERSION,
    PUBLIC_ACTIONS,
    ControlError,
    Pin,
    PrefixTarget,
    PublicResult,
    Status,
    TargetOutcome,
    TokensTarget,
)


def public_capabilities(caps):
    directions = caps.get("directions", {})
    pins = caps.get("pin_protections", ()) if caps.get("available", True) else ()
    return dict(
        protocol_version=PROTOCOL_VERSION,
        schema_versions={name: (ACTION_VERSION,) for name in PUBLIC_ACTIONS},
        actions={"sglang.pin": (ACTION_VERSION,)} if pins else {},
        paths={},
        pin_protections=pins,
        pin_query="claim" if pins else None,
        pin_submission=("independent", "attached") if pins else (),
        pin_request_cached="successful_checkpoint_existing_host_prefix"
        if pins
        else None,
        pin_attached_claim_id="returned_effective_claim_id_per_binding"
        if pins
        else None,
        pin_owner_task="optional_caller_identity" if pins else None,
        hash_only_execution=False,
        prefix=dict(
            namespace=caps.get("namespace"),
            layout_id=caps.get("layout_id"),
            page_size=directions.get("page_alignment"),
            components=caps.get("components", ()),
            hash_scheme=HASH_SCHEME,
            token_material="complete_prefix",
        ),
        result_version="1.0",
    )


def normalize_public(payload: Pin, caps):
    capability = public_capabilities(caps)
    protections = (
        capability["pin_protections"]
        if payload.mode == "acquire"
        else caps.get("pin_protections", ())
    )
    conditions = {p["condition"] for p in protections if p["tier"] == "host"}
    requested = {
        condition
        for condition, enabled in (
            ("eviction_exclusion", payload.ttl_seconds is not None),
            ("eviction_priority", payload.priority is not None),
        )
        if enabled
    }
    required = payload.required_protection
    if (
        not conditions
        or payload.location.tier != "host"
        or not requested <= conditions
        or required is not None
        and (
            required not in conditions
            or payload.mode == "acquire"
            and required not in requested
        )
    ):
        raise ControlError(
            "unsupported", "Requested local host protection is unavailable"
        )
    target = payload.target
    if payload.mode != "acquire":
        # Claim identity suffices even after its content/session was removed.
        return payload
    if (
        target != "request_cached"
        and not isinstance(target, PrefixTarget)
        and target.allow_clip
    ):
        raise ControlError("invalid_payload", "PIN requires an unclipped target")
    if isinstance(target, PrefixTarget):
        c = target.compatibility
        expected = capability["prefix"]
        if c.namespace != expected["namespace"]:
            raise ControlError("incompatible_namespace", "Prefix namespace differs")
        if (
            c.layout_id != expected["layout_id"]
            or c.page_size != expected["page_size"]
            or set(c.components) != set(expected["components"])
            or c.hash_scheme != HASH_SCHEME
        ):
            raise ControlError(
                "unsupported", "Prefix layout/page/components/hash scheme differs"
            )
        if target.start_token != 0 or target.token_ids is None:
            raise ControlError(
                "unsupported", "Execution requires complete token material from zero"
            )
        # Derive the cache key from the complete token prefix and logical salt.
        target = TokensTarget(
            namespace=c.namespace,
            token_ids=target.token_ids,
            cache_salt=c.cache_salt or None,
            layout_id=c.layout_id,
        )
    return msgspec.structs.replace(payload, target=target)


def project_result(current, update, sequence):
    """Project owner-confirmed protection outcomes to the public result."""
    public = current.public_result or PublicResult(
        owner_epoch=current.ref.context.engine_epoch,
        state_version=sequence,
        outcome_sequence=sequence,
    )
    accepted = (
        tuple(
            o
            for o in update.target_outcomes
            if "protection_active" in o.completed_steps
        )
        or public.accepted_scope
    )
    missing = tuple(o for o in update.target_outcomes if o.status == Status.MISSING)
    blocked = tuple(
        o
        for o in update.target_outcomes
        if o.status in (Status.DEFERRED, Status.PARTIAL)
    )
    if not update.target_outcomes and update.status in (
        Status.MISSING,
        Status.DEFERRED,
    ):
        scope = tuple(
            msgspec.structs.replace(
                o, status=update.status, code=update.code, reason=update.reason
            )
            for o in public.requested_scope
        )
        if update.status == Status.MISSING:
            missing = scope
        else:
            blocked = scope
    return msgspec.structs.replace(
        public,
        state_version=sequence,
        outcome_sequence=sequence,
        accepted_scope=accepted,
        missing_scope=missing,
        blocked_scope=blocked,
    )


def requested_outcomes(scope):
    return (TargetOutcome(coverage=scope.requested, status=Status.RECEIVED),)
