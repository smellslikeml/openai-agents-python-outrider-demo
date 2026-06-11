"""Honor in-band "Recuse Signal" deny banners emitted by tools and servers.

Adapted from "Will the Agent Recuse Itself? Measuring LLM-Agent Compliance
with In-Band Access-Deny Signals" (https://arxiv.org/abs/2606.06460v1).

The paper proposes a third access mode that sits between "let the agent in"
and "hard-fail it": a lightweight, published in-band deny signal -- the
*Recuse Signal* -- that a server emits over an existing channel (an SSH
banner, a PostgreSQL NOTICE) asking a connecting automated agent to
voluntarily withdraw. It is a cooperative governance control, the robots.txt
analogue for live access -- explicitly NOT a security boundary.

The paper's empirical finding is that the signal is *cooperative*: agents
recuse when it is present, but an explicit operator-authorization framing can
flip a capable agent to proceed. This module turns that finding into a
deterministic enforcement layer on top of the SDK's tool-output guardrail:

* When a tool's output carries a Recuse Signal banner, the agent recuses by
  default (the tool result is withheld from the model with a recusal notice).
* A trusted, call-site `operator_authorization` flag -- and only that flag --
  can override the recusal and let the result through. The banner text can
  never authorize itself; an ``authorized=...`` field embedded in the banner
  is deliberately ignored so untrusted server data cannot opt itself out of
  the recusal policy.

Banner mini-standard (one line, case-insensitive key):

    X-Recuse-Signal: 1.0; action=recuse; reason="maintenance window";
        policy="https://ops.example/recuse"; scope="db-primary"

Only ``action=recuse`` triggers a recusal; any other action (or a missing
banner) is treated as "no signal" and the tool proceeds normally.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from agents import (
    ToolGuardrailFunctionOutput,
    ToolOutputGuardrail,
    ToolOutputGuardrailData,
    tool_output_guardrail,
)

# The published banner marker. Servers emit this over an existing channel.
RECUSE_SIGNAL_MARKER = "X-Recuse-Signal"

# Matches the marker line and captures everything after the colon.
_BANNER_RE = re.compile(
    rf"{re.escape(RECUSE_SIGNAL_MARKER)}\s*:\s*(?P<body>[^\r\n]+)", re.IGNORECASE
)

# Matches ``key=value`` and ``key="quoted value"`` parameter pairs.
_PARAM_RE = re.compile(r'(?P<key>[A-Za-z_][\w-]*)\s*=\s*(?:"(?P<q>[^"]*)"|(?P<v>[^;]*))')

# Banner-supplied fields that must never grant authorization. Authorization is
# a trusted call-site decision, not something serialized banner data may claim.
_UNTRUSTED_AUTH_KEYS = {"authorized", "authorization", "override", "allow"}


@dataclass
class RecuseSignal:
    """A parsed Recuse Signal banner.

    Attributes:
        action: The requested action (e.g. ``"recuse"``). Only ``"recuse"``
            asks the agent to withdraw.
        reason: Human-readable reason for the deny signal, if provided.
        policy: URL of the published recusal policy, if provided.
        scope: Resource scope the signal applies to, if provided.
        version: Mini-standard version declared by the banner, if any.
        raw: The raw banner body, verbatim.
    """

    action: str
    reason: str | None = None
    policy: str | None = None
    scope: str | None = None
    version: str | None = None
    raw: str = ""

    @property
    def asks_recusal(self) -> bool:
        """True when the banner asks the agent to voluntarily withdraw."""
        return self.action.lower() == "recuse"


def parse_recuse_signal(text: str) -> RecuseSignal | None:
    """Parse a Recuse Signal banner out of arbitrary tool output text.

    Args:
        text: Tool output (or any channel text) that may embed a banner.

    Returns:
        The parsed :class:`RecuseSignal`, or ``None`` when no banner is found.
    """
    match = _BANNER_RE.search(text or "")
    if not match:
        return None

    body = match.group("body").strip()
    params: dict[str, str] = {}
    leading_version: str | None = None

    for param in _PARAM_RE.finditer(body):
        key = param.group("key").lower()
        value = param.group("q") if param.group("q") is not None else (param.group("v") or "")
        value = value.strip()
        # Deliberately drop any self-asserted authorization fields: a banner
        # may ask for recusal, never grant itself permission to proceed.
        if key in _UNTRUSTED_AUTH_KEYS:
            continue
        params[key] = value

    # A bare leading token (e.g. "1.0" before the first key=value) is the
    # version declaration in the mini-standard.
    first_token = body.split(";", 1)[0].strip()
    if first_token and "=" not in first_token:
        leading_version = first_token

    return RecuseSignal(
        action=params.get("action", "recuse"),
        reason=params.get("reason"),
        policy=params.get("policy"),
        scope=params.get("scope"),
        version=params.get("version", leading_version),
        raw=body,
    )


def _recusal_message(signal: RecuseSignal) -> str:
    parts = ["The resource published a Recuse Signal asking automated agents to withdraw."]
    if signal.reason:
        parts.append(f"Reason: {signal.reason}.")
    if signal.scope:
        parts.append(f"Scope: {signal.scope}.")
    if signal.policy:
        parts.append(f"Policy: {signal.policy}.")
    parts.append("Recusing from this resource. Do not retry without operator authorization.")
    return " ".join(parts)


def make_recuse_guardrail(
    *,
    operator_authorization: bool = False,
    enforce: bool = False,
    name: str | None = None,
) -> ToolOutputGuardrail[Any]:
    """Build a tool-output guardrail that honors in-band Recuse Signals.

    The returned guardrail inspects each tool output for a Recuse Signal
    banner and decides cooperatively:

    * No banner, or a non-recuse action -> allow the tool result through.
    * Banner present and ``operator_authorization`` is True -> allow, recording
      that a trusted operator override was applied (mirrors the paper's finding
      that an operator-authorization framing flips a capable agent to proceed).
    * Banner present, no authorization, ``enforce`` False -> reject the tool
      content with a recusal notice so the agent withdraws but the run
      continues (the default cooperative mode).
    * Banner present, no authorization, ``enforce`` True -> raise an exception
      to halt the run (treat the signal as a hard stop).

    ``operator_authorization`` is a trusted call-site decision. The banner text
    can never set it; self-asserted authorization fields in the banner are
    ignored during parsing.

    Args:
        operator_authorization: When True, a trusted operator has authorized
            proceeding despite the signal.
        enforce: When True, recusal halts the run instead of withholding the
            single tool result.
        name: Optional guardrail name (defaults to the function name).

    Returns:
        A :class:`ToolOutputGuardrail` ready to attach to a tool's
        ``tool_output_guardrails`` list.
    """

    @tool_output_guardrail(name=name or "honor_recuse_signal")
    def honor_recuse_signal(data: ToolOutputGuardrailData) -> ToolGuardrailFunctionOutput:
        signal = parse_recuse_signal(str(data.output))
        if signal is None or not signal.asks_recusal:
            return ToolGuardrailFunctionOutput.allow(output_info={"recuse_signal": None})

        info: dict[str, Any] = {
            "recuse_signal": {
                "action": signal.action,
                "reason": signal.reason,
                "policy": signal.policy,
                "scope": signal.scope,
                "version": signal.version,
            },
            "tool": data.context.tool_name,
        }

        if operator_authorization:
            # Trusted operator override: proceed and surface that we did.
            info["operator_authorization"] = True
            return ToolGuardrailFunctionOutput.allow(output_info=info)

        if enforce:
            return ToolGuardrailFunctionOutput.raise_exception(output_info=info)

        return ToolGuardrailFunctionOutput.reject_content(
            message=_recusal_message(signal),
            output_info=info,
        )

    return honor_recuse_signal
