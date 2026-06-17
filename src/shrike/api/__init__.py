"""The operation/verb surface: the ``actions`` registry, the ``tools`` MCP
binding, and the ``mcp_adapter`` call policy.

Sits between ``harness/`` (the verbs it drives) and ``server/`` (the host that
exposes it). Speaks the top-level contract (``schemas``, ``errors``) directly.
"""
