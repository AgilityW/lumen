"""Lumen exception hierarchy — all custom exceptions live here.

Library code MUST NOT call sys.exit(1). Instead, raise the appropriate
LumenError subclass. The CLI entry point catches these and exits cleanly.
"""


class LumenError(Exception):
    """Base for all Lumen custom exceptions."""


class ConfigError(LumenError):
    """Configuration or API key errors."""


class APIError(LumenError):
    """LLM API call failures (auth, rate limit, network)."""


class ParseError(LumenError):
    """File parsing errors (format, corruption, missing text layer)."""


class UserInterrupt(LumenError):
    """User-initiated exit (e.g. quit at GATE review). Not an error."""
