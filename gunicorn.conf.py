"""Mandatory configuration for the supported production HTTP launcher."""
from typing import Any

reload = False


def on_starting(server: Any) -> None:
    # Effective values include both configuration and CLI/environment overrides.
    if server.cfg.reload:
        raise ValueError('Production automatic reload is disabled')


def on_reload(server: Any) -> None:
    on_starting(server)
