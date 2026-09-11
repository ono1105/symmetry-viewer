from __future__ import annotations

import json
from pathlib import Path

from ..source_kinds import source_kind_ui_config_json
from ..viewer.animation_path import STATIONARY_ANIMATION_SECONDS


def viewer_constants_json() -> str:
    """Values the browser must agree with Python on, injected at page build.

    The renderer needs these before any API response arrives, so they cannot
    simply ride along on the animation payload. Defining them twice invites the
    two copies to drift apart silently.
    """
    return json.dumps({"stationary_animation_seconds": STATIONARY_ANIMATION_SECONDS})


_HTML_TEMPLATE = Path(__file__).with_name("browser_ui.html").read_text(encoding="utf-8")
HTML = (
    _HTML_TEMPLATE
    .replace("__STRUCTURE_KIND_CONFIG__", source_kind_ui_config_json())
    .replace("__VIEWER_CONSTANTS__", viewer_constants_json())
)
