from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal


SourceKind = Literal["crystal", "molecule", "empty"]

SOURCE_KIND_CRYSTAL = "crystal"
SOURCE_KIND_MOLECULE = "molecule"
SOURCE_KIND_EMPTY = "empty"


@dataclass(frozen=True)
class SourceKindUi:
    label: str
    input_label: str
    symmetry_label: str
    operation_panel_title: str
    view_center_label: str


SOURCE_KIND_UI: dict[str, SourceKindUi] = {
    SOURCE_KIND_CRYSTAL: SourceKindUi(
        label="Crystal",
        input_label="CIF crystal",
        symmetry_label="Space group",
        operation_panel_title="Operations",
        view_center_label="View center [x y z] - fractional",
    ),
    SOURCE_KIND_MOLECULE: SourceKindUi(
        label="Molecule",
        input_label="XYZ molecule",
        symmetry_label="Point group",
        operation_panel_title="Molecular Operations",
        view_center_label="View center [x y z] - Cartesian \u212b",
    ),
    SOURCE_KIND_EMPTY: SourceKindUi(
        label="Empty",
        input_label="structure",
        symmetry_label="Symmetry",
        operation_panel_title="Operations",
        view_center_label="View center [x y z]",
    ),
}


def normalize_source_kind(value: str | None, *, default: str = SOURCE_KIND_CRYSTAL) -> str:
    if value in SOURCE_KIND_UI:
        return value
    return default


def source_kind_ui_config_json() -> str:
    return json.dumps(
        {
            key: {
                "label": value.label,
                "inputLabel": value.input_label,
                "symmetryLabel": value.symmetry_label,
                "operationPanelTitle": value.operation_panel_title,
                "viewCenterLabel": value.view_center_label,
            }
            for key, value in SOURCE_KIND_UI.items()
        },
        ensure_ascii=True,
        sort_keys=True,
    )
