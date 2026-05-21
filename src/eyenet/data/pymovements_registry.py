from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pymovements as pm


@dataclass(frozen=True)
class PyMovementsDatasetInfo:
    name: str
    long_name: str | None
    description: str
    root: str
    sampling_rate_hz: float | None
    screen_width_px: float | None
    screen_height_px: float | None
    screen_width_cm: float | None
    screen_height_cm: float | None
    viewing_distance_cm: float | None
    resources: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def list_pymovements_datasets() -> list[str]:
    return list(pm.DatasetLibrary().names())


def get_pymovements_dataset_info(name: str, root: str | Path) -> PyMovementsDatasetInfo:
    definition = pm.DatasetLibrary().get(name)
    experiment = definition.experiment
    screen = getattr(experiment, "screen", None)
    eyetracker = getattr(experiment, "eyetracker", None)
    resources = []
    for resource in getattr(definition, "resources", []) or []:
        source = getattr(resource, "source", None)
        resources.append(
            {
                "content": getattr(resource, "content", None),
                "filename": getattr(source, "filename", None),
                "url": getattr(source, "url", None),
                "md5": getattr(source, "md5", None),
                "filename_pattern": getattr(resource, "filename_pattern", None),
                "load_kwargs": getattr(resource, "load_kwargs", None),
            }
        )

    return PyMovementsDatasetInfo(
        name=definition.name,
        long_name=getattr(definition, "long_name", None),
        description=getattr(definition, "description", "") or "",
        root=str(root),
        sampling_rate_hz=getattr(eyetracker, "sampling_rate", None),
        screen_width_px=getattr(screen, "width_px", None),
        screen_height_px=getattr(screen, "height_px", None),
        screen_width_cm=getattr(screen, "width_cm", None),
        screen_height_cm=getattr(screen, "height_cm", None),
        viewing_distance_cm=getattr(screen, "distance_cm", None),
        resources=resources,
    )


def make_pymovements_dataset(name: str, root: str | Path) -> pm.Dataset:
    return pm.Dataset(name, path=Path(root))
