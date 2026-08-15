"""One audited view of every model parameter.

Values are DEFINED in their domain modules, next to the code that uses them
and the citation that justifies them. This module only COLLECTS them, so the
registry can never drift from the physics: it introspects the params classes
and module constants at import time.

Each entry carries a provenance tag:

  measured    measured in the animal; citation at the definition site
  published   a standard published model's value (Wicks 1996, Kunert 2014, RFT)
  assay       assay geometry -- a modelling choice, named and documented
  tuned       fit by hand to a behavioural target; the target is named at the
              definition site
  inferred    read off a measurement made in a DIFFERENT cell type or
              context, where the quantity itself has never been measured.
              Distinct from tuned: nothing was fitted, and distinct from
              measured, because the animal was never asked this question.
              The inference and what bounds it are stated at the definition
  scripted    a placeholder for a mechanism the model does not have yet --
              these are the numbers the project is trying to delete

Overrides are per domain: construct the params class with different values
(`BodyParams(freq_hz=0.9)`, `dataclasses.replace(SimConfig(), seed=3)`) or use
`override()` below, and pass the result to the relevant subsystem. Nothing
here mutates global state.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

from . import assays, body, lifecycle, nervous_system, simulation


@dataclass(frozen=True)
class Param:
    name: str          # dotted: "body.freq_hz"
    value: Any
    provenance: str    # measured | published | assay | tuned | scripted
    where: str         # module that defines it


def _collect(domain: str, cls, tags: dict[str, str], where: str) -> dict:
    """Pull the public scalar attributes of a params class into the registry.

    Plain classes and dataclasses both expose their defaults as class
    attributes, so vars() covers both. Missing provenance tags are a hard
    error: an untagged parameter is exactly the magic number this registry
    exists to eliminate.
    """
    out = {}
    for name, value in vars(cls).items():
        if name.startswith("_") or not isinstance(value, (int, float, str, bool)):
            continue
        if name not in tags:
            raise KeyError(f"{where}.{name} has no provenance tag")
        out[f"{domain}.{name}"] = Param(f"{domain}.{name}", value,
                                        tags[name], where)
    return out


def _module_consts(domain: str, names: dict[str, str], module) -> dict:
    """Collect named module-level constants (assay geometry etc.)."""
    return {f"{domain}.{n}": Param(f"{domain}.{n}", getattr(module, n), tag,
                                   module.__name__)
            for n, tag in names.items()}


PARAMETERS: dict[str, Param] = {
    **_collect("body", body.BodyParams, body.PROVENANCE, "worm/body.py"),
    **_collect("neural", nervous_system.NeuralParams,
               nervous_system.PROVENANCE, "worm/nervous_system.py"),
    **_collect("sim", simulation.SimConfig, simulation.PROVENANCE,
               "worm/simulation.py"),
    **_collect("life", lifecycle.LifecycleParams, lifecycle.PROVENANCE,
               "worm/lifecycle.py"),
    **_module_consts("assay", assays.CONST_PROVENANCE, assays),
}


def get(name: str) -> Any:
    return PARAMETERS[name].value


def override(params, **changes):
    """A params instance with some values changed, without mutating the original."""
    if dataclasses.is_dataclass(params):
        return dataclasses.replace(params, **changes)
    import copy
    new = copy.copy(params)
    for k, v in changes.items():
        if not hasattr(new, k):
            raise AttributeError(f"{type(params).__name__} has no parameter {k!r}")
        setattr(new, k, v)
    return new


def audit() -> str:
    """Every parameter, its value and its provenance, grouped by domain."""
    lines = []
    by_domain: dict[str, list[Param]] = {}
    for p in PARAMETERS.values():
        by_domain.setdefault(p.name.split(".")[0], []).append(p)
    for domain, params in sorted(by_domain.items()):
        lines.append(domain)
        for p in sorted(params, key=lambda p: p.name):
            short = p.name.split(".", 1)[1]
            lines.append(f"  {short:<24} {p.value!r:<12} {p.provenance}")
    scripted = [p.name for p in PARAMETERS.values() if p.provenance == "scripted"]
    tuned = [p.name for p in PARAMETERS.values() if p.provenance == "tuned"]
    lines.append(f"\n{len(PARAMETERS)} parameters: "
                 f"{len(tuned)} tuned, {len(scripted)} scripted placeholders")
    return "\n".join(lines)


if __name__ == "__main__":
    print(audit())
