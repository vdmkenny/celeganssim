"""Minimal xlsx reader, standard library only.

Cook et al. publish their corrected connectome as an Excel workbook rather than
a CSV, and this project depends on numpy and nothing else. An xlsx file is a
zip of XML, so reading the one sheet layout we need takes far less than an
Excel library would cost as a dependency.

Handles what those sheets actually contain: shared strings, inline strings,
numbers, and gaps where a cell is simply absent from the XML. It does not
handle formulas, dates, or styles, and it is not a general reader.
"""

from __future__ import annotations

import re
import zipfile
from xml.etree import ElementTree as ET

NS = {
    "m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "p": "http://schemas.openxmlformats.org/package/2006/relationships",
}
CELL_REF = re.compile(r"^([A-Z]+)(\d+)$")


def _col_index(ref: str) -> int:
    """Column number from a cell reference: A -> 0, Z -> 25, AA -> 26."""
    m = CELL_REF.match(ref)
    letters = m.group(1) if m else ref
    n = 0
    for ch in letters:
        n = n * 26 + (ord(ch) - 64)
    return n - 1


def _shared_strings(zf: zipfile.ZipFile) -> list[str]:
    try:
        root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
    except KeyError:
        return []
    out = []
    for si in root.findall("m:si", NS):
        # A string can be split across several runs; concatenate them.
        out.append("".join(t.text or "" for t in si.iter(f"{{{NS['m']}}}t")))
    return out


def _sheet_paths(zf: zipfile.ZipFile) -> dict[str, str]:
    """Sheet name to its part path, resolved through the workbook relationships."""
    rels = {}
    root = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
    for rel in root.findall("p:Relationship", NS):
        rels[rel.get("Id")] = rel.get("Target")
    out = {}
    wb = ET.fromstring(zf.read("xl/workbook.xml"))
    for sh in wb.iter(f"{{{NS['m']}}}sheet"):
        target = rels.get(sh.get(f"{{{NS['r']}}}id"), "")
        if target.startswith("/"):
            target = target[1:]
        elif not target.startswith("xl/"):
            target = "xl/" + target
        out[sh.get("name")] = target
    return out


def sheet_names(path) -> list[str]:
    with zipfile.ZipFile(path) as zf:
        return list(_sheet_paths(zf))


def read_sheet(path, name: str) -> list[list]:
    """One sheet as a dense list of rows, blanks as None.

    Rows and columns are padded so every row has the same length, because the
    XML omits empty cells entirely and an adjacency matrix is mostly empty.
    """
    with zipfile.ZipFile(path) as zf:
        paths = _sheet_paths(zf)
        if name not in paths:
            raise KeyError(f"no sheet {name!r}; have {list(paths)}")
        strings = _shared_strings(zf)
        root = ET.fromstring(zf.read(paths[name]))

    rows: list[list] = []
    width = 0
    for row in root.iter(f"{{{NS['m']}}}row"):
        cells: dict[int, object] = {}
        for c in row.findall("m:c", NS):
            ref, typ = c.get("r", ""), c.get("t")
            v = c.find("m:v", NS)
            if typ == "s":                      # shared string
                if v is None or v.text is None:
                    continue
                val = strings[int(v.text)]
            elif typ == "inlineStr":
                is_el = c.find("m:is", NS)
                val = "".join(t.text or "" for t in is_el.iter(f"{{{NS['m']}}}t")) \
                    if is_el is not None else None
            elif v is None or v.text is None:
                continue
            else:
                try:
                    val = float(v.text)
                except ValueError:
                    val = v.text
            if val is None or val == "":
                continue
            j = _col_index(ref)
            cells[j] = val
            width = max(width, j + 1)
        idx = int(row.get("r", len(rows) + 1)) - 1
        while len(rows) < idx:
            rows.append({})
        rows.append(cells)

    return [[r.get(j) for j in range(width)] for r in rows]
