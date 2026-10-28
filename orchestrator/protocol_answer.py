"""'What is the protocol for <rhythm>?' answered without the LLM.

Structured care-protocol steps (`common/protocols/care_protocols.yaml`, in the graph) when the rhythm
has them; otherwise the protocol document's own sections for that rhythm, read verbatim. The 3B model
mixed the SVT section into VT answers ("stable VT: adenosine"), so no text here is generated.
Spoken answers stop after `SPOKEN_POINTS` points and say what is left.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from common.event_types import event_type_from_text

from .report import spoken_event

SPOKEN_POINTS = 3
# "the full protocol": no cap. "the rest" / "tell me more": what the capped answer left out.
_ALL = re.compile(r"\b(?:all|full|whole|complete|entire|every)\b", re.IGNORECASE)
_REST = re.compile(r"\b(?:rest|remaining|more|continue|go on)\b", re.IGNORECASE)
# A heading word the caller can ask for ("long-term management of PVC") to pick sections.
_SECTION_WORDS = ("acute", "emergency", "long term", "chronic", "stable", "unstable", "pulseless",
                  "prevention", "post resuscitation")
_VALUE_ONLY = {"vital", "name", "drug", "modality", "periodicity", "when", "route", "duration"}
_STEP = re.compile(r"^\s*\d+\.\s*\[(\w+)\]\s*(.*?)\s*(?:\((.*)\))?\s*$")


@dataclass
class ProtocolAnswer:
    text: str
    citations: list[str] = field(default_factory=list)


def speak_step(rendered: str) -> str:
    """'4. [escalation] escalation (trigger=HR > 150, escalate_to=on-call, within=15 min)' ->
    'Escalation: if HR > 150, escalate to on-call, within 15 min'."""
    m = _STEP.match(rendered or "")
    if not m:
        return (rendered or "").strip()
    kind, body, detail = m.groups()
    label = kind.replace("_", " ").capitalize()
    said = [] if body.replace(" ", "_") == kind else [body]
    for item in re.split(r",\s*(?=\w+=)", detail or ""):
        if "=" not in item:
            continue
        key, value = (s.strip() for s in item.split("=", 1))
        said.append(value if key in _VALUE_ONLY else f"if {value}" if key == "trigger"
                    else f"{key.replace('_', ' ')} {value}")
    return f"{label}: {', '.join(said)}" if said else label


def section_points(text: str) -> list[str]:
    """A section's body (heading line dropped) as plain sentences: one per bullet or paragraph,
    markdown removed."""
    body = text.split("\n", 1)[1] if "\n" in text else ""
    points = []
    for line in re.split(r"\n\s*(?=[*\-]\s|\d+\.\s)|\n\s*\n", body):
        line = re.sub(r"^\s*(?:[*\-]|\d+\.)\s+", "", line)
        line = re.sub(r"[*_`#]+", "", " ".join(line.split())).strip()
        if not line or line.lower().startswith("disclaimer"):
            continue
        line = line if line[-1] in ".!?:" else line + "."
        if points and points[-1].endswith(":"):  # a sub-heading leads into its first item
            points[-1] += " " + line
        else:
            points.append(line)
    return points


def wants_more(q: str) -> bool:
    """'the rest' / 'the full protocol' / 'long-term management': a follow-up for more of it."""
    return bool(_ALL.search(q) or _REST.search(q)) or any(w in _plain(q) for w in _SECTION_WORDS)


def _plain(text: str) -> str:
    """Lower case, hyphens as spaces: "Long-Term" and "long term" compare equal."""
    return text.lower().replace("-", " ")


def _section_name(heading: str) -> str:
    """'3. PVCs Protocol › Acute & Emergency Care' -> 'Acute and Emergency Care'."""
    return heading.split(" › ")[-1].replace("&", "and")


def _heading(source: str) -> str:
    return source.split("#", 1)[1] if "#" in source else source


def _cap(points: list[str], spoken: bool, q: str) -> tuple[list[str], int]:
    if not spoken or _ALL.search(q):
        return points, 0
    # The capped answer read the first SPOKEN_POINTS, in this same order. "More about the long-term
    # management" asks for that section from its start instead.
    if _REST.search(q) and not any(w in _plain(q) for w in _SECTION_WORDS):
        return points[SPOKEN_POINTS:] or points, 0
    return points[:SPOKEN_POINTS], max(0, len(points) - SPOKEN_POINTS)


def answer_from_steps(row: dict, etype: str, *, spoken: bool, question: str) -> ProtocolAnswer:
    steps = [s if s.endswith(".") else s + "." for s in map(speak_step, row.get("steps") or [])]
    shown, rest = _cap(steps, spoken, question)
    first = len(steps) - len(shown) + 1 if shown[:1] != steps[:1] else 1  # "go on": 4., 5.
    text = f"{row.get('protocol')} protocol: " + " ".join(
        f"{i}. {s}" for i, s in enumerate(shown, first))
    if rest:
        text += f" {rest} more step{'s' if rest != 1 else ''}; ask for the full " \
                f"{spoken_event(etype)} protocol."
    return ProtocolAnswer(text, [f"protocol:{row.get('protocol')}"])


def answer_from_sections(hits, etype: str, *, spoken: bool, question: str) -> ProtocolAnswer | None:
    """The protocol document that covers `etype` best, its matching sections in relevance order."""
    found = [h for h in hits if not str(h.source).startswith("report:")
             and event_type_from_text(_heading(h.source)) == etype and section_points(h.text)]
    wanted = [w for w in _SECTION_WORDS if w in _plain(question)]
    if wanted:
        found = [h for h in found
                 if any(w in _plain(_heading(h.source)) for w in wanted)] or found
    if not found:
        return None
    by_doc: dict[str, list] = {}
    for h in found:  # hits arrive best first; keep that order within a document
        if all(h.source != o.source for o in by_doc.get(h.doc_id, [])):
            by_doc.setdefault(h.doc_id, []).append(h)
    doc = max(by_doc, key=lambda d: (len(by_doc[d]), max(h.score for h in by_doc[d])))
    # Document order (headings are numbered/titled), not relevance: "the rest" must follow on.
    sections = sorted(by_doc[doc], key=lambda h: _heading(h.source))
    points = [(_heading(h.source), p) for h in sections for p in section_points(h.text)]
    shown, rest = _cap(points, spoken, question)
    if spoken:  # the section headings, not the file name ("vt vf"), say where it comes from
        text = f"{spoken_event(etype)} protocol. " + " ".join(
            f"{_section_name(head)}: "
            + " ".join(p for h2, p in shown if h2 == head)
            for head in dict.fromkeys(h for h, _ in shown))
        text = text[0].upper() + text[1:]
    else:
        text = f"From {doc}:\n" + "\n".join(
            f"\n{head}\n" + "\n".join(f"- {p}" for h2, p in shown if h2 == head)
            for head in dict.fromkeys(h for h, _ in shown))
    if rest:
        later = [h for h in dict.fromkeys(h for h, _ in points[len(shown):])]
        text += (f" {rest} more point{'s' if rest != 1 else ''}, on "
                 f"{', '.join(_section_name(h) for h in later)}; ask for the full "
                 f"{spoken_event(etype)} protocol, or for one of those sections.")
    others = sorted(set(by_doc) - {doc})
    if others and not spoken:
        text += "\n\nAlso covered in: " + ", ".join(others) + "."
    return ProtocolAnswer(text, [h.source for h in sections])
