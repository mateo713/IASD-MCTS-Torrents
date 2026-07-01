from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any

from core.identifiers import extract_hp_chunk, extract_species_from_details, hp_fraction_from_condition, normalize_identifier


@dataclass(slots=True)
class ParsedEvent:
    room_id: str
    event_type: str
    payload: dict[str, Any]
    raw_line: str
    subject: str | None = None
    target: str | None = None
    data: dict[str, Any] = field(default_factory=dict)


def parse_showdown_message(message: str) -> list[ParsedEvent]:
    """Parse one websocket payload into normalized room-scoped events."""
    events: list[ParsedEvent] = []
    room_id = ""

    for raw_line in message.splitlines():
        if not raw_line:
            continue

        if raw_line.startswith(">"):
            room_id = raw_line[1:]
            continue

        if not raw_line.startswith("|"):
            continue

        parts = raw_line.split("|")
        if len(parts) < 2:
            continue

        tag = parts[1]

        if tag == "turn" and len(parts) >= 3:
            turn = int(parts[2])
            events.append(ParsedEvent(room_id, "turn", {"turn": turn}, raw_line, data={"turn": turn}))
        elif tag == "request" and len(parts) >= 3:
            try:
                request_json = json.loads(parts[2])
            except json.JSONDecodeError:
                request_json = {}
            events.append(ParsedEvent(room_id, "request", {"request": request_json}, raw_line, data={"request": request_json}))
        elif tag == "win" and len(parts) >= 3:
            events.append(ParsedEvent(room_id, "win", {"winner": parts[2]}, raw_line, target=parts[2], data={"winner": parts[2]}))
        elif tag == "tie":
            events.append(ParsedEvent(room_id, "tie", {}, raw_line))
        elif tag == "error" and len(parts) >= 3:
            events.append(ParsedEvent(room_id, "error", {"message": parts[2]}, raw_line, data={"message": parts[2]}))
        elif tag in {"switch", "drag", "replace"} and len(parts) >= 5:
            ident = parts[2].strip()
            details = parts[3].strip()
            condition = parts[4].strip()
            species = extract_species_from_details(details)
            hp_fraction = hp_fraction_from_condition(condition)
            status = None
            hp_chunk = extract_hp_chunk(condition)
            if condition and " " in condition:
                status = condition.split(" ", 1)[1].strip() or None
            events.append(
                ParsedEvent(
                    room_id,
                    tag,
                    {"ident": ident, "details": details, "condition": condition, "species": species},
                    raw_line,
                    subject=ident,
                    target=species,
                    data={"condition": condition, "hp_fraction": hp_fraction, "hp_chunk": hp_chunk, "status": status},
                )
            )
        elif tag == "move" and len(parts) >= 4:
            ident = parts[2].strip()
            move_name = parts[3].strip()
            target = parts[4].strip() if len(parts) >= 5 else None
            move_id = normalize_identifier(move_name)
            events.append(
                ParsedEvent(
                    room_id,
                    tag,
                    {"ident": ident, "move": move_name, "target": target, "move_id": move_id},
                    raw_line,
                    subject=ident,
                    target=target,
                    data={"move_id": move_id, "move": move_name},
                )
            )
        elif tag in {"-damage", "-heal"} and len(parts) >= 4:
            ident = parts[2].strip()
            condition = parts[3].strip()
            hp_fraction = hp_fraction_from_condition(condition)
            events.append(
                ParsedEvent(
                    room_id,
                    tag,
                    {"ident": ident, "condition": condition, "hp_fraction": hp_fraction},
                    raw_line,
                    subject=ident,
                    data={"condition": condition, "hp_fraction": hp_fraction},
                )
            )
        elif tag in {"-item", "-enditem", "-ability", "-status", "-curestatus", "-boost", "-unboost", "-immune", "-activate", "-start", "-end", "-sidestart", "-sideend", "-weather", "-fieldstart", "-fieldend", "-terrain"}:
            ident = parts[2].strip() if len(parts) >= 3 else None
            value = parts[3].strip() if len(parts) >= 4 else None
            normalized_value = normalize_identifier(value)
            events.append(
                ParsedEvent(
                    room_id,
                    tag,
                    {"parts": parts[2:], "ident": ident, "value": value, "normalized_value": normalized_value},
                    raw_line,
                    subject=ident,
                    target=value,
                    data={"value": value, "normalized_value": normalized_value},
                )
            )
        else:
            events.append(ParsedEvent(room_id, tag, {"parts": parts[2:]}, raw_line))

    return events
