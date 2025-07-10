from datetime import datetime
from typing import List, Dict, Any

def parse_keypress_events(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "id": e.get("id"),
            "key_code": e["key_code"],
            "key_label": e["key_label"],
            "event_type": e["event_type"],
            "duration_ms": e["duration_ms"],
            "timestamp": datetime.fromisoformat(e["timestamp"]),
            "digram_key1": e.get("digram_key1"),
            "digram_key2": e.get("digram_key2"),
            "context_screen": e["context_screen"],
            "field_name": e.get("field_name"),
        }
        for e in events
    ]

def parse_swipe_events(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "start_x": e["start_x"],
            "start_y": e["start_y"],
            "end_x": e["end_x"],
            "end_y": e["end_y"],
            "distance": e["distance"],
            "duration_ms": e["duration_ms"],
            "timestamp": datetime.fromisoformat(e["timestamp"]),
            "direction": e["direction"],
            "context_screen": e["context_screen"],
        }
        for e in events
    ]

def parse_scroll_events(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "start_offset": e["startOffset"],
            "end_offset": e["endOffset"],
            "distance": e["distance"],
            "duration_ms": e["durationMs"],
            "timestamp": datetime.fromisoformat(e["timestamp"]),
            "context_screen": e["contextScreen"],
            "direction": e["direction"],
        }
        for e in events
    ]

def parse_tap_events(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "x": e["x"],
            "y": e["y"],
            "duration_ms": e["duration_ms"],
            "timestamp": datetime.fromisoformat(e["timestamp"]),
            "context_screen": e["context_screen"],
        }
        for e in events
    ]

def parse_sensor_events(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "type": e["type"],
            "x": e["x"],
            "y": e["y"],
            "z": e["z"],
            "timestamp": datetime.fromisoformat(e["timestamp"]),
            "context_screen": e["context_screen"],
        }
        for e in events
    ]

def parse_full_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    events = payload.get("events", {})
    return {
        "user_id": payload.get("id"),
        "keypress_events": parse_keypress_events(events.get("keypress_events", [])),
        "swipe_events": parse_swipe_events(events.get("swipe_events", [])),
        "tap_events": parse_tap_events(events.get("tap_events", [])),
        "sensor_events": parse_sensor_events(events.get("sensor_events", [])),
        "scroll_events": parse_scroll_events(events.get("scroll_events", [])),
    }
