#!/usr/bin/env python3
"""Safe CLI used by the Hermes voiceprint-management skill."""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request

BASE_URL = os.getenv("VOICEPRINT_BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")


def request(method: str, path: str, data: bytes | None = None, content_type: str | None = None):
    headers = {"Accept": "application/json"}
    if content_type:
        headers["Content-Type"] = content_type
    req = urllib.request.Request(BASE_URL + path, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            body = response.read()
            return json.loads(body) if body else None
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"voiceprint backend unavailable: {exc.reason}") from exc


def user_path(speaker_id: str, suffix: str = "") -> str:
    return f"/api/voiceprint/users/{urllib.parse.quote(speaker_id, safe='')}{suffix}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Manage voice identities through the project backend")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="List all voice identities")
    show = sub.add_parser("show", help="Show one voice identity")
    show.add_argument("--speaker-id", required=True)

    rename = sub.add_parser("rename", help="Update a voice identity's name")
    rename.add_argument("--speaker-id", required=True)
    rename.add_argument("--display-name")
    rename.add_argument("--preferred-name")

    add = sub.add_parser("add-current", help="Add the current utterance as another template")
    add.add_argument("--speaker-id", required=True)
    add.add_argument("--audio-ref", required=True)
    add.add_argument("--label", default="voice-command")

    templates = sub.add_parser("templates", help="List a speaker's voiceprint templates")
    templates.add_argument("--speaker-id", required=True)

    delete_template = sub.add_parser("delete-template", help="Delete one voiceprint template")
    delete_template.add_argument("--speaker-id", required=True)
    delete_template.add_argument("--voiceprint-id", required=True)
    delete_template.add_argument("--confirm", action="store_true")

    delete_voiceprints = sub.add_parser("delete-voiceprints", help="Delete all templates but keep the user profile")
    delete_voiceprints.add_argument("--speaker-id", required=True)
    delete_voiceprints.add_argument("--confirm", action="store_true")

    delete_speaker = sub.add_parser("delete-speaker", help="Delete the user profile and every template")
    delete_speaker.add_argument("--speaker-id", required=True)
    delete_speaker.add_argument("--confirm", action="store_true")

    args = parser.parse_args()
    if args.command.startswith("delete-") and not args.confirm:
        parser.error(f"{args.command} requires --confirm after explicit user confirmation")
    if args.command == "list":
        result = request("GET", "/api/voiceprint/users")
    elif args.command == "show":
        result = request("GET", user_path(args.speaker_id))
    elif args.command == "rename":
        changes = {}
        if args.display_name is not None:
            changes["display_name"] = args.display_name
        if args.preferred_name is not None:
            changes["preferred_name"] = args.preferred_name
        if not changes:
            parser.error("rename needs --display-name or --preferred-name")
        result = request(
            "PATCH",
            user_path(args.speaker_id),
            json.dumps(changes, ensure_ascii=False).encode(),
            "application/json",
        )
    elif args.command == "add-current":
        query = urllib.parse.urlencode({
            "audio_ref": args.audio_ref,
            "label": args.label,
        })
        result = request("POST", user_path(args.speaker_id, f"/voiceprints/current?{query}"), b"")
    elif args.command == "templates":
        result = request("GET", user_path(args.speaker_id, "/voiceprints"))
    elif args.command == "delete-template":
        result = request(
            "DELETE",
            user_path(args.speaker_id, f"/voiceprints/{urllib.parse.quote(args.voiceprint_id, safe='')}"),
        )
    elif args.command == "delete-voiceprints":
        result = request("DELETE", user_path(args.speaker_id, "/voiceprints"))
    elif args.command == "delete-speaker":
        result = request("DELETE", user_path(args.speaker_id))
    else:
        parser.error("unsupported command")
        return 2

    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(1) from exc
