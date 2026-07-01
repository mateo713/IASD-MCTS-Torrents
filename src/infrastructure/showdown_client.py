from __future__ import annotations

import json
from dataclasses import dataclass
import logging
import asyncio
import time

import requests
import websockets


class LoginError(RuntimeError):
    pass


@dataclass(slots=True)
class ShowdownCredentials:
    username: str
    password: str | None = None


class ShowdownClient:
    """Async Pokemon Showdown websocket client for local or remote servers."""

    def __init__(self, websocket_uri: str, credentials: ShowdownCredentials) -> None:
        self.websocket_uri = websocket_uri
        self.credentials = credentials
        self.websocket: websockets.WebSocketClientProtocol | None = None
        self.last_challenge_time = 0.0

    async def connect(self) -> None:
        logging.debug("Connecting websocket to %s", self.websocket_uri)
        self.websocket = await websockets.connect(self.websocket_uri)
        logging.debug("Connected websocket to %s", self.websocket_uri)

    async def close(self) -> None:
        if self.websocket is not None:
            await self.websocket.close()

    async def send(self, room: str, commands: list[str]) -> None:
        if self.websocket is None:
            raise RuntimeError("Client is not connected")
        payload = room + "|" + "|".join(commands)
        await self.websocket.send(payload)

    async def recv(self) -> str:
        if self.websocket is None:
            raise RuntimeError("Client is not connected")
        return await self.websocket.recv()

    async def login(self) -> str:
        client_id, challstr = await self._get_challstr()
        logging.debug("Login: retrieved challstr for %s", self.credentials.username)
        assertion = await self._request_assertion(client_id, challstr)
        logging.debug("Login: obtained assertion for %s", self.credentials.username)
        await self.send("", [f"/trn {self.credentials.username},0,{assertion}"])
        while True:
            message = await self.recv()
            parts = message.split("|")
            if len(parts) >= 3 and parts[1] == "updateuser":
                return parts[2].strip().lstrip("~")
            if len(parts) >= 3 and parts[1] == "nametaken":
                raise LoginError(parts[3] if len(parts) > 3 else "Username was rejected")

    async def search_ladder(self, battle_format: str) -> None:
        await self.send("", [f"/search {battle_format}"])

    async def challenge(self, target_username: str, battle_format: str) -> None:
        self.last_challenge_time = time.time()
        await self.send("", [f"/challenge {target_username},{battle_format}"])

    async def accept_challenge(self, challenger_username: str) -> None:
        await self.send("", [f"/accept {challenger_username}"])

    async def choose(self, room_id: str, choice_command: str, rqid: int | None = None) -> None:
        command = choice_command if rqid is None else f"{choice_command}|{rqid}"
        await self.send(room_id, [command])

    async def update_team(self, packed_team: str) -> None:
        await self.send("", [f"/utm {packed_team}"])

    async def _get_challstr(self) -> tuple[str, str]:
        while True:
            message = await self.recv()
            parts = message.split("|")
            if len(parts) >= 4 and parts[1] == "challstr":
                return parts[2], parts[3]

    async def _request_assertion(self, client_id: str, challstr: str) -> str:
        login_payload = {
            "act": "getassertion",
            "userid": self.credentials.username,
            "challstr": f"{client_id}|{challstr}",
        }

        def do_post_guest() -> str:
            response = requests.post(
                "https://play.pokemonshowdown.com/action.php?",
                data=login_payload,
                timeout=20,
            )
            if response.status_code != 200 or not response.text:
                raise LoginError("Could not retrieve guest assertion")
            return response.text

        def do_post_auth() -> str:
            response = requests.post(
                "https://play.pokemonshowdown.com/api/login",
                data={
                    "name": self.credentials.username,
                    "pass": self.credentials.password,
                    "challstr": f"{client_id}|{challstr}",
                },
                timeout=20,
            )
            if response.status_code != 200:
                raise LoginError("Could not authenticate against Showdown API")
            body = json.loads(response.text[1:])
            if "assertion" not in body:
                raise LoginError(f"Unexpected login response: {body}")
            return body["assertion"]

        # Run the blocking HTTP request in a thread to avoid blocking the event loop.
        if self.credentials.password:
            return await asyncio.to_thread(do_post_auth)
        return await asyncio.to_thread(do_post_guest)
