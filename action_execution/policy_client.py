"""Minimal, self-contained policy client for the WatchAct LIBERO executor.

The executor talks to a *policy server* over a WebSocket. The server owns the
model; the executor only sends observations and receives action chunks. This
keeps WatchAct decoupled from any particular model framework -- any server that
speaks the protocol below works (openpi's ``WebsocketPolicyServer`` does).

Wire protocol (msgpack with a small numpy extension, identical to openpi's):
  1. On connect the server immediately sends one msgpack message: its metadata
     dict (may be empty).
  2. Per step: the client sends a msgpack-encoded observation dict and receives
     a msgpack-encoded response dict.

Observation dict sent by the executor:
    {
      "observation/image":       uint8 HxWx3   (base RGB, resized+padded)
      "observation/wrist_image": uint8 HxWx3   (wrist RGB, resized+padded)
      "observation/state":       float (8,)    (eef pos[3] + axis-angle[3] + gripper[2])
      "prompt":                  str           (language instruction)
    }
Response dict expected from the server:
    {"actions": float array of shape (T, 7)}   # T >= replan_steps

This module vendors the numpy<->msgpack hooks so it has no dependency on
``msgpack_numpy`` or ``openpi_client`` -- only ``msgpack`` and ``websockets``
(imported lazily, so importing this module never requires a live server).
"""

from __future__ import annotations

import functools
import logging
import time
from typing import Any, Dict

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# numpy <-> msgpack hooks (vendored from openpi_client.msgpack_numpy)
# ---------------------------------------------------------------------------
def _pack_array(obj: Any) -> Any:
    if isinstance(obj, (np.ndarray, np.generic)) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")
    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }
    if isinstance(obj, np.generic):
        return {b"__npgeneric__": True, b"data": obj.item(), b"dtype": obj.dtype.str}
    return obj


def _unpack_array(obj: Any) -> Any:
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


class WebsocketPolicy:
    """Connects to a WebSocket policy server and forwards observations to it.

    Args:
        host: server host, or a full ``ws://`` / ``wss://`` URI.
        port: server port (ignored if ``host`` is a full URI).
        api_key: optional bearer key sent as an ``Authorization`` header.
    """

    def __init__(self, host: str = "0.0.0.0", port: int | None = None, api_key: str | None = None):
        # Lazy imports so this module is importable without the client deps.
        import msgpack  # noqa: PLC0415
        import websockets.sync.client  # noqa: PLC0415

        self._msgpack = msgpack
        self._ws_connect = websockets.sync.client.connect
        self._packer = functools.partial(msgpack.Packer, default=_pack_array)()

        if host.startswith("ws"):
            self._uri = host
        else:
            self._uri = f"ws://{host}"
        if port is not None:
            self._uri += f":{port}"
        self._api_key = api_key
        self._ws, self.server_metadata = self._wait_for_server()

    def _wait_for_server(self):
        logger.info("Waiting for policy server at %s ...", self._uri)
        while True:
            try:
                headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
                conn = self._ws_connect(
                    self._uri, compression=None, max_size=None, additional_headers=headers
                )
                metadata = self._msgpack.unpackb(conn.recv(), object_hook=_unpack_array)
                return conn, metadata
            except ConnectionRefusedError:
                logger.info("Still waiting for policy server...")
                time.sleep(5)

    def infer(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Send one observation, return the server's response dict (has 'actions')."""
        self._ws.send(self._packer.pack(observation))
        response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"Error in policy server:\n{response}")
        return self._msgpack.unpackb(response, object_hook=_unpack_array)

    def reset(self) -> None:
        pass
