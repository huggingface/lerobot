import asyncio
import logging
import traceback
import einops

import numpy as np
import torch
import websockets.asyncio.server
import websockets.frames

import lerobot.common.utils.msgpack_utils as msgpack_utils
from lerobot.common.policies.pretrained import PreTrainedPolicy

class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Currently only implements the `load` and `infer` methods.
    """

    def __init__(
        self,
        policy: PreTrainedPolicy,
        host: str = "0.0.0.0",
        port: int = 8000,
        metadata: dict | None = None,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}        

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        async with websockets.asyncio.server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
        ) as server:
            await server.serve_forever()
            
    async def preprocess_observation(self, observations: dict) -> dict[str, torch.Tensor]:
        images = observations['images']
        return_observations = {}
        
        for imgkey, img in images.items():
            img = torch.from_numpy(img)
            img = einops.rearrange(img, "h w c -> c h w").contiguous()
            c, h, w = img.shape
            if h == 360:
                img = torch.nn.functional.pad(img,pad=(0, 0, 60, 60), mode='constant', value=0)
            c, h, w = img.shape

            assert c < h and c < w, f"expect channel last images, but instead got {img.shape=}"
            assert img.dtype == torch.uint8, f"expect torch.uint8, but instead {img.dtype=}"
            img = img.unsqueeze(0)
            img = img.type(torch.float32)
            img /= 255
            
            imgkey = f"observation.images.{imgkey}"
            print('add a key: ', imgkey, img.shape)
            return_observations[imgkey] = img
            
        return_observations["observation.state"] = torch.from_numpy(observations["state"]).float()
        return_observations["observation.state"] = return_observations["observation.state"].unsqueeze(0)
        
        return return_observations
            

    async def _handler(self, websocket: websockets.asyncio.server.ServerConnection):
        logging.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_utils.Packer()

        await websocket.send(packer.pack(self._metadata))

        while True:
            try:
                # example
                # obs = {
                #     "images": {
                #         "cam_high": numpy.NDArray,
                #         "cam_right_wrist": numpy.NDArray,
                #     },
                #     "state": numpy.NDarray,
                #     "prompt": "xxx text"
                # }
                obs = msgpack_utils.unpackb(await websocket.recv())
                
                obs = await self.preprocess_observation(obs)
                for key in obs:
                    if isinstance(obs[key], torch.Tensor):
                        obs[key] = obs[key].to("cuda", non_blocking=True)
                        
                with torch.inference_mode():
                    action = self._policy.select_action_chunk(obs)
                    action = action.squeeze(0)
                    print("inference once with action:", action.shape, action)
                    res = {"actions": action.cpu().numpy()}
                await websocket.send(packer.pack(res))
            except websockets.ConnectionClosed:
                logging.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise
