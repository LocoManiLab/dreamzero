#!/usr/bin/env python3
"""DreamZero policy server speaking Isaac-GR00T's ZMQ protocol (tcp://0.0.0.0:5550 / get_action).

DreamZero ships a *websocket* server (`socket_test_optimized_AR.py`) whose observation
schema is RoboArena/DROID-shaped (`observation/exterior_image_0_left`, 7-DOF joints, ...).
soma's real-robot loop and the tele_humanoid_benchmark sim client both speak the gr00t
REQ/REP wire format with a G1 SONIC observation instead, so this adapter wraps
`GrootSimPolicy` directly -- the same class the websocket server and
`scripts/open_loop_yam.py` use -- and serves the 78-D SONIC action.

Mirrors PRTS's `deployment/run_prts_server.py` and starVLA's
`deployment/model_server/run_starvla_server.py`; keep the three in sync if the wire
format changes.

Wire protocol (matches gr00t.policy.server_client.PolicyServer):
    Transport : ZMQ REQ/REP
    Encoding  : msgpack-numpy, request = {"endpoint": str, "data": {...}}
    Endpoints : ping, kill, reset, get_action, get_modality_config

Action layout (78-D, identical to the other G1 SONIC models):
    action.motion_token(64) + action.left_hand_joints(7) + action.right_hand_joints(7)
DreamZero already returns exactly these three keys (they are the `modality_keys` of
`modality_config_unitree_g1_sonic`), so unlike PRTS there is no slicing to do -- but the
shapes still have to be normalised to (T, D) float32.

FRAME HISTORY -- the one thing that is NOT like the other soma servers. DreamZero is a
world model with a blockwise-causal KV cache over video, so `get_action` is stateful:
the action head resets `current_start_frame` when it sees a single-frame clip and
otherwise extends the cache. This server therefore keeps a per-episode frame buffer and
feeds 1 frame on the first call of an episode and `--frames-per-chunk` (default 4 =
2 x num_frame_per_block) afterwards -- exactly what `socket_test_optimized_AR.py` does
with its `FRAMES_PER_CHUNK`. `reset` clears the buffer, so the sim client's per-episode
reset is what keeps episodes independent. Without that, episode N+1 would be conditioned
on episode N's video.

Usage:
    python deployment/run_dreamzero_server.py \
        --model-path checkpoints/dreamzero_g1_sonic/checkpoint-2000 \
        --port 5550 --initial-prompt "walk over to the toilet and stand next to it"
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any, Optional

import numpy as np

MOTION_TOKEN_DIM = 64
HAND_DIM = 7
ACTION_DIM = MOTION_TOKEN_DIM + 2 * HAND_DIM      # 78

# The action heads DreamZero returns, in the order the 78-D vector is packed. Must match
# `modality_config_unitree_g1_sonic.action.modality_keys` in
# groot/vla/configs/data/dreamzero/base_48_wan_fine_aug_relative.yaml.
ACTION_KEYS = (("action.motion_token", MOTION_TOKEN_DIM),
               ("action.left_hand_joints", HAND_DIM),
               ("action.right_hand_joints", HAND_DIM))

# The 43-D G1 state, group by group, in the order observation.state packs them (soma's
# deploy_living_room_vla.py _STATE_SLICES and the dataset's meta/modality.json agree).
# These are also the `state.*` modality keys the model was trained with, so each group is
# handed over separately rather than concatenated. Do NOT sort: alphabetical order
# scrambles the vector. `projected_gravity` is an EXTRA field the sim client sends that is
# not part of the 43 and must not be forwarded.
STATE_ORDER = ("left_leg", "right_leg", "waist", "left_arm",
               "left_hand", "right_arm", "right_hand")
STATE_DIMS = {"left_leg": 6, "right_leg": 6, "waist": 3, "left_arm": 7,
              "left_hand": 7, "right_arm": 7, "right_hand": 7}

LANGUAGE_KEY = "annotation.human.task_description"
VIDEO_KEY = "video.ego_view"


class DreamZeroPolicy:
    """Adapts GrootSimPolicy.lazy_joint_forward_causal() to the gr00t get_action() contract."""

    def __init__(self, policy, initial_prompt: Optional[str] = None,
                 video_frames_per_chunk: int = 8, video_stride: int = 3,
                 upsample_factor: int = 1):
        self.policy = policy
        self._current_prompt = initial_prompt
        # HOW THE CLIP IS BUILT -- this has to match TRAINING, not intuition.
        # `lerobot_sharded.py` samples a chunk's video as
        #     per_step_offsets = [i * 3 * chunk_stride for i in range(8)]
        # i.e. EIGHT frames at stride 3*chunk_stride, spanning one action horizon
        # (24 * chunk_stride raw frames). Serving four CONSECUTIVE frames instead --
        # which is what upstream's DROID AR server does, and what this file did until
        # 2026-08-26 -- hands the model a 0.08 s window at 50 Hz where training used
        # 0.48 s: half the frames over a sixth of the time, a clip shape it never saw.
        # Nothing errors; the policy just degrades. SIMPLE's own DreamZero G1 client
        # (src/simple/baselines/dreamzero.py) derives it the same way:
        #     video_frames_per_chunk = 8, video_stride = action_horizon // 8
        self.n_frames = max(1, int(video_frames_per_chunk))
        self.video_stride = max(1, int(video_stride))
        self.upsample_factor = max(1, int(upsample_factor))
        self._frames: list[np.ndarray] = []
        self._first_call = True
        self.expect_wh = self._expected_resolution(policy)
        if self.expect_wh:
            print(f"[dreamzero] incoming frames will be resized to "
                  f"{self.expect_wh[0]}x{self.expect_wh[1]} (WxH)", flush=True)

    @staticmethod
    def _expected_resolution(policy) -> Optional[tuple[int, int]]:
        """(W, H) the checkpoint's eval transform will ACCEPT, or None.

        `VideoToTensor.check_input` hard-asserts the incoming resolution against
        `original_resolutions`, which the transform took from the checkpoint's
        DatasetMetadata -- and for a Wan 5B head `GrootSimPolicy.__init__` REWRITES that
        metadata to the action head's `target_video_height/width` (160x320) before building
        the transform. So the eval path expects frames already at the model's resolution,
        while a G1 ego camera sends its native 240x320:

            ValueError: Error applying transform 0 to data: ... Video video.ego_view has
            invalid resolution (320, 240), expected (320, 160)

        Training never sees this because VideoCrop/VideoResize sit later in the same chain
        and do the resizing there. Reading the number back off the instantiated transform
        (rather than re-deriving it from the config) means the server resizes to exactly
        what the validator is about to compare against, for both backbones -- and it is a
        no-op for a 14B checkpoint, whose metadata keeps the native resolution.
        """
        tf = getattr(policy, "eval_transform", None)
        for t in getattr(tf, "transforms", []) or []:
            res = getattr(t, "original_resolutions", None)
            if res:
                for key in (VIDEO_KEY, VIDEO_KEY.split(".", 1)[1]):
                    if key in res:
                        w, h = res[key]
                        return int(w), int(h)
        print("[dreamzero] could not read the expected video resolution off the eval "
              "transform; frames will be sent through unresized", flush=True)
        return None

    def _resize(self, frame: np.ndarray) -> np.ndarray:
        if not self.expect_wh:
            return frame
        w, h = self.expect_wh
        if (frame.shape[1], frame.shape[0]) == (w, h):
            return frame
        import cv2
        # INTER_AREA for downscaling: it is what an area-average resize does and avoids the
        # aliasing a nearest/bilinear downscale would put into the VAE input.
        interp = cv2.INTER_AREA if (frame.shape[0] > h or frame.shape[1] > w) else cv2.INTER_LINEAR
        return cv2.resize(frame, (w, h), interpolation=interp)

    # ---- gr00t endpoints ---------------------------------------------------
    def ping(self) -> dict:
        return {"status": "ok", "message": "dreamzero gr00t-compat server is running"}

    def get_modality_config(self) -> dict:
        return {}

    def reset(self, options: dict | None = None) -> dict:
        """Start a new episode: drop the video history so the causal cache restarts."""
        prompt = (options or {}).get("prompt")
        if prompt is not None:
            self._current_prompt = prompt
        self._frames.clear()
        self._first_call = True
        print(f"[dreamzero] reset: frame buffer cleared, prompt={self._current_prompt!r}",
              flush=True)
        return {"status": "reset", "prompt": self._current_prompt}

    def get_action(self, observation: dict[str, Any],
                   options: dict[str, Any] | None = None
                   ) -> tuple[dict[str, Any], dict[str, Any]]:
        import torch
        from tianshou.data import Batch

        t0 = time.monotonic()

        prompt = self._extract_prompt(observation) or self._current_prompt
        if prompt is None:
            raise RuntimeError(
                "No prompt set. Pass --initial-prompt, call reset(prompt=...), or include "
                "obs['language']['annotation.human.task_description'][0][0].")
        if prompt != self._current_prompt:
            # The action head resets its cache on a language change anyway; clearing the
            # buffer here keeps the two in step instead of feeding a 4-frame clip whose
            # first frames belong to the previous instruction.
            self._frames.clear()
            self._first_call = True
        self._current_prompt = prompt

        frame = self._resize(self._extract_image(observation))   # [H,W,3] uint8
        self._frames.append(frame)
        span = 1 + (self.n_frames - 1) * self.video_stride   # raw frames the clip covers
        if len(self._frames) > span:
            del self._frames[: len(self._frames) - span]

        if self._first_call:
            # One frame on the first call of an episode: the action head resets
            # `current_start_frame` when it sees a single-frame clip, which is what starts
            # the KV cache cleanly (see the module docstring).
            clip = self._frames[-1:]
            self._first_call = False
        else:
            # Sample BACKWARD from the newest frame so the most recent observation is
            # always the last element -- the action head takes the tail latent frames, so a
            # clip that ends in the past would condition on stale video. Indices are
            # clamped at 0, which repeats the oldest frame early in an episode when the
            # buffer has not filled yet.
            idx = [max(0, len(self._frames) - 1 - i * self.video_stride)
                   for i in range(self.n_frames)]
            clip = [self._frames[i] for i in reversed(idx)]

        obs: dict[str, Any] = {
            VIDEO_KEY: np.stack(clip, axis=0).astype(np.uint8),   # (T, H, W, 3)
            LANGUAGE_KEY: prompt,
        }
        state_obs = self._extract_state(observation)             # state.<group> -> (1, D)
        obs.update(state_obs)
        # DZ_DUMP_OBS=<path>: append what the CLIENT actually sends, so the closed-loop
        # observation distribution can be compared against the training data's. The action
        # path is provable by replaying recorded actions; the observation path is not, and
        # a convention mismatch there looks exactly like an undertrained policy.
        if os.environ.get("DZ_DUMP_OBS"):
            import pickle
            with open(os.environ["DZ_DUMP_OBS"], "ab") as fh:
                pickle.dump({"state": {k: v.copy() for k, v in state_obs.items()},
                             "img_mean": float(frame.mean()),
                             "img_shape": tuple(frame.shape)}, fh)

        t_inf = time.monotonic()
        with torch.no_grad():
            result, _video_pred = self.policy.lazy_joint_forward_causal(Batch(obs=obs))
        infer_ms = (time.monotonic() - t_inf) * 1000.0

        action_dict = {}
        horizon = 0
        for key, dim in ACTION_KEYS:
            chunk = self._as_chunk(self._get_act(result, key), key, dim)
            chunk = self._upsample(chunk)
            action_dict[key] = np.ascontiguousarray(chunk, dtype=np.float32)
            horizon = max(horizon, chunk.shape[0])

        print(f"[dreamzero] get_action: clip_T={len(clip)}@stride{self.video_stride} "
              f"-> H_out={horizon} "
              f"(x{self.upsample_factor}); infer={infer_ms:.0f} ms; prompt={prompt!r}",
              flush=True)
        return action_dict, {"inference_time_s": time.monotonic() - t0,
                             "chunk_horizon": int(horizon)}

    # ---- internals ---------------------------------------------------------
    @staticmethod
    def _get_act(result, key: str):
        """`result.act` is a plain dict for a dict obs, but tianshou may hand back a Batch."""
        act = getattr(result, "act", result)
        if isinstance(act, dict):
            if key in act:
                return act[key]
        elif hasattr(act, key):
            return getattr(act, key)
        available = list(act.keys()) if hasattr(act, "keys") else dir(act)
        raise RuntimeError(
            f"checkpoint did not return {key!r}; got {available}. Was it trained with "
            f"data=dreamzero/unitree_g1_sonic_relative (action.motion_token + the two hand "
            f"heads)?")

    @staticmethod
    def _as_chunk(value, key: str, dim: int) -> np.ndarray:
        import torch
        if isinstance(value, torch.Tensor):
            # .float() first: numpy cannot represent bfloat16 and the error surfaces far
            # from here (PRTS hit exactly this, mislabelled as a msgpack bug).
            value = value.detach().float().cpu().numpy()
        arr = np.asarray(value, dtype=np.float32)
        while arr.ndim > 2:                 # drop a leading batch axis
            arr = arr[0]
        if arr.ndim == 1:
            arr = arr[None, :] if arr.shape[0] == dim else arr[:, None]
        if arr.shape[-1] != dim:
            raise RuntimeError(f"{key} has width {arr.shape[-1]}, expected {dim}")
        return arr

    @staticmethod
    def _extract_prompt(obs: dict) -> Optional[str]:
        lang = obs.get("language") or {}
        v = lang.get(LANGUAGE_KEY)
        while isinstance(v, (list, tuple, np.ndarray)) and len(v):
            v = v[0]
        if isinstance(v, bytes):
            v = v.decode()
        return v if isinstance(v, str) and v else None

    @staticmethod
    def _extract_image(obs: dict) -> np.ndarray:
        """Newest ego-view frame as HWC uint8.

        The tele sim client sends `img[np.newaxis, np.newaxis]`, i.e. (1, 1, H, W, 3) --
        FIVE dims, under the key "ego_view" (not "video.ego_view"). Real-robot callers may
        send (T, H, W, 3) or bare (H, W, 3). Peel every leading singleton/time axis instead
        of assuming a fixed rank.
        """
        video = obs.get("video") or {}
        for key in ("video.ego_view", "ego_view", "video.webcam", "webcam"):
            arr = video.get(key) if key in video else obs.get(key)
            if arr is None:
                continue
            arr = np.asarray(arr)
            while arr.ndim > 3:            # drop batch/time axes, keeping the newest frame
                arr = arr[-1]
            if arr.ndim != 3 or arr.shape[-1] not in (1, 3, 4):
                raise RuntimeError(f"ego-view image has unexpected shape {np.shape(arr)}; "
                                   f"expected (..., H, W, 3)")
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            return arr
        raise RuntimeError(f"No ego-view image in observation; keys={list(obs)} "
                           f"video={list(video)}")

    @classmethod
    def _extract_state(cls, obs: dict) -> dict[str, np.ndarray]:
        """The seven `state.<group>` arrays the model was trained on, each (1, D) float64.

        `float64` because that is what the training dataloader produces for state columns
        and what `StateActionToTensor` is fed; the transform casts anyway, but matching it
        keeps the eval path byte-identical to `scripts/open_loop_yam.py`.
        """
        state = obs.get("state") or {}
        out: dict[str, np.ndarray] = {}
        for group in STATE_ORDER:
            v = state.get(group)
            if v is None:
                raise RuntimeError(
                    f"observation is missing state group {group!r}; got {sorted(state)}. "
                    f"A G1 SONIC client sends all of {list(STATE_ORDER)}.")
            v = np.asarray(v, dtype=np.float64)
            while v.ndim > 1:              # client sends (1, 1, D); keep the newest row
                v = v[-1]
            v = v.ravel()
            if v.shape[0] != STATE_DIMS[group]:
                raise RuntimeError(f"state.{group} has {v.shape[0]} dims, "
                                   f"expected {STATE_DIMS[group]}")
            out[f"state.{group}"] = v.reshape(1, -1)
        return out

    def _upsample(self, chunk: np.ndarray) -> np.ndarray:
        """Linearly interpolate the action chunk when the control loop wants more steps."""
        if self.upsample_factor == 1 or chunk.shape[0] < 2:
            return chunk
        t_in = np.arange(chunk.shape[0], dtype=np.float32)
        t_out = np.linspace(0, chunk.shape[0] - 1,
                            chunk.shape[0] * self.upsample_factor, dtype=np.float32)
        return np.stack([np.interp(t_out, t_in, chunk[:, d])
                         for d in range(chunk.shape[1])], axis=1).astype(np.float32)


def serve(policy: DreamZeroPolicy, host: str, port: int) -> None:
    import msgpack_numpy as mnp
    import zmq

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REP)
    sock.bind(f"tcp://{host}:{port}")
    print(f"[dreamzero] PolicyServer ready on tcp://{host}:{port}", flush=True)

    while True:
        try:
            req = mnp.unpackb(sock.recv(), raw=False)
            endpoint = req.get("endpoint", "get_action")
            data = req.get("data") or {}

            if endpoint == "ping":
                resp = policy.ping()
            elif endpoint == "kill":
                sock.send(mnp.packb({"status": "ok"}))
                break
            elif endpoint == "reset":
                resp = policy.reset(data)
            elif endpoint == "get_modality_config":
                resp = policy.get_modality_config()
            elif endpoint == "get_action":
                action, info = policy.get_action(data.get("observation", data),
                                                 data.get("options"))
                # The sim client does `action, _info = resp[0], resp[1]` -- it expects a
                # LIST, not a dict. Returning the action dict directly makes the client die
                # with a misleading `KeyError: 0` while the server logs look fine.
                resp = [action, info]
            else:
                resp = {"error": f"unknown endpoint {endpoint!r}"}
            sock.send(mnp.packb(resp))
        except zmq.error.ContextTerminated:
            break
        except Exception as e:                       # keep serving after a bad request
            import traceback
            traceback.print_exc()
            try:
                sock.send(mnp.packb({"error": str(e)}))
            except Exception:
                pass
    print("[dreamzero] server stopped", flush=True)


def load_policy(args) -> DreamZeroPolicy:
    # torch._dynamo off before importing the model: the causal WAN blocks are compiled
    # lazily and a recompile storm on varying clip lengths costs minutes per call.
    import torch._dynamo
    torch._dynamo.config.disable = True

    import torch
    import torch.distributed as dist

    if args.dreamzero_repo:
        sys.path.insert(0, args.dreamzero_repo)

    # GrootSimPolicy.__init__ calls dist.get_rank() unconditionally, so a plain
    # single-process run still needs a process group (same trick as scripts/open_loop_yam.py).
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", str(args.master_port))
        dist.init_process_group(backend="gloo", world_size=1, rank=0)

    from groot.vla.data.schema import EmbodimentTag
    from groot.vla.model.n1_5.sim_policy import GrootSimPolicy

    tag = EmbodimentTag(args.embodiment_tag)
    print(f"[dreamzero] loading {args.model_path} (tag={tag.value}, device={args.device})",
          flush=True)
    # The base the LoRA was trained on top of. MUST be supplied whenever the checkpoint
    # was saved with save_lora_only=true: `load_lora` otherwise rebuilds the DiT from
    # `dit_version` (raw Wan2.1) and applies the LoRA delta to THAT, silently discarding
    # the DreamZero pretraining the action head was fitted against. Read from the
    # checkpoint's own conf.yaml so it cannot be forgotten.
    base = args.pretrained_base
    if base is None and not args.no_pretrained_base:
        base = _training_pretrained_base(args.model_path)
        if base:
            print(f"[dreamzero] LoRA base from the checkpoint's conf.yaml: {base}", flush=True)
    if base and not os.path.isdir(base):
        raise SystemExit(
            f"[dreamzero] pretrained base {base} does not exist. Serving a save_lora_only "
            f"checkpoint without it puts the LoRA on raw Wan2.1 weights and the policy "
            f"silently degrades to its action prior. Pass --pretrained-base, or "
            f"--no-pretrained-base if this checkpoint really was trained from raw Wan.")

    policy = GrootSimPolicy(
        embodiment_tag=tag,
        model_path=args.model_path,
        device=args.device,
        tokenizer_path_override=args.tokenizer_path,
        pretrained_base_path=base,
    )
    print(f"[dreamzero] loaded. action heads = {[k for k, _ in ACTION_KEYS]} "
          f"(action_dim={ACTION_DIM})", flush=True)

    stride = args.video_stride
    if stride is None:
        # Derive from the CHECKPOINT, not from a manifest default. Training samples a
        # chunk's video as 8 frames spread over one action horizon, so the correct serving
        # stride is horizon // frames -- and the horizon is a property of the checkpoint
        # (24 for runs before 2026-08-26, 48 after, matching DreamZero-AgiBot). A default
        # baked into the manifest silently mis-serves every checkpoint trained with the
        # other value: no error, just a clip shape the model never saw.
        horizon = _training_action_horizon(args.model_path)
        if horizon is None:
            stride = 3
            print(f"[dreamzero] could not read num_action_per_block from the checkpoint; "
                  f"falling back to --video-stride {stride}", flush=True)
        else:
            stride = max(1, horizon // args.video_frames_per_chunk)
            print(f"[dreamzero] checkpoint trained with action horizon {horizon} "
                  f"-> video_stride {stride} ({args.video_frames_per_chunk} frames "
                  f"spanning {stride * args.video_frames_per_chunk} raw frames)", flush=True)
    return DreamZeroPolicy(policy, args.initial_prompt, args.video_frames_per_chunk,
                           stride, args.upsample_factor)


def _training_pretrained_base(model_path: str) -> Optional[str]:
    """`pretrained_model_path` from the checkpoint's own experiment_cfg/conf.yaml."""
    import os
    for base in (model_path, os.path.dirname(model_path.rstrip("/"))):
        cfg = os.path.join(base, "experiment_cfg", "conf.yaml")
        if not os.path.isfile(cfg):
            continue
        try:
            import yaml
            d = yaml.safe_load(open(cfg))
        except Exception as e:  # noqa: BLE001
            print(f"[dreamzero] cannot parse {cfg}: {e}", flush=True)
            return None
        v = (d or {}).get("pretrained_model_path")
        return v if isinstance(v, str) and v else None
    return None


def _training_action_horizon(model_path: str) -> Optional[int]:
    """`num_action_per_block` from the checkpoint's own experiment_cfg/conf.yaml."""
    import os
    for base in (model_path, os.path.dirname(model_path.rstrip("/"))):
        cfg = os.path.join(base, "experiment_cfg", "conf.yaml")
        if not os.path.isfile(cfg):
            continue
        try:
            import yaml
            d = yaml.safe_load(open(cfg))
        except Exception as e:  # noqa: BLE001
            print(f"[dreamzero] cannot parse {cfg}: {e}", flush=True)
            return None
        for path in (("num_action_per_block",),
                     ("action_head_cfg", "config", "diffusion_model_cfg", "num_action_per_block"),
                     ("action_head_cfg", "config", "action_horizon")):
            cur = d
            for k in path:
                cur = cur.get(k) if isinstance(cur, dict) else None
                if cur is None:
                    break
            if isinstance(cur, int):
                return cur
    return None


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="DreamZero policy server (gr00t-compatible, :5550/get_action).")
    p.add_argument("--model-path", required=True,
                   help="DreamZero checkpoint dir (contains config.json, experiment_cfg/)")
    p.add_argument("--dreamzero-repo", default=None,
                   help="prepend to sys.path if dreamzero isn't pip-installed")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=5550)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--embodiment-tag", default="unitree_g1_sonic",
                   help="must match meta/embodiment.json of the training dataset")
    p.add_argument("--initial-prompt", default=None)
    p.add_argument("--tokenizer-path", default=None,
                   help="override the umt5-xxl path baked into the checkpoint's "
                        "experiment_cfg/conf.yaml (training-machine absolute path)")
    p.add_argument("--video-frames-per-chunk", type=int, default=8,
                   help="frames per clip after the first call of an episode. MUST match "
                        "training's per-chunk video sampling (8 in lerobot_sharded.py).")
    p.add_argument("--video-stride", type=int, default=None,
                   help="raw frames between sampled clip frames. Default: read the training "
                        "action horizon out of the checkpoint and use horizon//frames, which "
                        "is what training sampled. Pass a value only to override that.")
    p.add_argument("--pretrained-base", default=None,
                   help="the checkpoint the LoRA was trained on top of (e.g. "
                        "DreamZero-AgiBot). Default: read `pretrained_model_path` out of "
                        "the checkpoint's own conf.yaml. REQUIRED for save_lora_only "
                        "checkpoints -- without it the LoRA lands on raw Wan2.1 weights.")
    p.add_argument("--no-pretrained-base", action="store_true",
                   help="serve without a base. Only for a checkpoint genuinely trained "
                        "from the raw Wan backbone.")
    p.add_argument("--upsample-factor", type=int, default=1)
    p.add_argument("--master-port", type=int, default=29500,
                   help="torch.distributed rendezvous port for the world_size=1 group")
    args = p.parse_args(argv)

    serve(load_policy(args), args.host, args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
