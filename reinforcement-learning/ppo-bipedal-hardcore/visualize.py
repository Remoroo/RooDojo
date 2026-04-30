"""
visualize.py — Interactive visualizer for the BipedalWalker PPO agent.

Loads checkpoint.pt (saved by ppo_agent.py's PPOAgent.save()) and runs the
policy deterministically inside a rendered gymnasium environment.

Usage
-----
  python visualize.py                          # defaults (hardcore env, best checkpoint)
  python visualize.py --env BipedalWalker-v3  # easy env
  python visualize.py --checkpoint custom.pt  # different checkpoint
  python visualize.py --episodes 5 --stochastic   # sample rather than use mean
  python visualize.py --no-render             # headless, just print stats
  python visualize.py --width 960 --height 540   # scale the viewer (and recording) output
  python visualize.py --video-out run.mp4        # record H.264 (requires ffmpeg on PATH)
  python visualize.py --no-render --video-out run.mp4   # headless capture
  python visualize.py --video-out hq.mp4 --video-crf 17 --video-preset slow --video-pix-fmt yuv444p
      # stronger encode (BipedalWalker is only 600x400 native; this cuts artifacts, not adds detail)

Controls (when a window is open)
---------------------------------
  Q / Esc  → quit
  R        → reset current episode immediately
  S        → toggle stochastic / deterministic policy
  ↑ / ↓   → increase / decrease playback speed
"""

import argparse
import os
import shutil
import subprocess
import sys
import time

import gymnasium as gym
import numpy as np
import torch

# ---------------------------------------------------------------------------
# numpy compatibility shim
# ---------------------------------------------------------------------------
# Checkpoints pickled under numpy >= 2.0 reference 'numpy._core.*', which
# does not exist in numpy < 2.0.  Redirect those references to 'numpy.core'.
try:
    import numpy._core  # noqa: F401 — already available, nothing to do
except ModuleNotFoundError:
    import numpy.core as _np_core
    import types
    _shim = types.ModuleType("numpy._core")
    _shim.__dict__.update(_np_core.__dict__)
    _shim.multiarray = _np_core.multiarray
    sys.modules.setdefault("numpy._core", _shim)
    sys.modules.setdefault("numpy._core.multiarray", _np_core.multiarray)
    sys.modules.setdefault("numpy._core.numeric", getattr(_np_core, "numeric", _np_core))

# ---------------------------------------------------------------------------
# Import classes directly from ppo_agent so we never diverge from its config
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ppo_agent import PPOAgent, ActorCritic, RunningMeanStd  # noqa: F401 (re-exported for clarity)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DEFAULT_HIDDEN_DIM = 256


def _infer_hidden_dim_from_checkpoint(checkpoint_path: str) -> int | None:
    """Peek at a PPO checkpoint's state_dict and return `hidden_dim`.

    The `ActorCritic` module in both ppo/ppo_agent.py and
    try_now_repos/<env>/ppo_core.py stores the first hidden layer as
    `actor_net.0.weight` with shape `(hidden_dim, state_dim)`. Reading
    that shape gives us the width the checkpoint was trained with —
    which is the ONLY source of truth. The alternatives (config.py,
    CLI flags, env-tables) all drift silently: e.g. LunarLander bakes
    its baseline with HIDDEN_DIM=128 but the historical `ppo/`
    default is 256, so without inference the clip-watcher builds a
    256-wide net and load_state_dict blows up with

        size mismatch for actor_net.0.weight: copying a param with
        shape torch.Size([128, 8]) from checkpoint, the shape in
        current model is torch.Size([256, 8]).

    Returns None if the file can't be read or the key is absent; the
    caller falls back to its explicit arg or `_DEFAULT_HIDDEN_DIM`.
    """
    try:
        state = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
    except Exception:
        return None
    model = state.get("model") if isinstance(state, dict) else None
    if not isinstance(model, dict):
        return None
    w = model.get("actor_net.0.weight")
    if w is None or not hasattr(w, "shape") or len(w.shape) < 1:
        return None
    return int(w.shape[0])


def load_agent(
    checkpoint_path: str,
    env_name: str,
    device: str,
    env_kwargs: dict | None = None,
    hidden_dim: int | None = None,
) -> PPOAgent:
    """Probe the env for dims, build an agent, then load the checkpoint.

    `env_kwargs` is forwarded to `gym.make`; critical for envs that
    expose a different action-space shape under a flag (notably
    `LunarLander-v2` where `continuous=True` swaps the space from
    `Discrete(4)` to `Box(2,)`). Without it the probe here reads a
    shape that doesn't match the checkpoint's actor head and fails
    with `IndexError: tuple index out of range` — or worse, silently
    builds a mis-shaped network.

    `hidden_dim` must match the width used during training. When
    explicitly passed, we honor it (useful for tests / pinning). When
    `None` (the common case — run_clips.sh does not know per-env
    widths), we INFER it from the checkpoint's state_dict by reading
    `actor_net.0.weight.shape[0]`. The checkpoint is the source of
    truth; config.py / CLI flags drift. If inference fails we fall
    back to the historical 256 default so the standalone
    `python ppo/visualize.py --env BipedalWalkerHardcore ...` CLI
    path still works unchanged against older checkpoints.
    """
    probe = gym.make(env_name, **(env_kwargs or {}))
    state_dim = probe.observation_space.shape[0]
    action_dim = probe.action_space.shape[0]
    probe.close()

    if hidden_dim is not None:
        resolved_hidden = int(hidden_dim)
        hidden_source = "explicit"
    else:
        inferred = _infer_hidden_dim_from_checkpoint(checkpoint_path)
        if inferred is not None:
            resolved_hidden = inferred
            hidden_source = "inferred-from-checkpoint"
        else:
            resolved_hidden = _DEFAULT_HIDDEN_DIM
            hidden_source = "default-fallback"

    # PPOAgent.__init__ signature: (state_dim, action_dim, device, lr, hidden_dim, total_steps)
    # We replicate the defaults from ppo_agent.py exactly so the architecture matches.
    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        device=device,
        lr=2.5e-4,
        hidden_dim=resolved_hidden,
        total_steps=500_000,
    )
    agent.load(checkpoint_path)
    agent.actor_critic.eval()
    print(f"[visualize] Loaded checkpoint '{checkpoint_path}'")
    print(f"[visualize] Global step from checkpoint: {agent.global_step:,}")
    print(
        f"[visualize] Env: {env_name}  |  state_dim={state_dim}  "
        f"action_dim={action_dim}  hidden_dim={resolved_hidden} "
        f"(source={hidden_source})"
    )
    return agent


def get_action(agent: PPOAgent, obs: np.ndarray, stochastic: bool) -> np.ndarray:
    """Return the policy action for a single (non-batched) observation."""
    norm_obs = agent.normalize_obs(obs)
    t = torch.FloatTensor(norm_obs).unsqueeze(0).to(agent.device)

    with torch.no_grad():
        if stochastic:
            action, _, _ = agent.actor_critic.get_action_and_value(t)
        else:
            # Deterministic: use the actor mean, then tanh-squash (matches training exactly)
            action_mean, log_std, _ = agent.actor_critic(t)
            action = torch.tanh(action_mean)

    return action.squeeze(0).cpu().numpy()


def _resolve_output_size(
    native_w: int, native_h: int, width: int | None, height: int | None
) -> tuple[int, int]:
    """Pick output dimensions; one of width/height may be omitted to preserve aspect ratio."""
    if width is None and height is None:
        return native_w, native_h
    if width is not None and height is not None:
        return width, height
    if width is not None:
        h = max(1, int(round(native_h * width / native_w)))
        return width, h
    assert height is not None
    w = max(1, int(round(native_w * height / native_h)))
    return w, height


def _resize_frame(frame: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    """Return uint8 RGB (H, W, 3) at the requested size."""
    if frame.shape[1] == out_w and frame.shape[0] == out_h:
        return np.ascontiguousarray(frame, dtype=np.uint8)
    from PIL import Image

    img = Image.fromarray(frame)
    img = img.resize((out_w, out_h), Image.Resampling.LANCZOS)
    return np.asarray(img, dtype=np.uint8)


class _FfmpegVideoWriter:
    """Stream raw RGB frames to ffmpeg (libx264). Requires ffmpeg on PATH."""

    def __init__(
        self,
        path: str,
        width: int,
        height: int,
        fps: float,
        *,
        crf: int = 23,
        preset: str = "medium",
        tune: str | None = None,
        output_pix_fmt: str = "yuv420p",
    ) -> None:
        if shutil.which("ffmpeg") is None:
            raise RuntimeError(
                "ffmpeg was not found on PATH; install ffmpeg to use --video-out, "
                "or add it to PATH."
            )
        cmd: list[str] = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-s",
            f"{width}x{height}",
            "-pix_fmt",
            "rgb24",
            "-r",
            str(fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            preset,
            "-crf",
            str(crf),
            "-pix_fmt",
            output_pix_fmt,
        ]
        if tune:
            cmd.extend(["-tune", tune])
        cmd.append(path)
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def write(self, rgb_hwc: np.ndarray) -> None:
        assert self._proc.stdin is not None
        if rgb_hwc.dtype != np.uint8:
            rgb_hwc = rgb_hwc.astype(np.uint8, copy=False)
        self._proc.stdin.write(np.ascontiguousarray(rgb_hwc).tobytes())

    def close(self) -> None:
        if self._proc is None:
            return
        if self._proc.stdin:
            self._proc.stdin.close()
        ret = self._proc.wait()
        self._proc = None
        if ret != 0:
            raise RuntimeError(f"ffmpeg exited with status {ret}")


# ---------------------------------------------------------------------------
# Main visualisation loop
# ---------------------------------------------------------------------------

def run_visualization(
    env_name: str,
    checkpoint_path: str,
    device: str,
    num_episodes: int,
    stochastic: bool,
    render: bool,
    delay: float,
    video_out: str | None,
    video_fps: float,
    width: int | None,
    height: int | None,
    video_crf: int,
    video_preset: str,
    video_tune: str | None,
    video_pix_fmt: str,
) -> None:
    agent = load_agent(checkpoint_path, env_name, device)

    use_rgb = bool(video_out) or width is not None or height is not None
    if not render and not video_out and (width is not None or height is not None):
        print("[visualize] Ignoring --width/--height without a window or --video-out.")
        width = height = None
        use_rgb = False

    if video_out and shutil.which("ffmpeg") is None:
        print("[visualize] ERROR: --video-out requires ffmpeg on PATH.")
        sys.exit(1)

    if use_rgb:
        render_mode: str | None = "rgb_array"
    elif render:
        render_mode = "human"
    else:
        render_mode = None

    env = gym.make(env_name, render_mode=render_mode)

    ep_stochastic = stochastic
    speed_delay = delay

    print("\n" + "=" * 60)
    if render:
        print("  Window controls:")
        print("    Q / Esc  → quit")
        print("    R        → reset episode now")
        print("    S        → toggle stochastic/deterministic")
        print("    ↑ / ↓   → faster / slower playback")
    if video_out:
        print(
            f"  Recording → {video_out}  @ {video_fps:g} fps  "
            f"(libx264 preset={video_preset} crf={video_crf} pix_fmt={video_pix_fmt})"
        )
    print("=" * 60 + "\n")

    pygame = None
    if render:
        try:
            import pygame as _pygame

            pygame = _pygame
        except ImportError:
            print("[visualize] pygame not found — keyboard controls / scaled window disabled.")

    episode = 0
    total_rewards = []
    video: _FfmpegVideoWriter | None = None
    out_w = out_h = None
    screen = None
    clock = None
    meta_fps = int(env.metadata.get("render_fps", 50))

    try:
        while episode < num_episodes:
            obs, _ = env.reset()
            ep_reward = 0.0
            ep_steps = 0
            done = False
            force_reset = False

            while not done:
                if use_rgb:
                    raw = env.render()
                    if raw is None:
                        raise RuntimeError("env.render() returned None in rgb_array mode")
                    nw, nh = int(raw.shape[1]), int(raw.shape[0])
                    if out_w is None:
                        out_w, out_h = _resolve_output_size(nw, nh, width, height)
                        print(
                            f"[visualize] Output size {out_w}x{out_h} "
                            f"(native {nw}x{nh})"
                        )
                        if video_out and video is None:
                            video = _FfmpegVideoWriter(
                                video_out,
                                out_w,
                                out_h,
                                video_fps,
                                crf=video_crf,
                                preset=video_preset,
                                tune=video_tune,
                                output_pix_fmt=video_pix_fmt,
                            )
                            print(
                                f"[visualize] ffmpeg writer started ({video_out})  "
                                f"output {out_w}x{out_h}  (game native {nw}x{nh}; "
                                "larger size upscales but does not add new detail)"
                            )
                        if render and pygame is not None and screen is None:
                            pygame.init()
                            pygame.display.init()
                            screen = pygame.display.set_mode((out_w, out_h))
                            pygame.display.set_caption("BipedalWalker — visualize.py")
                            clock = pygame.time.Clock()

                    frame = _resize_frame(raw, out_w, out_h)
                    if video is not None:
                        video.write(frame)

                    if render and pygame is not None and screen is not None:
                        surf = pygame.image.frombuffer(
                            frame.tobytes(),
                            (frame.shape[1], frame.shape[0]),
                            "RGB",
                        ).convert()
                        screen.blit(surf, (0, 0))
                        pygame.display.flip()
                        if clock is not None:
                            clock.tick(meta_fps)

                if pygame is not None and render:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            _print_summary(total_rewards)
                            return
                        if event.type == pygame.KEYDOWN:
                            if event.key in (pygame.K_q, pygame.K_ESCAPE):
                                _print_summary(total_rewards)
                                return
                            if event.key == pygame.K_r:
                                force_reset = True
                            elif event.key == pygame.K_s:
                                ep_stochastic = not ep_stochastic
                                mode_label = "stochastic" if ep_stochastic else "deterministic"
                                print(f"  [visualize] Switched to {mode_label} policy")
                            elif event.key == pygame.K_UP:
                                speed_delay = max(0.0, speed_delay - 0.01)
                                print(f"  [visualize] Delay: {speed_delay:.3f}s/step")
                            elif event.key == pygame.K_DOWN:
                                speed_delay = min(0.5, speed_delay + 0.01)
                                print(f"  [visualize] Delay: {speed_delay:.3f}s/step")

                if force_reset:
                    break

                action = get_action(agent, obs, ep_stochastic)
                obs, reward, terminated, truncated, info = env.step(action)
                ep_reward += reward
                ep_steps += 1
                done = terminated or truncated

                if speed_delay > 0:
                    time.sleep(speed_delay)

            if not force_reset:
                episode += 1
                total_rewards.append(ep_reward)
                mode_label = "stochastic" if ep_stochastic else "deterministic"
                print(
                    f"  Episode {episode:>3d}/{num_episodes}  |  "
                    f"Reward: {ep_reward:>8.2f}  |  Steps: {ep_steps:>5d}  |  "
                    f"Policy: {mode_label}"
                )

        _print_summary(total_rewards)
    finally:
        if video is not None:
            video.close()
            print(f"[visualize] Closed video writer ({video_out})")
        env.close()


def _print_summary(rewards):
    if not rewards:
        return
    print("\n" + "=" * 60)
    print(f"  Episodes completed : {len(rewards)}")
    print(f"  Mean reward        : {np.mean(rewards):.2f}")
    print(f"  Std reward         : {np.std(rewards):.2f}")
    print(f"  Min / Max reward   : {np.min(rewards):.2f} / {np.max(rewards):.2f}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Short-clip renderer (used by scripts/try_now/clip_watch.py)
# ---------------------------------------------------------------------------

def render_clip(
    *,
    checkpoint_path: str,
    env_name: str,
    clip_seconds: float,
    video_out: str,
    device: str | None = None,
    env_kwargs: dict | None = None,
    hidden_dim: int | None = None,
    width: int = 1280,
    height: int = 720,
    video_fps: float = 50.0,
    video_crf: int = 23,
    video_preset: str = "medium",
) -> None:
    """Render a short fixed-duration mp4 clip from a checkpoint.

    Reuses the existing `_FfmpegVideoWriter` and the same get_action /
    env.step loop as `run_visualization`, but exits after
    `clip_seconds * env.metadata["render_fps"]` frames instead of
    `num_episodes` episodes. Headless — no pygame, no keyboard input.

    Added in Stage 1 for the `visualize.py --watch-checkpoints` loop
    (see §5.6 of the Try Now plan). Kept here rather than in
    `clip_watch.py` so the ffmpeg invocation stays colocated with
    `_FfmpegVideoWriter` and the numpy-compat shim at the top of this
    file.
    """
    device = device or detect_device()
    agent = load_agent(
        checkpoint_path, env_name, device,
        env_kwargs=env_kwargs, hidden_dim=hidden_dim,
    )

    # Merge caller-supplied env_kwargs with the render-mode flag so a
    # baseline repo can forward e.g. `continuous=True` without being
    # able to drop `render_mode`. Using dict-unpack with render_mode
    # placed last guarantees it wins over anything in env_kwargs and
    # sidesteps the "multiple values for keyword" TypeError that the
    # naive `gym.make(..., render_mode=..., **env_kwargs)` would raise.
    env = gym.make(env_name, **{**(env_kwargs or {}), "render_mode": "rgb_array"})
    meta_fps = int(env.metadata.get("render_fps", 50))
    total_frames = int(max(1, round(clip_seconds * meta_fps)))

    video: _FfmpegVideoWriter | None = None
    out_w = out_h = None
    frames_written = 0

    try:
        while frames_written < total_frames:
            obs, _ = env.reset()
            done = False
            while not done and frames_written < total_frames:
                raw = env.render()
                if raw is None:
                    raise RuntimeError("env.render() returned None in rgb_array mode")
                nw, nh = int(raw.shape[1]), int(raw.shape[0])
                if out_w is None:
                    out_w, out_h = _resolve_output_size(nw, nh, width, height)
                    video = _FfmpegVideoWriter(
                        video_out, out_w, out_h, video_fps,
                        crf=video_crf, preset=video_preset, output_pix_fmt="yuv420p",
                    )
                frame = _resize_frame(raw, out_w, out_h)
                assert video is not None
                video.write(frame)
                frames_written += 1

                action = get_action(agent, obs, stochastic=False)
                obs, _reward, terminated, truncated, _info = env.step(action)
                done = terminated or truncated
    finally:
        if video is not None:
            video.close()
        env.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Interactive visualizer for the BipedalWalker PPO agent.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--env",
        default="BipedalWalkerHardcore-v3",
        help="Gymnasium env ID (default: BipedalWalkerHardcore-v3)",
    )
    parser.add_argument(
        "--checkpoint",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoint.pt"),
        help="Path to checkpoint.pt (default: checkpoint.pt next to this script)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device: 'cpu', 'cuda', 'mps'. Auto-detected when omitted.",
    )
    parser.add_argument(
        "--episodes",
        "--num-episodes",
        "--num_episodes",
        type=int,
        default=10,
        metavar="N",
        help="Number of episodes to run (default: 10). Same as --num-episodes / --num_episodes.",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        default=False,
        help="Sample from the policy distribution rather than using the mean action.",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        default=False,
        help="Disable the render window (headless / benchmark mode).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="Seconds to sleep between steps (default: 0.0 = as fast as possible).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Output width in pixels (scales rgb frames). Omit with --height to preserve aspect.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Output height in pixels (scales rgb frames). Omit with --width to preserve aspect.",
    )
    parser.add_argument(
        "--video-out",
        default=None,
        metavar="PATH",
        help="Write an H.264 MP4 to PATH (requires ffmpeg on PATH). Uses rgb_array pipeline.",
    )
    parser.add_argument(
        "--video-fps",
        type=float,
        default=50.0,
        help="Frames per second for --video-out (default: 50, matching the env render_fps).",
    )
    parser.add_argument(
        "--video-crf",
        type=int,
        default=23,
        metavar="N",
        help="libx264 quality: 0=lossless (huge), 17-18≈visually lossless, 23=default, 28=smaller/worse.",
    )
    parser.add_argument(
        "--video-preset",
        default="medium",
        metavar="NAME",
        help="libx264 speed/size tradeoff: ultrafast…veryslow (default: medium). Slower = better compression.",
    )
    parser.add_argument(
        "--video-tune",
        default=None,
        metavar="NAME",
        help="Optional libx264 -tune (e.g. animation, film). Often try 'animation' for flat vector-style games.",
    )
    parser.add_argument(
        "--video-pix-fmt",
        default="yuv420p",
        choices=("yuv420p", "yuv444p"),
        help="Chroma subsampling: yuv420p (default, widest compatibility) or yuv444p (sharper color edges, larger files).",
    )

    # ── Watch-mode flags (Stage 1, §5.6 of try_now_implementation_plan.md) ──
    # When --watch-checkpoints is set the single-shot path above is
    # bypassed and we hand off to scripts/try_now/clip_watch.py, which
    # polls the directory, renders a clip per new checkpoint, uploads
    # to R2, and writes the URL into --latest-pointer.
    parser.add_argument(
        "--watch-checkpoints",
        default=None,
        metavar="DIR",
        help="Watch DIR for new checkpoint_<step>.pt files; "
             "render + upload a clip per new checkpoint.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=60.0,
        metavar="SECONDS",
        help="Watch-mode poll interval in seconds (default: 60).",
    )
    parser.add_argument(
        "--clip-seconds",
        type=float,
        default=10.0,
        metavar="SECONDS",
        help="Watch-mode clip duration in seconds (default: 10).",
    )
    parser.add_argument(
        "--upload-to",
        default=None,
        metavar="URL",
        help="Watch-mode R2/S3 destination, e.g. s3://try-now-clips/<run_id>/",
    )
    parser.add_argument(
        "--latest-pointer",
        default=None,
        metavar="PATH",
        help="Watch-mode pointer file; the executor tails this and emits clip URLs.",
    )
    parser.add_argument(
        "--clip-tmp-dir",
        default="/tmp/try_now_clips",
        metavar="DIR",
        help="Watch-mode scratch dir for local mp4 files before upload.",
    )
    parser.add_argument(
        "--env-kwargs-json",
        default=None,
        metavar="JSON",
        help=(
            "JSON-encoded env_kwargs forwarded to gym.make() and the agent. "
            "Required for envs whose baseline policy was trained with "
            "non-default kwargs (e.g. LunarLander-v2 with "
            "{\"continuous\": true}). Without this the watch-mode clip "
            "renderer silently instantiates the wrong action space and "
            "every render fails with a shape mismatch."
        ),
    )

    args = parser.parse_args()
    if not 0 <= args.video_crf <= 51:
        parser.error("--video-crf must be between 0 and 51")
    if args.env_kwargs_json is not None:
        import json as _json
        try:
            parsed_kwargs = _json.loads(args.env_kwargs_json)
        except _json.JSONDecodeError as exc:
            parser.error(f"--env-kwargs-json is not valid JSON: {exc}")
        if not isinstance(parsed_kwargs, dict):
            parser.error(
                f"--env-kwargs-json must decode to a JSON object, "
                f"got {type(parsed_kwargs).__name__}"
            )
        args.env_kwargs = parsed_kwargs
    else:
        args.env_kwargs = None
    if args.watch_checkpoints is not None:
        # Watch-mode requires the upload sink and pointer file. Fail
        # fast here rather than 60 s into the loop with a confusing
        # "missing config" traceback from clip_watch.
        missing = [
            flag for flag, val in (
                ("--upload-to", args.upload_to),
                ("--latest-pointer", args.latest_pointer),
            ) if val is None
        ]
        if missing:
            parser.error(
                f"--watch-checkpoints requires {' and '.join(missing)}"
            )
    return args


def detect_device() -> str:
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


if __name__ == "__main__":
    args = parse_args()

    if args.watch_checkpoints is not None:
        # Watch-mode: delegate to deploy_scripts/try_now/clip_watch.py so
        # the single-shot path here stays a leaf function. The import is
        # lazy so `visualize.py` outside watch-mode doesn't grow a
        # `deploy_scripts/` import dependency.
        #
        # Historical note: this used to import `scripts.try_now.clip_watch`,
        # but the file lives under `deploy_scripts/try_now/` (the `scripts/`
        # directory on disk only holds operator shims like `unwedge.sh`).
        # The old path crashed every clip-sidecar restart with
        # ModuleNotFoundError and the systemd unit exhausted its
        # StartLimitBurst budget within a minute — see the bug that sent
        # us here (clips never rendered, UI "Before/Best" stuck on the
        # baseline clip).
        _REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sys.path.insert(0, _REPO_ROOT)
        from deploy_scripts.try_now.clip_watch import main_from_visualize_args  # noqa: E402

        sys.exit(main_from_visualize_args(args))

    device = args.device if args.device else detect_device()
    print(f"[visualize] Using device: {device}")

    if not os.path.isfile(args.checkpoint):
        print(f"[visualize] ERROR: checkpoint not found at '{args.checkpoint}'")
        sys.exit(1)

    run_visualization(
        env_name=args.env,
        checkpoint_path=args.checkpoint,
        device=device,
        num_episodes=args.episodes,
        stochastic=args.stochastic,
        render=not args.no_render,
        delay=args.delay,
        video_out=args.video_out,
        video_fps=args.video_fps,
        width=args.width,
        height=args.height,
        video_crf=args.video_crf,
        video_preset=args.video_preset,
        video_tune=args.video_tune,
        video_pix_fmt=args.video_pix_fmt,
    )
