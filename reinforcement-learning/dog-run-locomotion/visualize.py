"""
visualize.py — Interactive visualizer for the dm_control Dog PPO agent.

Loads checkpoint.pt (saved by ppo_agent.py's PPOAgent.save()) and runs the
policy deterministically inside a rendered dm_control environment via shimmy.

Usage
-----
  python visualize.py                              # defaults (dog-run, best checkpoint)
  python visualize.py --env dm_control/dog-walk-v0 # walk env
  python visualize.py --checkpoint custom.pt       # different checkpoint
  python visualize.py --episodes 5 --stochastic    # sample rather than use mean
  python visualize.py --no-render                  # headless, just print stats
  python visualize.py --width 960 --height 540     # scale the viewer output
  python visualize.py --video-out run.mp4          # record H.264 (requires ffmpeg on PATH)
  python visualize.py --no-render --video-out run.mp4      # headless capture
  python visualize.py --camera-id 1                # different MuJoCo camera
  python visualize.py --video-out hq.mp4 --video-crf 17 --video-preset slow

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
import shimmy  # noqa: F401 — registers dm_control envs
import numpy as np
import torch

# ---------------------------------------------------------------------------
# numpy compatibility shim
# ---------------------------------------------------------------------------
try:
    import numpy._core  # noqa: F401
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
# Import from ppo_agent so we never diverge
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ppo_agent import PPOAgent, ActorCritic, RunningMeanStd, flatten_obs, get_flat_obs_dim  # noqa: F401


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_agent(checkpoint_path: str, env_name: str, device: str) -> PPOAgent:
    """Probe the env for dims, build an agent, then load the checkpoint."""
    state_dim, action_dim = get_flat_obs_dim(env_name)

    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        device=device,
        lr=3e-4,
        hidden_dim=512,
        total_steps=10_000_000,
    )
    agent.load(checkpoint_path)
    agent.actor_critic.eval()
    print(f"[visualize] Loaded checkpoint '{checkpoint_path}'")
    print(f"[visualize] Global step from checkpoint: {agent.global_step:,}")
    print(f"[visualize] Env: {env_name}  |  state_dim={state_dim}  action_dim={action_dim}")
    return agent


def get_action(agent: PPOAgent, obs_flat: np.ndarray, stochastic: bool) -> np.ndarray:
    """Return the policy action for a single (non-batched) flat observation."""
    norm_obs = agent.normalize_obs(obs_flat)
    t = torch.FloatTensor(norm_obs).unsqueeze(0).to(agent.device)

    with torch.no_grad():
        if stochastic:
            _, action_env, _, _ = agent.actor_critic.get_action_and_value(t)
            return action_env.squeeze(0).cpu().numpy()
        else:
            action_mean, log_std, _ = agent.actor_critic(t)
            action = torch.tanh(action_mean)
            return action.squeeze(0).cpu().numpy()


def _resolve_output_size(
    native_w: int, native_h: int, width: int | None, height: int | None
) -> tuple[int, int]:
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
    if frame.shape[1] == out_w and frame.shape[0] == out_h:
        return np.ascontiguousarray(frame, dtype=np.uint8)
    from PIL import Image
    img = Image.fromarray(frame)
    img = img.resize((out_w, out_h), Image.Resampling.LANCZOS)
    return np.asarray(img, dtype=np.uint8)


class _FfmpegVideoWriter:
    """Stream raw RGB frames to ffmpeg (libx264)."""

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
            raise RuntimeError("ffmpeg not found on PATH")
        cmd: list[str] = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-s", f"{width}x{height}",
            "-pix_fmt", "rgb24",
            "-r", str(fps),
            "-i", "-", "-an",
            "-c:v", "libx264",
            "-preset", preset,
            "-crf", str(crf),
            "-pix_fmt", output_pix_fmt,
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
    camera_id: int,
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

    render_kwargs = {'width': width or 640, 'height': height or 480, 'camera_id': camera_id}

    if use_rgb:
        render_mode = "rgb_array"
    elif render:
        render_mode = "human"
    else:
        render_mode = None

    if render_mode:
        env = gym.make(env_name, render_mode=render_mode, render_kwargs=render_kwargs)
    else:
        env = gym.make(env_name)

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
    print(f"  Camera ID: {camera_id}")
    print("=" * 60 + "\n")

    pygame = None
    if render:
        try:
            import pygame as _pygame
            pygame = _pygame
        except ImportError:
            print("[visualize] pygame not found — keyboard controls disabled.")

    episode = 0
    total_rewards = []
    video: _FfmpegVideoWriter | None = None
    out_w = out_h = None
    screen = None
    clock = None
    meta_fps = int(env.metadata.get("render_fps", 40))

    try:
        while episode < num_episodes:
            raw_obs, _ = env.reset()
            obs_flat = flatten_obs(raw_obs)
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
                        print(f"[visualize] Output size {out_w}x{out_h} (native {nw}x{nh})")
                        if video_out and video is None:
                            video = _FfmpegVideoWriter(
                                video_out, out_w, out_h, video_fps,
                                crf=video_crf, preset=video_preset,
                                tune=video_tune, output_pix_fmt=video_pix_fmt,
                            )
                            print(f"[visualize] ffmpeg writer started ({video_out})")
                        if render and pygame is not None and screen is None:
                            pygame.init()
                            pygame.display.init()
                            screen = pygame.display.set_mode((out_w, out_h))
                            pygame.display.set_caption("Dog Run — visualize.py")
                            clock = pygame.time.Clock()

                    frame = _resize_frame(raw, out_w, out_h)
                    if video is not None:
                        video.write(frame)

                    if render and pygame is not None and screen is not None:
                        surf = pygame.image.frombuffer(
                            frame.tobytes(), (frame.shape[1], frame.shape[0]), "RGB",
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

                action = get_action(agent, obs_flat, ep_stochastic)
                raw_obs, reward, terminated, truncated, info = env.step(action)
                obs_flat = flatten_obs(raw_obs)
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
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Interactive visualizer for the dm_control Dog PPO agent.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--env", default="dm_control/dog-run-v0",
        help="Gymnasium env ID (default: dm_control/dog-run-v0)",
    )
    parser.add_argument(
        "--checkpoint",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoint.pt"),
        help="Path to checkpoint.pt",
    )
    parser.add_argument("--device", default=None, help="Torch device (auto-detected when omitted)")
    parser.add_argument(
        "--episodes", "--num-episodes", "--num_episodes",
        type=int, default=10, metavar="N",
        help="Number of episodes to run (default: 10)",
    )
    parser.add_argument("--stochastic", action="store_true", default=False)
    parser.add_argument("--no-render", action="store_true", default=False)
    parser.add_argument("--delay", type=float, default=0.0)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--camera-id", type=int, default=0,
                        help="MuJoCo camera ID for rendering (default: 0 = tracking cam)")
    parser.add_argument("--video-out", default=None, metavar="PATH")
    parser.add_argument("--video-fps", type=float, default=40.0,
                        help="Frames per second for video (default: 40, matching dm_control)")
    parser.add_argument("--video-crf", type=int, default=23, metavar="N")
    parser.add_argument("--video-preset", default="medium", metavar="NAME")
    parser.add_argument("--video-tune", default=None, metavar="NAME")
    parser.add_argument(
        "--video-pix-fmt", default="yuv420p",
        choices=("yuv420p", "yuv444p"),
    )
    args = parser.parse_args()
    if not 0 <= args.video_crf <= 51:
        parser.error("--video-crf must be between 0 and 51")
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
        camera_id=args.camera_id,
    )
