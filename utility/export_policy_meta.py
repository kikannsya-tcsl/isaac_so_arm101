#!/usr/bin/env python3
# =============================================================================
# export_policy_meta.py
#
#   Reads back, from a LIVE Isaac Lab env, everything piper_policy_runner.py
#   needs to reproduce the training-time observation / action pipeline on the
#   real PiPER, and writes it to policy_meta.yaml. Optionally also exports the
#   RSL-RL checkpoint to TorchScript (policy.pt) and ONNX (policy.onnx).
#
#   Why read it back instead of copying the env cfg by hand?
#     The observation vector must be rebuilt EXACTLY as ObservationManager
#     built it: same terms, same order, same articulation joint order (which
#     is the USD traversal order, not joint1..joint6), same default offsets,
#     same scales / clips / history, same binary gripper commands. Every one
#     of those is a classic sim2real bug when transcribed manually.
#
#   Run inside the Isaac Lab container (isaac_so_arm101 venv), from the
#   project root so the task registry is importable:
#
#     cd /workspace/isaac_so_arm101
#     python scripts/export_policy_meta.py \
#         --task <Task-Id> \
#         --checkpoint logs/rsl_rl/lift/<run>/model_1499.pt \
#         --export
#
#   Output (in <ckpt dir>/exported/ unless --out is given):
#     policy_meta.yaml          <- copy to the ROS 2 container
#     policy.pt / policy.onnx   <- with --export (play.py writes the same files)
#
#   If the task includes a camera sensor the env needs --enable_cameras; it is
#   forced on here because it is harmless for a metadata dump.
# =============================================================================

import argparse
import inspect
import importlib
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Export policy metadata for real-robot deployment.")
parser.add_argument("--task", type=str, required=True, help="gym task id (as passed to train.py / play.py)")
parser.add_argument("--checkpoint", type=str, default=None, help="RSL-RL checkpoint, e.g. logs/rsl_rl/lift/<run>/model_1499.pt")
parser.add_argument("--out", type=str, default=None, help="output YAML path (default: <ckpt dir>/exported/policy_meta.yaml)")
parser.add_argument("--export", action="store_true", help="also export policy.pt and policy.onnx next to the YAML")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="agent cfg entry point in the gym registry")
parser.add_argument("--obs-group", type=str, default="policy",
                    help="observation group(s) the ACTOR consumes, comma separated. Default: taken from the "
                         "agent cfg obs_groups['policy'] if present, else the 'policy' group.")
parser.add_argument("--task-module", type=str, default="isaac_so_arm101.tasks",
                    help="python module that registers the task (imported for its side effect)")
parser.add_argument("--encoder", type=str, default=None,
                    help="image encoder to export as encoder.pt. 'term:<obs term>.<attr>' for a model held by a "
                         "class-based term (e.g. term:rgbd.model), or 'pkg.module:attr[.sub]' for a module global. "
                         "Without it, the script only lists nn.Module candidates it can see from the obs function.")
parser.add_argument("--no-camera-sample", action="store_true", help="skip writing sample_camera.npz")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---- everything below needs the sim app to exist --------------------------
import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

import isaaclab_tasks  # noqa: E402,F401  registers the built-in tasks
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg  # noqa: E402

try:
    __import__(args_cli.task_module)  # registers the project's tasks
except ImportError as exc:  # pragma: no cover
    print(f"[warn] could not import {args_cli.task_module}: {exc} "
          f"(fine if the task is registered elsewhere)")


# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------
def _f(x):
    """torch / numpy scalar -> python float."""
    return float(x.item() if hasattr(x, "item") else x)


def _per_joint(x, n):
    """Broadcast scalar / (n,) / (num_envs, n) to a plain list of n floats. None if it can't."""
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().double()
        if x.numel() == 1:
            return [float(x.item())] * n
        if x.numel() % n != 0:
            return None
        return [float(v) for v in x.reshape(-1, n)[0]]
    if isinstance(x, (int, float)):
        return [float(x)] * n
    if isinstance(x, (list, tuple)) and len(x) == n:
        return [float(v) for v in x]
    return None


def _clip_per_joint(x, n):
    """Action clip tensor (num_envs, n, 2) -> [[lo, hi], ...]."""
    if x is None or not isinstance(x, torch.Tensor):
        return None
    x = x.detach().cpu().double().reshape(-1, n, 2)[0]
    return [[float(a), float(b)] for a, b in x.tolist()]


def _names_from_ids(ids, all_names, names_hint=None):
    """SceneEntityCfg.joint_ids / body_ids after resolution -> explicit name list."""
    if isinstance(ids, slice):
        return list(all_names)                      # slice(None) == every joint/body
    if isinstance(ids, torch.Tensor):
        ids = ids.tolist()
    if isinstance(ids, (list, tuple)):
        return [all_names[int(i)] for i in ids]
    if names_hint:
        return list(names_hint)
    return list(all_names)


def _json_safe(v):
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, torch.Tensor):
        return v.detach().cpu().tolist()
    if isinstance(v, (list, tuple)):
        return [_json_safe(e) for e in v]
    if isinstance(v, dict):
        return {str(k): _json_safe(e) for k, e in v.items()}
    return str(v)


def _dump_params(params, scene):
    """Obs-term params -> YAML. SceneEntityCfg is expanded to explicit joint / body names."""
    out = {}
    for k, v in (params or {}).items():
        if hasattr(v, "name") and hasattr(v, "joint_ids"):        # SceneEntityCfg
            entry = {"name": v.name}
            asset = scene[v.name] if v.name in scene.keys() else None
            if asset is not None and hasattr(asset, "joint_names"):
                entry["joint_names"] = _names_from_ids(v.joint_ids, asset.joint_names, getattr(v, "joint_names", None))
            if asset is not None and hasattr(asset, "body_names"):
                entry["body_names"] = _names_from_ids(getattr(v, "body_ids", slice(None)),
                                                      asset.body_names, getattr(v, "body_names", None))
            out[k] = entry
        else:
            out[k] = _json_safe(v)
    return out


def _cfg_to_dict(cfg):
    if cfg is None:
        return None
    if hasattr(cfg, "to_dict"):
        try:
            return _json_safe(cfg.to_dict())
        except Exception:
            pass
    if hasattr(cfg, "__dict__"):
        return {k: _json_safe(v) for k, v in vars(cfg).items() if not k.startswith("_")}
    return _json_safe(cfg)


# -----------------------------------------------------------------------------
def main():
    task = args_cli.task

    # ---- output location ---------------------------------------------------
    if args_cli.out:
        out_path = os.path.abspath(args_cli.out)
    elif args_cli.checkpoint:
        out_path = os.path.join(os.path.dirname(os.path.abspath(args_cli.checkpoint)), "exported", "policy_meta.yaml")
    else:
        out_path = os.path.abspath("policy_meta.yaml")
    out_dir = os.path.dirname(out_path)
    os.makedirs(out_dir, exist_ok=True)

    # ---- build one env exactly as play.py would ----------------------------
    env_cfg = parse_env_cfg(task, device=args_cli.device, num_envs=1)
    env = gym.make(task, cfg=env_cfg)
    env_u = env.unwrapped
    env.reset()

    scene = env_u.scene
    robot = scene["robot"]
    om = env_u.observation_manager
    am = env_u.action_manager
    cm = getattr(env_u, "command_manager", None)

    meta = {
        "schema": 1,
        "generated_by": "export_policy_meta.py",
        "task": task,
        "checkpoint": os.path.abspath(args_cli.checkpoint) if args_cli.checkpoint else None,
    }

    # ---- timing --------------------------------------------------------------
    physics_dt = _f(env_u.physics_dt)
    decimation = int(env_u.cfg.decimation)
    step_dt = _f(env_u.step_dt)
    meta["env"] = {
        "physics_dt": physics_dt,
        "decimation": decimation,
        "step_dt": step_dt,
        "policy_hz": 1.0 / step_dt,
        "episode_length_s": _f(env_u.cfg.episode_length_s),
    }

    # ---- robot -------------------------------------------------------------
    jn = list(robot.joint_names)
    J = len(jn)
    d = robot.data
    limits = getattr(d, "joint_pos_limits", None)
    if limits is None:                              # older Isaac Lab naming
        limits = getattr(d, "joint_limits", None)
    meta["robot"] = {
        "asset_name": "robot",
        "root_body": robot.body_names[0] if robot.body_names else None,
        "body_names": list(robot.body_names),
        "joint_names": jn,
        "default_joint_pos": d.default_joint_pos[0].detach().cpu().tolist(),
        "default_joint_vel": d.default_joint_vel[0].detach().cpu().tolist(),
        "soft_joint_pos_limits": d.soft_joint_pos_limits[0].detach().cpu().tolist(),
        "joint_pos_limits": limits[0].detach().cpu().tolist() if limits is not None else None,
        "joint_vel_limits": _per_joint(getattr(d, "joint_vel_limits", None), J),
    }

    # ---- agent cfg (needed before observations: it says what the ACTOR eats) --
    agent_cfg = None
    try:
        agent_cfg = load_cfg_from_registry(task, args_cli.agent)
    except Exception as exc:
        print(f"[warn] could not load agent cfg '{args_cli.agent}': {exc}")

    # rsl_rl >= 2.3 maps network -> observation groups, e.g.
    #     obs_groups = {"policy": ["policy"], "critic": ["policy", "critic"]}
    # The actor input is the CONCATENATION of its groups, in that order. With an
    # asymmetric setup the critic's extra (privileged) groups are training-only
    # and are deliberately NOT reproduced on the real robot.
    obs_groups_cfg = getattr(agent_cfg, "obs_groups", None)
    if args_cli.obs_group != "policy":
        actor_groups = [g.strip() for g in args_cli.obs_group.split(",")]
        critic_groups = []
    elif isinstance(obs_groups_cfg, dict):
        actor_groups = list(obs_groups_cfg.get("policy") or ["policy"])
        critic_groups = list(obs_groups_cfg.get("critic") or [])
    else:
        actor_groups = ["policy"]
        critic_groups = [g for g in om.active_terms.keys() if g == "critic"]
    for g in actor_groups:
        if g not in om.active_terms:
            raise KeyError(f"observation group '{g}' is not in the env "
                           f"(available: {list(om.active_terms.keys())})")
    print(f"[info] actor observation groups = {actor_groups}"
          + (f"; critic-only groups (not needed on hardware) = "
             f"{[g for g in critic_groups if g not in actor_groups]}" if critic_groups else ""))

    # ---- observations (flattened across the actor's groups, in order) --------
    obs_terms = []
    obs_dim = 0
    for group in actor_groups:
        names = list(om.active_terms[group])
        dims = list(om.group_obs_term_dim[group])
        try:
            term_cfgs = [om.get_term_cfg(group, n) for n in names]
        except AttributeError:                      # very old API
            term_cfgs = list(om._group_obs_term_cfgs[group])
        if hasattr(om, "group_obs_concatenate") and not bool(om.group_obs_concatenate.get(group, True)):
            print(f"[warn] observation group '{group}' has concatenate_terms=False; "
                  f"the runner expects a flat vector.")

        for name, shape, cfg in zip(names, dims, term_cfgs):
            dim = 1
            for s in shape:
                dim *= int(s)
            obs_terms.append({
                "name": name,
                "group": group,
                "func": getattr(cfg.func, "__name__", str(cfg.func)),
                "dim": dim,
                "shape": [int(s) for s in shape],
                "params": _dump_params(getattr(cfg, "params", {}), scene),
                "scale": _json_safe(getattr(cfg, "scale", None)),
                "clip": list(cfg.clip) if getattr(cfg, "clip", None) else None,
                "history_length": int(getattr(cfg, "history_length", 0) or 0),
                "flatten_history_dim": bool(getattr(cfg, "flatten_history_dim", True)),
                "noise_in_training": getattr(cfg, "noise", None) is not None,
            })
        d = 1
        for s in om.group_obs_dim[group]:
            d *= int(s)
        obs_dim += d
    meta["observations"] = obs_terms

    # ---- actions -------------------------------------------------------------
    act_terms = []
    for name in am.active_terms:
        term = am.get_term(name)
        tcfg = term.cfg
        t_jn = list(getattr(term, "_joint_names", None) or [])
        if not t_jn:                                # fall back to regex resolution
            ids, t_jn = robot.find_joints(tcfg.joint_names, preserve_order=getattr(tcfg, "preserve_order", False))
        n = len(t_jn)
        entry = {
            "name": name,
            "type": type(term).__name__,
            "cfg_type": type(tcfg).__name__,
            "dim": int(term.action_dim),
            "joint_names": t_jn,
            "scale": _per_joint(getattr(term, "_scale", getattr(tcfg, "scale", None)), n),
            "offset": _per_joint(getattr(term, "_offset", getattr(tcfg, "offset", None)), n),
            "clip": _clip_per_joint(getattr(term, "_clip", None), n),
            "use_default_offset": bool(getattr(tcfg, "use_default_offset", False)),
            "preserve_order": bool(getattr(tcfg, "preserve_order", False)),
        }
        if hasattr(term, "_open_command"):
            entry["open_command"] = _per_joint(term._open_command, n)
            entry["close_command"] = _per_joint(term._close_command, n)
        if hasattr(tcfg, "rescale_to_limits"):
            entry["rescale_to_limits"] = bool(tcfg.rescale_to_limits)
        if hasattr(tcfg, "alpha"):
            entry["alpha"] = _json_safe(tcfg.alpha)
        act_terms.append(entry)
    act_dim = int(am.total_action_dim)
    meta["actions"] = act_terms

    # ---- commands (targets sampled during training) ---------------------------
    cmds = {}
    if cm is not None:
        for name in getattr(cm, "active_terms", []):
            term = cm.get_term(name)
            tcfg = term.cfg
            entry = {"type": type(term).__name__, "dim": int(term.command.shape[-1])}
            for key in ("body_name", "asset_name", "resampling_time_range", "make_quat_unique"):
                if hasattr(tcfg, key):
                    entry[key] = _json_safe(getattr(tcfg, key))
            if hasattr(tcfg, "ranges"):
                entry["ranges"] = _cfg_to_dict(tcfg.ranges)
            entry["sample"] = term.command[0].detach().cpu().tolist()
            cmds[name] = entry
    meta["commands"] = cmds

    # ---- policy / agent ------------------------------------------------------
    pol_cfg = getattr(agent_cfg, "policy", None) if agent_cfg else None
    net_class = getattr(pol_cfg, "class_name", type(pol_cfg).__name__ if pol_cfg else None)
    recurrent = bool(net_class and "Recurrent" in str(net_class))
    meta["policy"] = {
        "obs_groups": actor_groups,          # concatenated in this order -> actor input
        "critic_only_groups": [g for g in critic_groups if g not in actor_groups],
        "obs_dim": obs_dim,                  # ACTOR input width; the critic's is irrelevant here
        "action_dim": act_dim,
        "class_name": str(net_class) if net_class else None,
        "recurrent": recurrent,
        "deterministic": True,               # exported actor returns the mean, not a sample
        "clip_actions": _json_safe(getattr(agent_cfg, "clip_actions", None)) if agent_cfg else None,
        "empirical_normalization": bool(getattr(agent_cfg, "empirical_normalization", False)) if agent_cfg else None,
        "onnx_input": "obs",
        "onnx_output": "actions",
        "files": {},
    }
    if recurrent:
        print(f"[warn] {net_class} is recurrent: the exported policy carries hidden state. "
              f"The runner resets it on 'run' and must be driven at a fixed rate.")

    # ---- sample observation (env 0) for a runtime self-test ------------------
    obs_all = om.compute()
    sample_obs = torch.cat([obs_all[g][0].reshape(-1) for g in actor_groups]).detach().cpu().float().tolist()
    if len(sample_obs) != obs_dim:
        raise RuntimeError(f"sample obs is {len(sample_obs)} wide but the actor input is {obs_dim}")
    meta["sample"] = {"obs": sample_obs, "action": None}

    # ---- cameras: what the actor's image terms actually saw ------------------
    # Dumped so the real-camera node can be configured to match (resolution, FOV,
    # mount offset, depth clipping) and verified against a sim frame.
    RUNNER_NATIVE = {"joint_pos", "joint_pos_rel", "joint_pos_limit_normalized", "joint_vel", "joint_vel_rel",
                     "object_position_in_robot_root_frame", "generated_commands", "last_action"}
    cameras = {}
    for sname, sensor in getattr(scene, "sensors", {}).items():
        scfg = sensor.cfg
        if not (hasattr(scfg, "width") and hasattr(scfg, "height")):
            continue
        spawn = getattr(scfg, "spawn", None)
        off = getattr(scfg, "offset", None)
        cam = {
            "class": type(sensor).__name__,
            "prim_path": getattr(scfg, "prim_path", None),
            "width": int(scfg.width), "height": int(scfg.height),
            "data_types": list(getattr(scfg, "data_types", [])),
            "update_period": _json_safe(getattr(scfg, "update_period", None)),
            "focal_length": _json_safe(getattr(spawn, "focal_length", None)),
            "horizontal_aperture": _json_safe(getattr(spawn, "horizontal_aperture", None)),
            "clipping_range": _json_safe(getattr(spawn, "clipping_range", None)),
            "offset": {"pos": _json_safe(getattr(off, "pos", None)), "rot": _json_safe(getattr(off, "rot", None)),
                       "convention": _json_safe(getattr(off, "convention", None))} if off is not None else None,
        }
        try:
            cam["intrinsic_matrix"] = sensor.data.intrinsic_matrices[0].detach().cpu().tolist()
        except Exception:
            pass
        fl, ha = cam["focal_length"], cam["horizontal_aperture"]
        if fl and ha:
            cam["hfov_deg"] = float(2.0 * np.degrees(np.arctan(0.5 * float(ha) / float(fl))))
        cameras[sname] = cam
    meta["cameras"] = cameras

    # ---- non-standard observation functions: keep their source next to the meta --
    custom_terms = [t for t in obs_terms if t["func"] not in RUNNER_NATIVE]
    encoder_candidates = []
    if custom_terms:
        src_path = os.path.join(out_dir, "obs_funcs_source.py")
        with open(src_path, "w") as f:
            f.write("# Source of observation terms the real-robot runner cannot compute natively.\n"
                    "# Reproduce these EXACTLY in the node that publishes them (see d405_feature_node.py).\n\n")
            term_cfg_by_name = {}
            for g in actor_groups:
                for n in om.active_terms[g]:
                    try:
                        term_cfg_by_name[(g, n)] = om.get_term_cfg(g, n)
                    except AttributeError:
                        term_cfg_by_name[(g, n)] = om._group_obs_term_cfgs[g][list(om.active_terms[g]).index(n)]
            for t in custom_terms:
                cfg = term_cfg_by_name[(t["group"], t["name"])]
                func = cfg.func
                target = func if inspect.isfunction(func) or inspect.isclass(func) else type(func)
                f.write(f"# ---- term '{t['name']}' : {t['func']}  ({inspect.getmodule(target).__name__})\n")
                try:
                    f.write(inspect.getsource(target) + "\n\n")
                except (OSError, TypeError) as exc:
                    f.write(f"# (source unavailable: {exc})\n\n")
                t["source_file"] = os.path.basename(src_path)
                t["module"] = inspect.getmodule(target).__name__
                # nn.Modules reachable from the term: instance attrs (class-based term) or module globals
                pools = []
                if not inspect.isfunction(func):
                    pools.append((f"term:{t['name']}.", vars(func)))
                pools.append((inspect.getmodule(target).__name__ + ":", vars(inspect.getmodule(target))))
                for prefix, d in pools:
                    for k, v in d.items():
                        if isinstance(v, torch.nn.Module):
                            encoder_candidates.append(f"{prefix}{k}  ({type(v).__name__})")
                        elif not isinstance(v, type) and hasattr(v, "__dict__"):
                            for kk, vv in list(vars(v).items())[:50]:
                                if isinstance(vv, torch.nn.Module):
                                    encoder_candidates.append(f"{prefix}{k}.{kk}  ({type(vv).__name__})")
        print(f"[info] custom obs terms {[t['name'] for t in custom_terms]} -> source in {src_path}")
        if encoder_candidates:
            print("[info] nn.Module candidates for --encoder:")
            for c in dict.fromkeys(encoder_candidates):
                print(f"         {c}")
        meta["encoder_candidates"] = list(dict.fromkeys(encoder_candidates))

    # ---- sample camera frame + the features the env computed from it ----------
    if cameras and custom_terms and not args_cli.no_camera_sample:
        arrays = {}
        for sname, sensor in scene.sensors.items():
            if sname not in cameras:
                continue
            for key, val in sensor.data.output.items():
                arr = val[0].detach().cpu().numpy()
                if key == "rgb":
                    arrays["rgb"] = arr.astype(np.uint8)
                elif key in ("distance_to_image_plane", "depth"):
                    arrays["depth"] = arr.astype(np.float32).reshape(arr.shape[0], arr.shape[1])
                else:
                    arrays[f"{sname}_{key}"] = arr
        off = 0
        for t in obs_terms:
            if t["func"] not in RUNNER_NATIVE:
                arrays[f"features_{t['name']}"] = np.asarray(sample_obs[off:off + t["dim"]], dtype=np.float32)
            off += t["dim"]
        npz = os.path.join(out_dir, "sample_camera.npz")
        np.savez_compressed(npz, **arrays)
        meta["sample"]["camera_file"] = os.path.basename(npz)
        print(f"[info] wrote {npz}: " + ", ".join(f"{k}{v.shape}" for k, v in arrays.items()))

    # ---- image encoder export ----------------------------------------------------
    if args_cli.encoder:
        modname, _, attr = args_cli.encoder.partition(":")
        parts = attr.split(".")
        if modname == "term":                     # instance attribute of a class-based observation term
            hit = [(g, n) for g in actor_groups for n in om.active_terms[g] if n == parts[0]]
            if not hit:
                raise KeyError(f"--encoder term:{parts[0]}: no such observation term in {actor_groups}")
            try:
                obj = om.get_term_cfg(*hit[0]).func
            except AttributeError:
                g, n = hit[0]
                obj = om._group_obs_term_cfgs[g][list(om.active_terms[g]).index(n)].func
            parts = parts[1:]
        else:
            obj = importlib.import_module(modname)
        for part in parts:
            obj = getattr(obj, part)
        if not isinstance(obj, torch.nn.Module):
            raise TypeError(f"--encoder {args_cli.encoder} is a {type(obj).__name__}, not an nn.Module")
        obj = obj.eval()
        enc_path = os.path.join(out_dir, "encoder.pt")
        try:
            scripted = torch.jit.script(obj)
        except Exception as exc:
            print(f"[warn] torch.jit.script failed ({exc}); tracing with a camera-shaped dummy instead")
            cam0 = next(iter(cameras.values()))
            dev = next(obj.parameters()).device
            scripted = None
            for ch in (4, 3, 1):
                try:
                    scripted = torch.jit.trace(obj, torch.zeros(1, ch, cam0["height"], cam0["width"], device=dev))
                    print(f"[info] traced with input (1,{ch},{cam0['height']},{cam0['width']})")
                    break
                except Exception:
                    continue
            if scripted is None:
                raise
        scripted.save(enc_path)
        meta["encoder"] = {"file": "encoder.pt", "source": args_cli.encoder, "class": type(obj).__name__,
                           "n_params": int(sum(p.numel() for p in obj.parameters()))}
        print(f"[info] exported encoder -> {enc_path}")

    # ---- optional export -----------------------------------------------------
    if args_cli.export and args_cli.checkpoint:
        try:
            from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
            from rsl_rl.runners import OnPolicyRunner

            if agent_cfg is None:
                raise RuntimeError("agent cfg unavailable")
            wrapped = RslRlVecEnvWrapper(env, clip_actions=getattr(agent_cfg, "clip_actions", None))
            runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
            runner.load(args_cli.checkpoint)
            # rsl_rl >= 2.3 keeps the network under .policy, older under .actor_critic.
            # Both exporters take only the ACTOR (deepcopy of .actor) plus the ACTOR's
            # normalizer; the critic head never reaches policy.pt / policy.onnx.
            policy_nn = getattr(runner.alg, "policy", None) or getattr(runner.alg, "actor_critic")
            normalizer = getattr(runner, "obs_normalizer", None)     # what play.py passes
            if normalizer is None:
                for attr in ("actor_obs_normalizer", "student_obs_normalizer"):
                    if hasattr(policy_nn, attr):
                        normalizer = getattr(policy_nn, attr)
                        break
            if isinstance(normalizer, dict):          # per-group normalizers (rsl_rl >= 2.3)
                normalizer = normalizer.get(actor_groups[0], next(iter(normalizer.values())))
            export_policy_as_jit(policy_nn, normalizer, path=out_dir, filename="policy.pt")
            export_policy_as_onnx(policy_nn, normalizer=normalizer, path=out_dir, filename="policy.onnx")
            meta["policy"]["files"] = {"jit": "policy.pt", "onnx": "policy.onnx"}
            print(f"[info] exported policy.pt / policy.onnx -> {out_dir}")
        except Exception as exc:
            print(f"[warn] export failed ({exc}). Run play.py once instead; it writes exported/policy.onnx.")

    # sample action through the exported TorchScript, if we have it
    jit_path = os.path.join(out_dir, "policy.pt")
    if os.path.exists(jit_path):
        try:
            model = torch.jit.load(jit_path, map_location="cpu").eval()
            with torch.no_grad():
                a = model(torch.tensor(sample_obs, dtype=torch.float32)[None, :])
            meta["sample"]["action"] = a[0].tolist()
            meta["policy"]["files"].setdefault("jit", "policy.pt")
        except Exception as exc:
            print(f"[warn] could not run {jit_path} on the sample obs: {exc}")
    if os.path.exists(os.path.join(out_dir, "policy.onnx")):
        meta["policy"]["files"].setdefault("onnx", "policy.onnx")

    # ---- write ---------------------------------------------------------------
    with open(out_path, "w") as f:
        yaml.safe_dump(meta, f, sort_keys=False, default_flow_style=None, width=120)

    print()
    print("=" * 70)
    print(f"  wrote {out_path}")
    print(f"  actor obs groups = {actor_groups}"
          + (f"  (critic-only, training only: {meta['policy']['critic_only_groups']})"
             if meta["policy"]["critic_only_groups"] else ""))
    print(f"  obs[{obs_dim}] = " + " | ".join(f"{t['group']}/{t['name']}:{t['dim']}" for t in obs_terms))
    print(f"  act[{act_dim}] = " + " | ".join(f"{t['name']}({t['type']}):{t['dim']}" for t in act_terms))
    print(f"  joints[{J}] = {jn}")
    print(f"  policy rate = {1.0 / step_dt:.1f} Hz (physics {physics_dt} s x decimation {decimation})")
    print("=" * 70)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
