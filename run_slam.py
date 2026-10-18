import argparse


def get_args():
    parser = argparse.ArgumentParser(description='Arguments to SLAM pipeline')
    parser.add_argument('config_path', type=str)
    parser.add_argument('--input_path', default="")
    parser.add_argument('--output_path', default="")
    parser.add_argument('--seed', type=int)
    parser.add_argument('--multi_gpu', action='store_true')
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--wandb_offline', action='store_true')
    parser.add_argument('--wandb_entity', type=str)
    parser.add_argument('--experiment_name', type=str)
    parser.add_argument('--group_name', type=str)
    parser.add_argument(
        '--registration_method',
        choices=[
            'fpfh', 'dinov2', 'gaussian_landmark',
            'sift', 'orb', 'akaze'],
        help='Override submap.registration_method for a comparable experiment run')
    parser.add_argument(
        '--descriptor_mode', choices=['baseline', 'single', 'multiview'],
        help='Select baseline, single-view, or multi-view Gaussian registration')
    parser.add_argument(
        '--max_landmarks', type=int, choices=[256, 512, 1024],
        help='Landmark budget K for single/multiview ablations')
    parser.add_argument(
        '--ambiguity_margin', type=float,
        help='Override the bidirectional Top-1/Top-2 descriptor margin')
    parser.add_argument(
        '--geometric_consistency',
        choices=['none', 'distance', 'distance_scale', 'distance_scale_relative_axis'],
        help='Select geometric-consistency ablation')
    parser.add_argument(
        '--registration_device', choices=['cpu', 'cuda'],
        help='Device for DINO patch extraction during inter-agent registration')
    parser.add_argument(
        '--registration_weights_path',
        help='DINO weights used by dinov2/gaussian_landmark registration')
    parser.add_argument(
        '--disable_fpfh_fallback', action='store_true',
        help='Do not fall back to FPFH when semantic coarse registration fails')
    parser.add_argument(
        '--enable_global_compaction', action='store_true',
        help='Enable post-PGO Gaussian compaction and periodic refinement pruning')
    parser.add_argument(
        '--disable_global_compaction', action='store_true',
        help='Disable post-PGO Gaussian compaction')
    parser.add_argument(
        '--reliability_mode', choices=['paper', 'legacy_exp'],
        help='Landmark reliability formula (paper is the production default)')
    parser.add_argument(
        '--enable_tsdf_export', action='store_true',
        help='Export a colored TSDF mesh after the completed SLAM run')
    parser.add_argument(
        '--disable_tsdf_export', action='store_true',
        help='Disable post-SLAM TSDF export')
    parser.add_argument(
        '--tsdf_zip', action='store_true',
        help='ZIP TSDF artifacts when post-SLAM export is enabled')
    return parser.parse_args()


def update_config_with_args(config, args):
    if args.input_path:
        config["data"]["input_path"] = args.input_path
    if args.output_path:
        config["data"]["output_path"] = args.output_path
    # Seed 0 is a valid and commonly used benchmark seed.
    if args.seed is not None:
        config["seed"] = args.seed
    if args.multi_gpu:
        config["multi_gpu"] = True
    if args.use_wandb:
        config["use_wandb"] = True
    if args.wandb_entity:
        config["wandb_entity"] = args.wandb_entity
    if args.wandb_offline:
        config["wandb_offline"] = True
    if args.experiment_name:
        config["experiment_name"] = args.experiment_name
    if args.group_name:
        config["group_name"] = args.group_name
    if args.registration_method:
        config["submap"]["registration_method"] = args.registration_method
    if getattr(args, "descriptor_mode", None):
        config.setdefault("registration", {})["descriptor_mode"] = args.descriptor_mode
    if getattr(args, "max_landmarks", None) is not None:
        config.setdefault("registration", {})["max_landmarks"] = args.max_landmarks
    if getattr(args, "ambiguity_margin", None) is not None:
        config.setdefault("registration", {})["ambiguity_margin"] = args.ambiguity_margin
    if getattr(args, "geometric_consistency", None):
        config.setdefault("registration", {})["geometric_consistency"] = (
            args.geometric_consistency)
    if getattr(args, "reliability_mode", None):
        config.setdefault("registration", {})["reliability_mode"] = (
            args.reliability_mode)
    if args.registration_device:
        config["submap"]["registration_device"] = args.registration_device
    if args.registration_weights_path:
        config["submap"]["registration_weights_path"] = (
            args.registration_weights_path)
    if args.disable_fpfh_fallback:
        config["submap"]["fallback_to_fpfh"] = False
        config.setdefault("registration", {})["fallback_to_fpfh"] = False
    enable_compaction = getattr(args, "enable_global_compaction", False)
    disable_compaction = getattr(args, "disable_global_compaction", False)
    if enable_compaction and disable_compaction:
        raise ValueError("global compaction cannot be both enabled and disabled")
    if enable_compaction:
        config.setdefault("global_compaction", {})["enabled"] = True
    if disable_compaction:
        config.setdefault("global_compaction", {})["enabled"] = False
    enable_tsdf = getattr(args, "enable_tsdf_export", False)
    disable_tsdf = getattr(args, "disable_tsdf_export", False)
    if enable_tsdf and disable_tsdf:
        raise ValueError("TSDF export cannot be both enabled and disabled")
    tsdf_config = config.setdefault("export", {}).setdefault("tsdf", {})
    if enable_tsdf:
        tsdf_config["enabled"] = True
    if disable_tsdf:
        tsdf_config["enabled"] = False
    if getattr(args, "tsdf_zip", False):
        tsdf_config["zip"] = True
    return config


def validate_registration_config(config):
    """Fail early when the selected method cannot run in the SLAM path."""
    submap_config = config.setdefault("submap", {})
    registration_config = config.setdefault("registration", {})
    descriptor_mode = str(
        registration_config.get("descriptor_mode", "baseline")).lower()
    if descriptor_mode not in {"baseline", "single", "multiview"}:
        raise ValueError(
            "registration.descriptor_mode must be baseline, single, or multiview")
    registration_config["descriptor_mode"] = descriptor_mode
    reliability_mode = str(
        registration_config.get("reliability_mode", "paper")).lower()
    if reliability_mode not in {"paper", "legacy_exp"}:
        raise ValueError(
            "registration.reliability_mode must be paper or legacy_exp")
    registration_config["reliability_mode"] = reliability_mode
    if descriptor_mode in {"single", "multiview"}:
        submap_config["registration_method"] = "gaussian_landmark"
    method = submap_config.get("registration_method", "fpfh").lower()
    submap_config["registration_method"] = method
    supported = {
        "fpfh", "dinov2", "gaussian_landmark", "sift", "orb", "akaze"
    }
    if method not in supported:
        raise ValueError(
            f"Unsupported submap.registration_method={method!r}; "
            f"expected one of {sorted(supported)}")

    anchor_data = submap_config.get("anchor_data")
    if method != "fpfh" and anchor_data not in {"depth", "render_depth"}:
        raise ValueError(
            f"registration_method={method!r} requires anchor_data='depth' "
            "or 'render_depth'.")

    submap_config.setdefault("initial_transformation_unknown", True)
    submap_config.setdefault("PGO_GTSAM", True)
    submap_config.setdefault("fallback_to_fpfh", True)
    submap_config.setdefault("registration_options", {})
    registration_config.setdefault(
        "fallback_to_fpfh", submap_config["fallback_to_fpfh"])
    config.setdefault("global_compaction", {}).setdefault("enabled", False)
    config.setdefault("export", {}).setdefault("tsdf", {}).setdefault(
        "enabled", False)
    return config


if __name__ == "__main__":
    args = get_args()

    # Keep argument discovery usable before optional CUDA/SLAM dependencies are
    # installed; real execution still imports and validates the full pipeline.
    from src.entities.magic_slam import MAGiCSLAM
    from src.utils.io_utils import load_config
    from src.utils.utils import setup_seed

    config = load_config(args.config_path)
    config = update_config_with_args(config, args)
    config = validate_registration_config(config)

    setup_seed(config["seed"])
    coga_slam = MAGiCSLAM(config)
    coga_slam.run()
