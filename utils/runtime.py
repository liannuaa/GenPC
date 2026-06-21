from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _mapping_get(mapping, key, default=None):
    if mapping is None:
        return default
    if isinstance(mapping, dict):
        return mapping.get(key, default)
    return getattr(mapping, key, default)


def resolve_path(path, base=PROJECT_ROOT):
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def cfg_section(cfg, name):
    return getattr(cfg, name, None)


def cfg_path(cfg, section, key, legacy_key=None, default=None):
    value = _mapping_get(cfg_section(cfg, section), key)
    if value is None and legacy_key is not None:
        value = getattr(cfg, legacy_key, None)
    if value is None:
        value = default
    if value is None:
        raise KeyError(f"Missing config path: {section}.{key}")
    return resolve_path(value)


def output_dir(cfg):
    return cfg_path(cfg, "paths", "output_dir", legacy_key="output_path", default="workspace")


def data_dir(cfg):
    return cfg_path(cfg, "paths", "data_dir", default="data")


def gt_dir(cfg):
    return cfg_path(cfg, "paths", "gt_dir", default=data_dir(cfg) / "GT")


def models_dir(cfg):
    return cfg_path(cfg, "paths", "models_dir", default="models")


def sample_dir(cfg, flag):
    return output_dir(cfg) / str(flag)


def sample_file(cfg, flag, filename):
    return sample_dir(cfg, flag) / filename


def model_path(cfg, key, default_relative):
    value = _mapping_get(cfg_section(cfg, "models"), key)
    if value is None:
        value = getattr(cfg, key, None)
    if value is None:
        return resolve_path(models_dir(cfg) / default_relative)

    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    if path.parts and path.parts[0] == models_dir(cfg).name:
        return resolve_path(path)
    return resolve_path(models_dir(cfg) / path)


def save_intermediates(cfg):
    outputs = cfg_section(cfg, "outputs")
    value = _mapping_get(outputs, "save_intermediates")
    if value is None:
        value = getattr(cfg, "save_intermediates", None)
    if value is None:
        value = getattr(cfg, "save", False)
    return bool(value)


def normalize_runtime_config(cfg):
    resolved_output = output_dir(cfg)
    cfg.output_path = str(resolved_output)
    cfg.data_dir = str(data_dir(cfg))
    cfg.gt_dir = str(gt_dir(cfg))
    cfg.models_dir = str(models_dir(cfg))
    return cfg


def require_path(path, label):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{label} not found at {path}")
    return path


def cleanup_intermediates(cfg, flag):
    if save_intermediates(cfg):
        return

    outputs = cfg_section(cfg, "outputs")
    keep_files = _mapping_get(outputs, "keep_files")
    if keep_files is None:
        keep_files = getattr(cfg, "keep_files", None)
    if keep_files is None:
        keep_files = [f"{flag}_fused.ply"]
    keep = {str(name).format(flag=flag) for name in keep_files}
    directory = sample_dir(cfg, flag)
    if not directory.exists():
        return

    for path in directory.iterdir():
        if path.name in keep:
            continue
        if path.is_file():
            path.unlink()


def cleanup_stage1_intermediates(cfg, flag):
    if save_intermediates(cfg):
        return

    keep = {"depth.png", "img.png", "point_uv.npy"}
    directory = sample_dir(cfg, flag)
    if not directory.exists():
        return

    for path in directory.iterdir():
        if path.name in keep:
            continue
        if path.is_file():
            path.unlink()
