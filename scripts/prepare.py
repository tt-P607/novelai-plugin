"""Create local settings without overwriting an existing configuration or credential."""

import argparse
from pathlib import Path

from novelai_private.config import Settings, config_path

ROOT = Path(__file__).resolve().parents[1]


def prepare(path: Path | None = None) -> Path:
    config = (path or config_path()).expanduser().resolve()
    config.parent.mkdir(parents=True, exist_ok=True)
    if not config.exists():
        config.write_bytes((ROOT / "config.example.toml").read_bytes())
    token = Settings.load(config).token_file
    if not token.exists():
        token.parent.mkdir(parents=True, exist_ok=True)
        token.write_text("", encoding="utf-8")
    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    created = prepare(args.config)
    print(f"本地配置：{created}")
    print(f"请在本机填写 Token：{Settings.load(created).token_file}")
