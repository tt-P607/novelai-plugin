from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from zipfile import ZipFile

from novelai_private import config

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"scripts/{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_portable_package_excludes_private_and_stale_files(tmp_path, monkeypatch):
    module = load_script("package_plugin")
    monkeypatch.setattr(module, "ROOT", tmp_path)
    for name in module.PUBLIC_FILES:
        dest = tmp_path / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if name == "pyproject.toml":
            dest.write_text('[project]\nname="fixture"\nversion="0.2.0"\n')
        else:
            dest.write_text("public fixture")
    (tmp_path / "src/novelai_private/__init__.py").write_text("")
    for name in (
        "local/token.txt",
        "outputs/request.json",
        "skills/novelai/references/personal.md",
    ):
        dest = tmp_path / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text("SYNTHETIC_PRIVATE_SENTINEL")
    module.package()
    stale = tmp_path / "dist/marketplace/plugin/stale-personal.txt"
    stale.write_text("SYNTHETIC_PRIVATE_SENTINEL")
    module.package()
    assert not stale.exists()
    with ZipFile(tmp_path / "dist/novelai-private-0.2.0.zip") as archive:
        names = archive.namelist()
        assert "LICENSE" in names and "NOTICE" in names
        assert not any(n.startswith(("local/", "outputs/")) or "personal" in n for n in names)
        for name in names:
            content = archive.read(name)
            assert b"SYNTHETIC_PRIVATE_SENTINEL" not in content
            assert str(tmp_path).encode() not in content
        entry = json.loads(archive.read(".mcp.json"))["mcpServers"]["novelai"]
        assert entry["command"] == "uv"
        assert "--config" not in entry["args"]


def test_new_config_is_outside_plugin_cache_and_override_is_respected(tmp_path, monkeypatch):
    monkeypatch.delenv("NOVELAI_PRIVATE_CONFIG", raising=False)
    monkeypatch.setattr(config, "__file__", str(tmp_path / "cache/src/novelai_private/config.py"))
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "user")
    assert config.config_path() == tmp_path / "user/.config/novelai-plugin/config.toml"
    legacy = tmp_path / "cache/local/config.toml"
    legacy.parent.mkdir(parents=True)
    legacy.write_text("")
    assert config.config_path() == legacy
    override = tmp_path / "custom/config.toml"
    monkeypatch.setenv("NOVELAI_PRIVATE_CONFIG", str(override))
    assert config.config_path() == override


def test_prepare_preserves_existing_config_and_custom_token(tmp_path):
    module = load_script("prepare")
    path = tmp_path / "config.toml"
    text = 'token_file="custom.txt"\nbase_url="https://image.novelai.net"\n'
    path.write_text(text)
    token = tmp_path / "custom.txt"
    token.write_text("synthetic-token-to-preserve")
    assert module.prepare(path) == path
    assert path.read_text() == text
    assert token.read_text() == "synthetic-token-to-preserve"
