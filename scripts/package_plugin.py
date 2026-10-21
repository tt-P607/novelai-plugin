"""Build a portable plugin from an explicit source allowlist, never from user data."""

import json
import shutil
import tempfile
import tomllib
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_FILES = (
    "pyproject.toml",
    "uv.lock",
    "config.example.toml",
    "README.md",
    "LICENSE",
    "NOTICE",
    "skills/novelai/SKILL.md",
    "skills/novelai/references/billing.md",
    "skills/novelai/references/prompting.md",
    "skills/novelai/references/workflows.md",
    "scripts/prepare.py",
    "scripts/call_tool.py",
    "scripts/refresh_schema.py",
    "scripts/package_plugin.py",
    "src/novelai_private/data/novelai-image-openapi.json",
)


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def package() -> Path:
    version = tomllib.loads((ROOT / "pyproject.toml").read_text("utf-8"))["project"]["version"]
    dist = ROOT / "dist"
    dist.mkdir(exist_ok=True)
    # Every build starts empty, so removed presets and stale files cannot survive a rebuild.
    with tempfile.TemporaryDirectory(prefix="public-build-", dir=dist) as temporary:
        marketplace = Path(temporary) / "marketplace"
        plugin = marketplace / "plugin"
        files = [ROOT / name for name in PUBLIC_FILES]
        files.extend(sorted((ROOT / "src/novelai_private").glob("*.py")))
        for source in files:
            if source.is_symlink():
                raise ValueError(f"Refusing a symlink in package sources: {source.name}")
            target = plugin / source.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        metadata = {
            "name": "novelai-private",
            "version": version,
            "description": "NovelAI 图像生成、看图、连续调整和原生图像接口",
            "interface": {
                "displayName": "NovelAI",
                "shortDescription": "生图、看图和连续修图",
                "longDescription": "本机运行，凭据本地配置，支持官方原生图像参数。",
                "developerName": "NovelAI Plugin Contributors",
                "category": "Productivity",
                "capabilities": ["Read", "Write"],
                "defaultPrompt": ["用 NovelAI 生图，并让我继续调整画面。"],
            },
            "skills": "./skills/",
            "mcpServers": "./.mcp.json",
            "requires_local_executor": True,
        }
        write_json(plugin / ".codex-plugin/plugin.json", metadata)
        write_json(
            plugin / "plugin.json",
            {
                "$schema": "https://agent-plugins.org/schemas/1.0.0/plugin.schema.json",
                "name": metadata["name"],
                "version": version,
                "description": metadata["description"],
                "extensions": {
                    "com.openai": {
                        "interface": metadata["interface"],
                        "requires_local_executor": True,
                    }
                },
            },
        )
        entry = {
            "command": "uv",
            "args": [
                "--directory",
                "${PLUGIN_ROOT}",
                "run",
                "--frozen",
                "--no-dev",
                "--cache-dir",
                ".uv-cache",
                "novelai-private",
            ],
            "tool_timeout_sec": 360,
            "startup_timeout_sec": 90,
        }
        write_json(plugin / ".mcp.json", {"mcpServers": {"novelai": entry}})
        write_json(
            plugin / "mcp.json",
            {
                "$schema": "https://agent-plugins.org/schemas/1.0.0/mcp.schema.json",
                "mcpServers": {"novelai": {"type": "stdio", **entry}},
            },
        )
        write_json(
            marketplace / ".agents/plugins/marketplace.json",
            {
                "name": "novelai-tools",
                "interface": {"displayName": "NovelAI Tools"},
                "plugins": [
                    {
                        "name": "novelai-private",
                        "source": {"source": "local", "path": "./plugin"},
                        "policy": {"installation": "AVAILABLE", "authentication": "ON_USE"},
                        "category": "Productivity",
                    }
                ],
            },
        )
        target = dist / "marketplace"
        if target.exists():
            # This exact generated directory is the only tree replaced by the build.
            if target.is_symlink() or target.resolve().parent != dist.resolve():
                raise ValueError("Refusing to replace a marketplace outside dist")
            shutil.rmtree(target)
        shutil.copytree(marketplace, target)
        zip_path = dist / f"novelai-private-{version}.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
            for file in sorted(plugin.rglob("*")):
                if file.is_file():
                    archive.write(file, file.relative_to(plugin))
    print(f"市场目录：{target}")
    print(f"插件包：{zip_path}")
    return target


if __name__ == "__main__":
    package()
