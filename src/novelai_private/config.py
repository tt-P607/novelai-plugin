from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit


class NovelAIError(Exception):
    """An actionable message safe to return to a tool caller."""


def config_path() -> Path:
    override = os.environ.get("NOVELAI_PRIVATE_CONFIG")
    if override:
        return Path(override).expanduser().resolve()
    # Preserve existing source installs; new installs use a stable user-owned directory.
    legacy = Path(__file__).resolve().parents[2] / "local" / "config.toml"
    if legacy.is_file():
        return legacy
    return Path.home() / ".config" / "novelai-plugin" / "config.toml"


@dataclass(frozen=True)
class Settings:
    path: Path
    output_dir: Path
    token_file: Path
    base_url: str = "https://image.novelai.net"
    timeout_seconds: float = 300
    preview_max_side: int = 1536
    proxy: str | None = None
    allowed_image_roots: tuple[Path, ...] = ()

    @classmethod
    def load(cls, path: Path | None = None) -> Settings:
        path = (path or config_path()).resolve()
        values = tomllib.loads(path.read_text(encoding="utf-8-sig")) if path.exists() else {}

        def relative(value: str) -> Path:
            p = Path(value).expanduser()
            return (p if p.is_absolute() else path.parent / p).resolve()

        base_url = values.get("base_url", "https://image.novelai.net").rstrip("/")
        parts = urlsplit(base_url)
        local = parts.hostname in {"localhost", "127.0.0.1", "::1"}
        if (
            parts.scheme not in ({"https", "http"} if local else {"https"})
            or not parts.hostname
            or parts.username
            or parts.password
            or parts.query
            or parts.fragment
        ):
            raise NovelAIError("base_url 必须是 HTTPS 地址（本机测试允许 HTTP），不能含凭据。")
        timeout = float(values.get("timeout_seconds", 300))
        preview = int(values.get("preview_max_side", 1536))
        if timeout <= 0 or preview < 64:
            raise NovelAIError("timeout_seconds 必须大于 0，preview_max_side 至少为 64。")
        return cls(
            path=path,
            output_dir=relative(values.get("output_dir", "../outputs")),
            token_file=relative(values.get("token_file", "token.txt")),
            base_url=base_url,
            timeout_seconds=timeout,
            preview_max_side=preview,
            proxy=values.get("proxy"),
            allowed_image_roots=tuple(relative(p) for p in values.get("allowed_image_roots", [])),
        )

    def token(self) -> str:
        token = os.environ.get("NOVELAI_API_TOKEN", "").strip()
        if not token and self.token_file.exists():
            token = self.token_file.read_text(encoding="utf-8-sig").strip()
        if not token:
            raise NovelAIError(f"尚未配置 Token。请在本机填写 {self.token_file}，不必发送到聊天。")
        if "\n" in token or "\r" in token:
            raise NovelAIError("Token 文件应仅包含一行 Persistent API Token。")
        return token.removeprefix("Bearer ").strip()

    def check_file(self, value: str) -> Path:
        path = Path(value).expanduser()
        if not path.is_absolute():
            raise NovelAIError("素材路径必须是绝对路径。")
        path = path.resolve()
        if self.allowed_image_roots and not any(
            path.is_relative_to(root) for root in self.allowed_image_roots
        ):
            raise NovelAIError("素材路径不在配置的 allowed_image_roots 内。")
        if not path.is_file():
            raise NovelAIError(f"素材文件不存在：{path}")
        if path.stat().st_size > 128 * 1024 * 1024:
            raise NovelAIError("素材超过本地 128 MiB 读取上限。")
        return path
