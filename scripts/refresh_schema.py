"""Explicitly refresh the public NovelAI image OpenAPI snapshot, without credentials."""

import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
URL = "https://image.novelai.net/docs/doc.json"


if __name__ == "__main__":
    request = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        raw = response.read(4 * 1024 * 1024)
    document = json.loads(raw)
    assert "/ai/generate-image" in document["paths"]
    assert "image.RequestParameters" in document["definitions"]
    path = ROOT / "src/novelai_private/data/novelai-image-openapi.json"
    path.write_bytes(raw)
    count = len(document["definitions"]["image.RequestParameters"]["properties"])
    print(f"官方 schema 已保存，参数数：{count}")
    print("重新打包并刷新本地插件后生效。")
