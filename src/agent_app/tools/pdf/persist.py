"""把渲染产物落盘并生成静态 URL（workflow 节点与 MCP tool url 模式共用）。"""

from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from agent_app.tools.pdf.render import RenderedPage

# 对外暴露渲染图片的静态文件 URL 前缀，需与 main.py 中的挂载点保持一致。
STATIC_URL_PREFIX = "/static/pdf_images"


def persist_rendered_images(
    rendered: list[RenderedPage],
    *,
    output_dir: Path,
    url_prefix: str = STATIC_URL_PREFIX,
    image_format: Literal["png", "jpeg"] = "png",
) -> list[dict[str, Any]]:
    """把渲染产物写入唯一子目录并返回静态 URL 与元数据。

    参数:
        rendered: 已渲染的页面产物列表。
        output_dir: 落盘根目录；每次调用在其中生成唯一子目录。
        url_prefix: 静态 URL 前缀。
        image_format: 文件扩展名对应的图片格式。

    返回值:
        每页的 page、url、width、height 字典列表。
    """
    render_id = uuid4().hex
    out_root = output_dir / render_id
    out_root.mkdir(parents=True, exist_ok=True)

    images: list[dict[str, Any]] = []
    for item in rendered:
        file_name = f"page_{item.page}.{image_format}"
        (out_root / file_name).write_bytes(item.image_bytes)
        images.append(
            {
                "page": item.page,
                "url": f"{url_prefix}/{render_id}/{file_name}",
                "width": item.width,
                "height": item.height,
            }
        )
    return images
