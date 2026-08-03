"""persist_rendered_images 的单元测试。"""

from pathlib import Path

from agent_app.tools.pdf.persist import STATIC_URL_PREFIX, persist_rendered_images
from agent_app.tools.pdf.render import RenderedPage


def _rendered() -> list[RenderedPage]:
    return [
        RenderedPage(page=1, image_bytes=b"a", width=10, height=20),
        RenderedPage(page=2, image_bytes=b"b", width=10, height=20),
    ]


def _relative(url: str) -> str:
    return url.split(f"{STATIC_URL_PREFIX}/", 1)[1]


def test_persist_writes_files_and_returns_urls(tmp_path: Path) -> None:
    images = persist_rendered_images(_rendered(), output_dir=tmp_path)

    assert [img["page"] for img in images] == [1, 2]
    assert images[0]["url"].endswith("/page_1.png")
    assert images[0]["width"] == 10
    assert images[0]["height"] == 20
    assert (tmp_path / _relative(images[0]["url"])).exists()


def test_persist_uses_unique_subdir_per_call(tmp_path: Path) -> None:
    first = persist_rendered_images(_rendered(), output_dir=tmp_path)
    second = persist_rendered_images(_rendered(), output_dir=tmp_path)

    def _render_id(url: str) -> str:
        return _relative(url).split("/", 1)[0]

    assert _render_id(first[0]["url"]) != _render_id(second[0]["url"])


def test_persist_supports_jpeg_extension(tmp_path: Path) -> None:
    images = persist_rendered_images(_rendered(), output_dir=tmp_path, image_format="jpeg")

    assert images[0]["url"].endswith("/page_1.jpeg")
