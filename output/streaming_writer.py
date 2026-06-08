"""
streaming_writer.py
流式 JSON 数组写入器，避免一次性构建大列表占用内存
"""

import json
from typing import Any, Optional


class StreamingJsonArrayWriter:
    """流式写入 JSON 数组文件

    用法:
        with StreamingJsonArrayWriter("output.json") as w:
            for item in items:
                w.append(item)

    每次 append 立即写入文件，内存中仅保留当前条目。
    """

    def __init__(self, path: str, encoder_cls: Optional[type] = None):
        self.path = path
        self.encoder_cls = encoder_cls
        self._f = None
        self._count = 0

    def __enter__(self):
        self._f = open(self.path, 'w', encoding='utf-8')
        self._f.write('[\n')
        self._f.flush()
        return self

    def __exit__(self, *exc):
        if self._f:
            self._f.write('\n]\n')
            self._f.close()

    def append(self, obj: Any):
        """写入一个条目并立即刷盘"""
        if self._count > 0:
            self._f.write(',\n')
        text = json.dumps(obj, ensure_ascii=False, indent=2,
                          cls=self.encoder_cls)
        # 缩进每个条目的行
        indented = '\n  '.join(text.split('\n'))
        self._f.write('  ' + indented)
        self._count += 1
        if self._count % 10 == 0:
            self._f.flush()

    @property
    def count(self) -> int:
        return self._count
