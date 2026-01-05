__all__ = [
    'Memo',
    'get_memo',
]

from typing import Any

Memo = dict[str, Any]

def get_memo(memo: Memo, key: str) -> Memo:
    config: Memo
    if key in memo:
        config = memo[key]
        assert isinstance(config, dict)
    else:
        config = dict()
        memo[key] = config
    return config
