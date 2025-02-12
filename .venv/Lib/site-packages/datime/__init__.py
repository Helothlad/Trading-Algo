__version__ = '2023.1'


def pretty_duration_short(duration_seconds: int) -> str:
    """Convert a duration in whole seconds to a string like '1h1m1s'. Durations less than 1 hour return '1m1s', and
    durations less than 1 minute return '1s'.

    >>> pretty_duration_short(12345)
    '3h25m45s'
    """
    hours, seconds_remaining = divmod(duration_seconds, 3600)
    minutes, seconds = divmod(seconds_remaining, 60)
    parts = []
    if hours > 0:
        parts.append(f'{hours}h')
        parts.append(f'{minutes}m')
    elif minutes > 0:
        parts.append(f'{minutes}m')
    parts.append(f'{seconds}s')
    return ''.join(parts)
