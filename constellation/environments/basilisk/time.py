from datetime import datetime


def str2datetime(standard_time: str) -> datetime:
    """Turn standard format('YYMMDDHHMMSS') into datetime."""
    standard_format = "%Y%m%d%H%M%S"
    date_object = datetime.strptime(standard_time, standard_format)
    return date_object


def datetime2str(date_object: datetime) -> str:
    """Turn datetime into standard format('YYMMDDHHMMSS')."""
    standard_format = "%Y%m%d%H%M%S"
    standard_time = date_object.strftime(standard_format)
    return standard_time


def datetime2basilisk(date_object: datetime) -> str:
    """Turn datetime into basilisk format."""
    basilisk_format = "%Y %b %d %H:%M:%S.%f (UTC)"
    basilisk_time = date_object.strftime(basilisk_format)
    return basilisk_time
