from datetime import datetime
from pathlib import Path


def parse_kitti_timestamp(timestamp_str: str):
    """
    解析 KITTI timestamp

    KITTI格式:
    2011-10-03 12:55:34.997992704

    返回:
    datetime对象
    """

    timestamp_str = timestamp_str.strip()

    # 去除纳秒最后多余位
    if "." in timestamp_str:
        date_part, nano_part = timestamp_str.split(".")

        # datetime最多支持6位微秒
        nano_part = nano_part[:6]

        timestamp_str = (
            date_part +
            "." +
            nano_part
        )

        return datetime.strptime(
            timestamp_str,
            "%Y-%m-%d %H:%M:%S.%f"
        )

    else:

        return datetime.strptime(
            timestamp_str,
            "%Y-%m-%d %H:%M:%S"
        )


def load_kitti_timestamps(timestamp_file):
    """
    读取KITTI timestamps.txt

    返回:
    list[datetime]
    """

    timestamp_file = Path(timestamp_file)

    if not timestamp_file.exists():
        raise FileNotFoundError(
            f"Timestamp file not found: {timestamp_file}"
        )

    timestamps = []

    with open(timestamp_file, "r") as f:

        for line in f:

            if line.strip():

                timestamps.append(
                    parse_kitti_timestamp(line)
                )

    return timestamps
