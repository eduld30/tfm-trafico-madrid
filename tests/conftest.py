import os
import time

import pytest
from pyspark.sql import SparkSession


@pytest.fixture(scope="session")
def spark():
    previous_timezone = os.environ.get("TZ")
    os.environ["TZ"] = "Etc/UTC"
    if hasattr(time, "tzset"):
        time.tzset()
    session = None
    try:
        session = (
            SparkSession.builder.master("local[2]")
            .appName("madrid-ml-unit-tests")
            .config("spark.ui.enabled", "false")
            .config("spark.sql.session.timeZone", "Etc/UTC")
            .getOrCreate()
        )
        yield session
    finally:
        if session is not None:
            session.stop()
        if previous_timezone is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = previous_timezone
        if hasattr(time, "tzset"):
            time.tzset()
