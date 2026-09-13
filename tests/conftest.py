import os
import sys
import time

import pytest
from pyspark.sql import SparkSession


@pytest.fixture(scope="session")
def spark():
    previous_timezone = os.environ.get("TZ")
    previous_pyspark_python = os.environ.get("PYSPARK_PYTHON")
    previous_pyspark_driver_python = os.environ.get("PYSPARK_DRIVER_PYTHON")
    os.environ["TZ"] = "Etc/UTC"
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
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
        if previous_pyspark_python is None:
            os.environ.pop("PYSPARK_PYTHON", None)
        else:
            os.environ["PYSPARK_PYTHON"] = previous_pyspark_python
        if previous_pyspark_driver_python is None:
            os.environ.pop("PYSPARK_DRIVER_PYTHON", None)
        else:
            os.environ["PYSPARK_DRIVER_PYTHON"] = previous_pyspark_driver_python
        if hasattr(time, "tzset"):
            time.tzset()
