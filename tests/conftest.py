from pathlib import Path

import duckdb
import pandas as pd
import pytest

DB = Path("data/processed/lab.duckdb")


@pytest.fixture(scope="session")
def con():
    c = duckdb.connect(str(DB), read_only=True)
    yield c
    c.close()


@pytest.fixture(scope="session")
def pit_df(con):
    return con.execute("SELECT * FROM pit_features").df()


@pytest.fixture(scope="session")
def naive_df(con):
    return con.execute("SELECT * FROM naive_features").df()


@pytest.fixture(scope="session")
def asof_df(con):
    return con.execute("SELECT * FROM asof_features").df()


@pytest.fixture(scope="session")
def orders_df(con):
    return con.execute("SELECT * FROM orders").df()