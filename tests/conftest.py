"""Reusable isolated demo collection fixtures."""

from collections.abc import Iterator

import pytest

from service import Collection, create_collection, dispose


@pytest.fixture(scope="module")
def demo_collection() -> Iterator[Collection]:
    collection = create_collection()
    yield collection
    dispose(collection)
