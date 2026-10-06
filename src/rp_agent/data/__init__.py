"""Everything that talks to rpdata, or holds what it returned.

- `catalogue`: views and parameter metadata.
- `validation`: pre-flight checks, with no get_data call.
- `gateway`: the only caller of `rpdata.get_data`.
- `store`: the per-user dataset store.
- `canonical`: canonical requests and cache keys.
"""

from .canonical import CanonicalRequest
from .catalogue import Catalogue
from .gateway import DataGateway
from .store import DatasetRecord, DataStore, InMemoryDataStore
from .validation import ParamValidator

__all__ = ["CanonicalRequest", "Catalogue", "DataGateway", "DataStore", "DatasetRecord",
           "InMemoryDataStore", "ParamValidator"]
