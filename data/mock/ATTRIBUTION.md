# Catalog data sources

The catalog in this folder (`*.json`) is built by `scripts/build_mock_data.py` from the
candidates in `data/sources/` (fetched once by `scripts/fetch_catalog_sources.py`).
Every entry has a `source` field.

| `source` | What | Licence |
|---|---|---|
| `curated` | 15 hand-picked landmarks per city (names real; ticket, hours, coordinates approximate) | part of this project |
| `wikidata:Q…` | Well-known sights: name and coordinates from [Wikidata](https://www.wikidata.org/) | CC0 1.0 |
| `overture:…` | Names and coordinates of hotels, restaurants and other places from [Overture Maps Foundation](https://overturemaps.org/) Places, release 2026-10-21.0 | CDLA-Permissive-2.0 (a few upstream records Apache-2.0 / CC0; see the `license` field in `data/sources/`) |

Contains data from © Overture Maps Foundation and its contributors.

**Everything except names and coordinates is estimated**: prices, tickets, opening hours,
ratings, hotel tier, cuisine and dietary tags come from rules and a fixed random seed in
`scripts/build_mock_data.py`. They are illustrative only and must not be read as real
information about these businesses. The catalog exists so that the demo, tests and
evaluation run offline and deterministically.

Coordinates are WGS84 (Overture / Wikidata). Chinese web maps use GCJ-02; convert before
overlaying the points on such a map.

Data from commercial map APIs (AMap, Baidu, Tencent, Google Places) is not stored here:
their terms forbid caching, storing or building datasets from their results.
