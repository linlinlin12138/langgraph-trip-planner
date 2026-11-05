"""Fetch real place names for the catalog from open data, once, into data/sources/.

    uv run --with duckdb --with opencc-python-reimplemented python scripts/fetch_catalog_sources.py

Sources (both allow storing and redistributing the data):
* Overture Maps Places (CDLA-Permissive-2.0; some records Apache-2.0 / CC0 from upstream
  datasets, kept per record in ``license``): hotels, restaurants, and attraction-like
  places (parks, museums, historic sites, malls, bars, venues).
* Wikidata (CC0): well-known sights, with the number of language editions (``sitelinks``)
  as a popularity signal.

Commercial map APIs (AMap, Baidu, Tencent, Google) forbid storing their results, so they
are not used here. This script only selects and cleans candidates; picking the final
catalog entries and filling in prices / hours / tags is done offline and deterministically
by ``scripts/build_mock_data.py``.

Network: the Overture S3 bucket over HTTPS (read anonymously), query.wikidata.org, and
DuckDB's httpfs extension (set DUCKDB_HTTPFS_EXTENSION to a local file if the default
http:// extension download is blocked).
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

OUT = Path(__file__).resolve().parents[1] / "data" / "sources"
RELEASE = "2026-10-21.0"
BUCKET = "https://overturemaps-us-west-2.s3.us-west-2.amazonaws.com"
# city -> (slug, center lat, center lon); same centers as the original hand-made catalog
CITIES = {
    "北京": ("bj", 39.9446, 116.3976),
    "成都": ("cd", 30.6746, 104.0356),
    "杭州": ("hz", 30.2468, 120.1378),
    "上海": ("sh", 31.2218, 121.4862),
    "西安": ("xa", 34.2492, 109.0007),
}
D_LAT, D_LON = 0.18, 0.21  # ~20 km box around the center

HOTEL = "basic_category IN ('hotel','lodging') OR cat IN ('hotel','hostel','resort','motel')"
RESTAURANT = "basic_category IN ('restaurant','casual_eatery') OR (cat LIKE '%restaurant%' AND cat <> 'fast_food_restaurant')"
ATTRACTION = (
    "cat IN ('historic_site','palace','monument','temple','buddhist_temple','taoist_temple','pagoda','park','botanical_garden',"
    "'garden','zoo','aquarium','scenic_viewpoint','museum','art_museum','history_museum','science_museum','art_gallery',"
    "'theatre','music_venue','opera_and_ballet','cultural_center','shopping_mall','department_store','market','bar',"
    "'cocktail_bar','pub','night_club','wine_bar','beer_bar','amusement_park','planetarium')"
)
LIMITS = {"hotel": 250, "restaurant": 400, "attraction": 300}

# Wikidata root classes -> our attraction category (subclasses included via P279*)
WD_ROOTS = {
    "Q33506": "museum", "Q22698": "nature", "Q1107656": "nature", "Q23397": "nature", "Q8502": "nature",
    "Q24398318": "history", "Q358": "history", "Q570116": "attraction", "Q24354": "culture",
}  # fmt: skip

_CJK = re.compile(r"[一-鿿]")


def _converter() -> Any:
    from opencc import OpenCC

    return OpenCC("t2s")


def clean_name(name: str | None, cc: Any) -> str | None:
    """Simplified Chinese, Latin-only parts dropped, ASCII parentheses; None if no Chinese left."""
    if not name or not _CJK.search(name):
        return None
    s = cc.convert(name).replace("（", "(").replace("）", ")").replace("　", " ")
    s = re.sub(r"\(\s*[^()一-鿿]*\)", "", s)  # "(Mausoleum of ...)" -> ""
    s = re.sub(r"(?<![\w一-鿿])[A-Za-z][A-Za-z'’&.\- ]*[A-Za-z.](?![\w])", "", s)  # leading/trailing English words
    s = re.sub(r"\s+", " ", s).strip(" -·|/,，")
    return s if _CJK.search(s) and len(s) <= 30 else None


def fetch_overture(cc: Any) -> dict[str, list[dict[str, Any]]]:
    import duckdb

    con = duckdb.connect()
    ext = os.environ.get("DUCKDB_HTTPFS_EXTENSION")
    con.execute(f"INSTALL '{ext}'" if ext else "INSTALL httpfs")
    con.execute("LOAD httpfs")
    proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
    if proxy:
        con.execute(f"SET http_proxy='{urllib.parse.urlparse(proxy).netloc.split('@')[-1]}'")
    if os.environ.get("SSL_CERT_FILE"):
        con.execute(f"SET ca_cert_file='{os.environ['SSL_CERT_FILE']}'")
    listing = urllib.request.urlopen(f"{BUCKET}/?list-type=2&prefix=release/{RELEASE}/theme=places/", timeout=60).read().decode()
    files = [f"{BUCKET}/{k}" for k in re.findall(r"<Key>([^<]+\.parquet)</Key>", listing)]
    boxes = " OR ".join(
        f"(bbox.ymin BETWEEN {la - D_LAT} AND {la + D_LAT} AND bbox.xmin BETWEEN {lo - D_LON} AND {lo + D_LON})"
        for _, la, lo in CITIES.values()
    )
    case = " ".join(
        f"WHEN bbox.ymin BETWEEN {la - D_LAT} AND {la + D_LAT} AND bbox.xmin BETWEEN {lo - D_LON} AND {lo + D_LON} THEN '{slug}'"
        for slug, la, lo in CITIES.values()
    )
    con.execute(f"""
        CREATE TABLE p AS SELECT CASE {case} END AS city, id, names."primary" AS name, basic_category, taxonomy."primary" AS cat,
               confidence, bbox.ymin AS lat, bbox.xmin AS lon, [s.license FOR s IN sources] AS licenses
        FROM read_parquet([{", ".join(f"'{f}'" for f in files)}])
        WHERE ({boxes}) AND coalesce(operating_status, 'open') = 'open' AND confidence >= 0.6
    """)
    out: dict[str, list[dict[str, Any]]] = {slug: [] for slug, _, _ in CITIES.values()}
    for kind, cond in (("hotel", HOTEL), ("restaurant", RESTAURANT), ("attraction", ATTRACTION)):
        rows = con.execute(
            f"SELECT city, id, name, cat, confidence, lat, lon, licenses FROM p WHERE {cond} ORDER BY city, confidence DESC, id"
        ).fetchall()
        for slug in out:
            seen: set[str] = set()
            picked = 0
            for city, oid, name, cat, conf, lat, lon, lic in rows:
                n = clean_name(name, cc) if city == slug else None
                if not n or n in seen:
                    continue
                seen.add(n)
                out[slug].append({"kind": kind, "id": oid, "name": n, "category": cat, "confidence": round(conf, 3),
                                  "lat": round(lat, 5), "lon": round(lon, 5), "license": sorted(set(lic))[0]})  # fmt: skip
                picked += 1
                if picked >= LIMITS[kind]:
                    break
    return out


def _sparql(q: str) -> list[dict[str, Any]]:
    url = "https://query.wikidata.org/sparql?format=json&query=" + urllib.parse.quote(q)
    req = urllib.request.Request(url, headers={"User-Agent": "trip-agents-catalog-builder/0.1 (offline eval catalog)"})
    for i in range(5):
        try:
            return json.load(urllib.request.urlopen(req, timeout=120))["results"]["bindings"]  # type: ignore[no-any-return]
        except Exception as e:
            print(f"  wikidata retry {i + 1}: {e}", file=sys.stderr)
            time.sleep(5 * (i + 1))
    raise RuntimeError("wikidata query failed")


def fetch_wikidata(cc: Any) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for slug, la, lo in CITIES.values():
        items: dict[str, dict[str, Any]] = {}
        for root, cat in WD_ROOTS.items():
            rows = _sparql(f"""SELECT ?item ?zh ?loc ?sitelinks WHERE {{
              SERVICE wikibase:around {{ ?item wdt:P625 ?loc . bd:serviceParam wikibase:center "Point({lo} {la})"^^geo:wktLiteral .
                                         bd:serviceParam wikibase:radius "20" . }}
              ?item wdt:P31/wdt:P279* wd:{root} .
              ?item rdfs:label ?zh . FILTER(LANG(?zh) = "zh")
              ?item wikibase:sitelinks ?sitelinks .
              FILTER NOT EXISTS {{ ?item wdt:P576 ?dissolved }}
            }}""")
            for r in rows:
                qid = r["item"]["value"].rsplit("/", 1)[-1]
                m = re.match(r"Point\(([-\d.]+) ([-\d.]+)\)", r["loc"]["value"])
                name = clean_name(r["zh"]["value"], cc)
                if not m or not name:
                    continue
                prev = items.get(qid)
                if prev is None or (prev["category"] == "attraction" and cat != "attraction"):
                    items[qid] = {"id": qid, "name": name, "category": cat, "sitelinks": int(r["sitelinks"]["value"]),
                                  "lat": round(float(m.group(2)), 5), "lon": round(float(m.group(1)), 5), "license": "CC0-1.0"}  # fmt: skip
            time.sleep(1)
        out[slug] = sorted(items.values(), key=lambda x: (-x["sitelinks"], x["id"]))
        print(f"  wikidata {slug}: {len(out[slug])} items")
    return out


def main() -> None:
    cc = _converter()
    OUT.mkdir(parents=True, exist_ok=True)
    print("overture ...")
    ov = fetch_overture(cc)
    print("wikidata ...")
    wd = fetch_wikidata(cc)
    for city, (slug, la, lo) in CITIES.items():
        doc = {"city": city, "center": [la, lo], "overture_release": RELEASE, "overture": ov[slug], "wikidata": wd[slug]}
        (OUT / f"{slug}.json").write_text(json.dumps(doc, ensure_ascii=False, indent=0), encoding="utf-8")
        kinds = {k: sum(1 for x in ov[slug] if x["kind"] == k) for k in LIMITS}
        print(f"wrote data/sources/{slug}.json: overture {kinds}, wikidata {len(wd[slug])}")


if __name__ == "__main__":
    main()
