"""Build the fixed catalog under data/mock/ from data/sources/ (offline, deterministic).

    uv run python scripts/build_mock_data.py

* Attractions: the 15 hand-curated landmarks per city (ids ``xx-a00``..``xx-a14``, kept
  stable), plus well-known sights from Wikidata and attraction-like places from Overture,
  until every interest category has a few options.
* Hotels and restaurants: real names and coordinates from Overture Maps.
* Prices, tickets, opening hours, ratings, hotel tier and dietary tags are NOT in the open
  data. They are estimated by the rules below (brand / cuisine keywords, category
  defaults, a fixed seed) and are illustrative only. The catalog only has to be fixed and
  internally consistent for evaluation; it is not a guide to real prices.
* Dietary tags: only restrictions that follow from the cuisine are modelled --
  ``no_spicy`` (not Sichuan / Hunan / hot pot / ...) and ``no_seafood`` (not seafood /
  Japanese / Cantonese ...).

Sources and licences: see data/mock/ATTRIBUTION.md. Re-fetch the sources with
``scripts/fetch_catalog_sources.py``.
"""

from __future__ import annotations

import json
import math
import random
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "sources"
OUT = ROOT / "data" / "mock"

# name, category, ticket, hours, open, close, area, lat, lon  (hand-curated, real landmarks, approximate numbers)
CURATED: dict[str, list[tuple[Any, ...]]] = {
    "北京": [
        ("故宫博物院", "history", 60, 4, "08:30", "17:00", "东城", 39.916, 116.397),
        ("天坛公园", "history", 34, 2.5, "06:00", "21:00", "东城", 39.882, 116.407),
        ("颐和园", "history", 30, 3.5, "06:30", "18:00", "海淀", 39.999, 116.275),
        ("八达岭长城", "history", 40, 5, "07:30", "17:00", "延庆", 40.359, 116.020),
        ("中国国家博物馆", "museum", 0, 3, "09:00", "17:00", "东城", 39.905, 116.401),
        ("景山公园", "nature", 2, 1.5, "06:00", "21:00", "西城", 39.925, 116.396),
        ("北海公园", "nature", 10, 2, "06:30", "21:00", "西城", 39.926, 116.389),
        ("南锣鼓巷", "culture", 0, 1.5, "09:00", "22:00", "东城", 39.937, 116.403),
        ("798艺术区", "culture", 0, 2.5, "10:00", "18:00", "朝阳", 39.984, 116.495),
        ("什刹海", "nightlife", 0, 2, "09:00", "23:30", "西城", 39.940, 116.385),
        ("三里屯太古里", "shopping", 0, 2, "10:00", "22:00", "朝阳", 39.934, 116.454),
        ("王府井大街", "shopping", 0, 2, "09:00", "22:00", "东城", 39.914, 116.411),
        ("雍和宫", "culture", 25, 1.5, "09:00", "16:30", "东城", 39.948, 116.417),
        ("奥林匹克公园", "nature", 0, 2, "06:00", "22:00", "朝阳", 40.001, 116.390),
        ("簋街", "food", 0, 1.5, "11:00", "23:30", "东城", 39.941, 116.421),
    ],
    "上海": [
        ("外滩", "nightlife", 0, 1.5, "06:00", "23:30", "黄浦", 31.240, 121.490),
        ("豫园", "history", 40, 2, "09:00", "16:30", "黄浦", 31.227, 121.492),
        ("上海博物馆", "museum", 0, 2.5, "09:00", "17:00", "黄浦", 31.228, 121.475),
        ("东方明珠", "nightlife", 199, 2, "09:00", "21:30", "浦东", 31.240, 121.500),
        ("上海迪士尼乐园", "culture", 475, 8, "09:00", "21:00", "浦东", 31.144, 121.657),
        ("田子坊", "culture", 0, 1.5, "10:00", "22:00", "黄浦", 31.209, 121.468),
        ("新天地", "shopping", 0, 1.5, "10:00", "23:00", "黄浦", 31.220, 121.475),
        ("南京路步行街", "shopping", 0, 2, "09:00", "22:00", "黄浦", 31.235, 121.478),
        ("武康路", "culture", 0, 1.5, "08:00", "22:00", "徐汇", 31.206, 121.437),
        ("朱家角古镇", "history", 0, 3.5, "08:30", "17:00", "青浦", 31.108, 121.055),
        ("上海科技馆", "museum", 45, 3, "09:00", "17:15", "浦东", 31.218, 121.544),
        ("世纪公园", "nature", 10, 2, "07:00", "18:00", "浦东", 31.216, 121.551),
        ("静安寺", "culture", 50, 1, "07:30", "17:00", "静安", 31.224, 121.446),
        ("城隍庙小吃广场", "food", 0, 1.5, "09:00", "21:00", "黄浦", 31.226, 121.491),
        ("上海中心观光厅", "nightlife", 180, 1.5, "09:00", "22:00", "浦东", 31.234, 121.505),
    ],
    "杭州": [
        ("西湖", "nature", 0, 3, "06:00", "23:00", "西湖", 30.243, 120.150),
        ("灵隐寺", "culture", 75, 2.5, "07:00", "17:30", "西湖", 30.241, 120.101),
        ("雷峰塔", "history", 40, 1.5, "08:00", "20:00", "西湖", 30.231, 120.149),
        ("西溪国家湿地公园", "nature", 80, 3, "08:30", "17:30", "西湖", 30.270, 120.063),
        ("河坊街", "food", 0, 1.5, "09:00", "22:00", "上城", 30.244, 120.169),
        ("中国茶叶博物馆", "museum", 0, 1.5, "09:00", "16:30", "西湖", 30.234, 120.128),
        ("浙江省博物馆", "museum", 0, 2, "09:00", "17:00", "西湖", 30.252, 120.149),
        ("宋城", "culture", 320, 4, "10:00", "21:00", "西湖", 30.177, 120.106),
        ("九溪烟树", "nature", 0, 2, "06:00", "18:00", "西湖", 30.207, 120.124),
        ("龙井村", "nature", 0, 2, "08:00", "18:00", "西湖", 30.225, 120.120),
        ("南宋御街", "history", 0, 1.5, "09:00", "22:00", "上城", 30.249, 120.171),
        ("湖滨银泰in77", "shopping", 0, 2, "10:00", "22:00", "上城", 30.256, 120.163),
        ("钱江新城灯光秀", "nightlife", 0, 1, "18:00", "23:00", "上城", 30.244, 120.214),
        ("良渚博物院", "museum", 0, 2.5, "09:00", "17:00", "余杭", 30.380, 120.022),
        ("拱宸桥", "history", 0, 1.5, "06:00", "22:00", "拱墅", 30.318, 120.141),
    ],
    "成都": [
        ("大熊猫繁育研究基地", "nature", 55, 3.5, "07:30", "18:00", "成华", 30.733, 104.147),
        ("宽窄巷子", "culture", 0, 2, "09:00", "22:00", "青羊", 30.670, 104.054),
        ("锦里", "food", 0, 1.5, "09:00", "22:30", "武侯", 30.646, 104.050),
        ("武侯祠", "history", 50, 2, "08:00", "18:00", "武侯", 30.646, 104.048),
        ("杜甫草堂", "history", 50, 2, "08:00", "18:00", "青羊", 30.660, 104.028),
        ("都江堰", "history", 80, 4, "08:00", "18:00", "都江堰", 31.001, 103.607),
        ("青城山", "nature", 80, 5, "08:00", "17:30", "都江堰", 30.900, 103.570),
        ("春熙路", "shopping", 0, 2, "10:00", "22:00", "锦江", 30.657, 104.081),
        ("成都远洋太古里", "shopping", 0, 2, "10:00", "22:00", "锦江", 30.654, 104.083),
        ("四川博物院", "museum", 0, 2.5, "09:00", "17:00", "青羊", 30.660, 104.035),
        ("金沙遗址博物馆", "museum", 70, 2.5, "08:00", "18:00", "青羊", 30.681, 104.012),
        ("人民公园鹤鸣茶社", "culture", 0, 1.5, "07:00", "21:00", "青羊", 30.658, 104.059),
        ("九眼桥酒吧街", "nightlife", 0, 2, "19:00", "23:30", "锦江", 30.640, 104.090),
        ("建设路小吃街", "food", 0, 1.5, "11:00", "23:00", "成华", 30.676, 104.104),
        ("蜀风雅韵川剧", "culture", 260, 1.5, "19:30", "21:30", "青羊", 30.664, 104.047),
    ],
    "西安": [
        ("秦始皇帝陵博物院(兵马俑)", "history", 120, 4, "08:30", "17:00", "临潼", 34.385, 109.279),
        ("西安城墙", "history", 54, 2.5, "08:00", "22:00", "碑林", 34.259, 108.947),
        ("大雁塔", "history", 25, 1.5, "08:00", "17:30", "雁塔", 34.218, 108.964),
        ("陕西历史博物馆", "museum", 0, 3, "09:00", "17:30", "雁塔", 34.224, 108.954),
        ("回民街", "food", 0, 2, "10:00", "23:00", "莲湖", 34.265, 108.940),
        ("大唐不夜城", "nightlife", 0, 2, "18:00", "23:30", "雁塔", 34.211, 108.966),
        ("钟鼓楼", "history", 50, 1.5, "08:30", "21:30", "碑林", 34.261, 108.943),
        ("华清宫", "history", 120, 3, "07:30", "18:00", "临潼", 34.363, 109.212),
        ("大唐芙蓉园", "culture", 120, 3, "09:00", "22:00", "雁塔", 34.211, 108.974),
        ("西安碑林博物馆", "museum", 65, 2, "08:00", "18:00", "碑林", 34.255, 108.953),
        ("小雁塔", "history", 0, 1.5, "09:00", "17:00", "碑林", 34.240, 108.937),
        ("华山", "nature", 160, 8, "07:00", "19:00", "华阴", 34.478, 110.090),
        ("永兴坊", "food", 0, 1.5, "10:00", "22:00", "新城", 34.269, 108.962),
        ("赛格国际购物中心", "shopping", 0, 2, "10:00", "22:00", "雁塔", 34.233, 108.947),
        ("曲江池遗址公园", "nature", 0, 1.5, "06:00", "22:00", "雁塔", 34.205, 108.981),
    ],
}
CITY_SLUG = {"北京": "bj", "上海": "sh", "杭州": "hz", "成都": "cd", "西安": "xa"}
N_ATTRACTIONS, N_HOTELS, N_RESTAURANTS = 45, 30, 40
MIN_PER_CATEGORY = {"history": 5, "museum": 5, "nature": 5, "culture": 5, "nightlife": 4, "shopping": 4, "food": 3}

# ----------------------------------------------------------------------------- name hygiene
# admin areas, schools, offices, hospitals, transport, half-cleaned names -> not a place to visit / stay / eat
_NOT_A_PLACE = re.compile(r"^\(|^[^()]{0,1}$|[区市县省]\)?$|\([^)]*[市区]\)|^(北京|上海|杭州|成都|西安)$|"
                          r"大学|学院|中学|小学|医院|公司|大厦|写字楼|"
                          r"地铁|火车站|机场|政府|派出所|超市|麦德龙|家乐福|沃尔玛|跳蚤|中南海|"
                          r"历险|极速|飞越|光轮|过山车|探险|之旅|奇遇")  # fmt: skip
# non-curated names must be plain Chinese: Latin or kana usually means a half-translated / foreign-script name
_FOREIGN = re.compile(r"[A-Za-zÀ-ÿ\u3040-\u30ff@「」]")
_RELIGIOUS = re.compile(r"寺|庙|观$|教堂|主教座堂|清真寺|礼拜寺|会堂|道院")
_SHOPPING_NAME = re.compile(r"广场|购物|百货|商场|荟$|中心$|天街|万象城|大悦城|银泰|万达")
_NATURE_NAME = re.compile(r"公园$|湿地|植物园|动物园|湖$|山$|泉$")
_FAST_FOOD = re.compile(r"肯德基|麦当劳|汉堡王|必胜客|星巴克|赛百味|德克士|达美乐|棒约翰|吉野家|味千|真功夫|永和大王|华莱士|老乡鸡|"
                        r"咖啡|甜品|奶茶|茶饮|面包|蛋糕|烘焙|冰淇淋|哈根达斯|喜茶|奈雪|瑞幸|酒吧|超市|便利")  # fmt: skip


def display_name(name: str) -> str:
    return re.sub(r"\s*\((北京|上海|杭州|成都|西安)\)$|^(北京|上海|杭州|成都|西安)\s+", "", name).strip()


def brand(name: str) -> str:
    """Chain key: "海底捞火锅 南京东路店" / "海底捞(望京)" / "北京全聚德烤鸭" -> "海底捞" / "全聚德"."""
    s = re.sub(r"^(北京|上海|杭州|成都|西安)", "", name)
    s = re.split(r"[\s(\-—•·]", s)[0]
    return s[:3]


def norm(name: str) -> str:
    return re.sub(r"[()\s·]|景区|风景区|旅游区|博物院|博物馆|公园|遗址", "", name)


def _km(a: tuple[float, float], b: tuple[float, float]) -> float:
    dy = (a[0] - b[0]) * 111.0
    dx = (a[1] - b[1]) * 111.0 * math.cos(math.radians(a[0]))
    return math.hypot(dx, dy)


def _area(lat: float, lon: float, curated: list[dict[str, Any]]) -> str:
    """District = district of the nearest hand-curated landmark (approximate)."""
    return min(curated, key=lambda a: _km((lat, lon), (a["lat"], a["lon"])))["area"]


# ----------------------------------------------------------------------------- attractions
_OV_CATEGORY = {
    **dict.fromkeys(["park", "botanical_garden", "garden", "zoo", "aquarium", "scenic_viewpoint"], "nature"),
    **dict.fromkeys(["museum", "art_museum", "history_museum", "science_museum", "art_gallery", "planetarium"], "museum"),
    **dict.fromkeys(["historic_site", "palace", "monument", "pagoda"], "history"),
    **dict.fromkeys(
        ["temple", "buddhist_temple", "taoist_temple", "theatre", "music_venue", "opera_and_ballet", "cultural_center", "amusement_park"],
        "culture",
    ),
    **dict.fromkeys(["shopping_mall", "department_store", "market"], "shopping"),
    **dict.fromkeys(["bar", "cocktail_bar", "pub", "night_club", "wine_bar", "beer_bar"], "nightlife"),
}
# category -> (ticket choices, recommended hours, open, close)
_ATTR_DEFAULTS: dict[str, tuple[list[int], float, str, str]] = {
    "history": ([0, 20, 30, 40, 50, 60], 2.0, "08:30", "17:30"),
    "museum": ([0, 0, 0, 20, 30, 50], 2.0, "09:00", "17:00"),
    "nature": ([0, 0, 10, 15, 20, 40], 2.0, "06:30", "21:00"),
    "culture": ([0, 10, 20, 30, 50, 80], 1.5, "09:00", "18:00"),
    "nightlife": ([0, 0, 0, 0, 30, 50], 2.0, "19:00", "23:30"),
    "shopping": ([0], 2.0, "10:00", "22:00"),
    "food": ([0], 1.5, "10:00", "22:30"),
}


def _by_name(name: str, cat: str | None) -> str | None:
    """Source categories are noisy; a few unambiguous name patterns win."""
    if cat is None:
        return None
    if _RELIGIOUS.search(name):
        return "culture"
    if _NATURE_NAME.search(name) and not re.search(r"博物馆|美术馆|纪念馆", name):
        return "nature"
    if _SHOPPING_NAME.search(name) and cat in ("history", "museum", "culture", "attraction"):
        return "shopping"
    return cat


def _wd_category(item: dict[str, Any]) -> str | None:
    name, cat = item["name"], item["category"]
    if cat == "attraction":  # Wikidata "tourist attraction": decide by name
        cat = "nature" if re.search(r"湖|山|园|湿地|泉|溪", name) else "history"
    return _by_name(name, cat)


def _ov_category(item: dict[str, Any]) -> str | None:
    if item["kind"] == "restaurant" and re.search(r"小吃街|美食街|夜市|食街|美食城", item["name"]):
        return "food"
    if item["kind"] != "attraction":
        return None
    return _by_name(item["name"], _OV_CATEGORY.get(item["category"]))


def build_attractions(city: str, src: dict[str, Any], rng: random.Random) -> list[dict[str, Any]]:
    slug = CITY_SLUG[city]
    out = [
        {"id": f"{slug}-a{i:02d}", "name": n, "city": city, "category": c, "ticket_price": float(t), "recommended_hours": h,
         "open_time": o, "close_time": cl, "rating": round(rng.uniform(4.3, 4.9), 1), "area": area, "lat": lat, "lon": lon,
         "source": "curated"}
        for i, (n, c, t, h, o, cl, area, lat, lon) in enumerate(CURATED[city])
    ]  # fmt: skip
    curated = list(out)
    # candidates in priority order: famous Wikidata items first (by language editions), then Overture (by confidence)
    cands: list[tuple[str, dict[str, Any], str]] = []
    for it in src["wikidata"]:
        cat = _wd_category(it)
        if cat and it["sitelinks"] >= 3:
            cands.append((cat, it, f"wikidata:{it['id']}"))
    for it in src["overture"]:
        cat = _ov_category(it)
        if cat:
            cands.append((cat, it, f"overture:{it['id']}"))

    def ok(it: dict[str, Any]) -> bool:
        if _NOT_A_PLACE.search(it["name"]) or _FOREIGN.search(it["name"]):
            return False
        n = norm(it["name"])
        for a in out:
            m = norm(a["name"])
            if n and m and (n in m or m in n):
                return False
            if _km((it["lat"], it["lon"]), (a["lat"], a["lon"])) < 0.25:
                return False
        return True

    def add(cat: str, it: dict[str, Any], source: str) -> None:
        tickets, hours, o, cl = _ATTR_DEFAULTS[cat]
        famous = it.get("sitelinks", 0) >= 10
        out.append({
            "id": f"{slug}-a{len(out):02d}", "name": display_name(it["name"]), "city": city, "category": cat,
            "ticket_price": float(rng.choice(tickets)), "recommended_hours": hours + rng.choice([-0.5, 0, 0, 0.5, 1.0]),
            "open_time": o, "close_time": cl, "rating": round(rng.uniform(4.4, 4.9) if famous else rng.uniform(4.0, 4.7), 1),
            "area": _area(it["lat"], it["lon"], curated), "lat": it["lat"], "lon": it["lon"], "source": source,
        })  # fmt: skip

    for cat, need in MIN_PER_CATEGORY.items():  # 1) every interest category gets a few options
        for c, it, source in cands:
            if sum(a["category"] == cat for a in out) >= need:
                break
            if c == cat and ok(it):
                add(c, it, source)
    for c, it, source in cands:  # 2) fill up with the most prominent remaining places
        if len(out) >= N_ATTRACTIONS:
            break
        if ok(it):
            add(c, it, source)
    return out


# ----------------------------------------------------------------------------- hotels
_LUXURY = re.compile(r"洲际|希尔顿(?!欢朋)|(?<!万)万豪|丽思|香格里拉|四季酒店|凯悦|喜来登|威斯汀|皇冠假日|"
                     r"索菲特|瑞吉|半岛|文华|柏悦|君悦|艾美|康莱德|"
                     r"华尔道夫|宝格丽|安缦|悦榕|铂尔曼|JW|丽晶|国宾馆|朗廷|费尔蒙|"
                     r"嘉里|英迪格|万丽|豪华精选|璞丽|钓鱼台")  # fmt: skip
_BUDGET = re.compile(r"汉庭|如家|7天|七天|锦江之星|格林豪泰|尚客优|宜必思|莫泰|速8|青年旅舍|青旅|宾馆|招待所|旅馆|客栈|公寓|民宿|快捷")
_MIDRANGE = re.compile(
    r"福朋|万豪行政公寓|全季|亚朵|桔子|维也纳|美居|智选假日|(?<!皇冠)假日酒店|丽枫|喆啡|欢朋|万怡|福朋|诺富特|戴斯|华美达|和颐|美豪|秋果|开元|雷迪森|都城"
)
TIER_PRICES = {"budget": (150, 320), "midrange": (380, 750), "luxury": (950, 2200)}
TIER_TARGET = {"budget": 9, "midrange": 12, "luxury": 9}


def build_hotels(city: str, src: dict[str, Any], curated: list[dict[str, Any]], rng: random.Random) -> list[dict[str, Any]]:
    slug, center = CITY_SLUG[city], tuple(src["center"])
    pool = [
        h for h in src["overture"]
        if h["kind"] == "hotel" and not _NOT_A_PLACE.search(h["name"]) and not _FOREIGN.search(h["name"])
        and not re.search(r"餐厅|中餐|饭店.+厅", h["name"]) and _km((h["lat"], h["lon"]), center) < 15
    ]  # fmt: skip
    by_tier: dict[str, list[dict[str, Any]]] = {"luxury": [], "midrange": [], "budget": [], "unknown": []}
    for h in pool:
        n = h["name"]
        t = "midrange" if _MIDRANGE.search(n) else "luxury" if _LUXURY.search(n) else "budget" if _BUDGET.search(n) else "unknown"
        by_tier[t].append(h)
    picked: list[tuple[str, dict[str, Any]]] = []
    for tier, n in TIER_TARGET.items():
        picked += [(tier, h) for h in by_tier[tier][:n]]
    unknown = iter(by_tier["unknown"])  # brandless names fill the gaps, "大酒店/饭店" counted as upscale
    for tier, n in TIER_TARGET.items():
        while sum(t == tier for t, _ in picked) < n:
            h = next(unknown, None)
            if h is None:
                break
            picked.append((tier, h))
    out = []
    for i, (tier, h) in enumerate(sorted(picked, key=lambda x: (list(TIER_TARGET).index(x[0]), -x[1]["confidence"], x[1]["id"]))):
        lo, hi = TIER_PRICES[tier]
        out.append({
            "id": f"{slug}-h{i:02d}", "name": display_name(h["name"]), "city": city, "tier": tier,
            "price_per_night": float(round(rng.uniform(lo, hi), -1)), "rating": round(rng.uniform(3.8, 4.9), 1),
            "area": _area(h["lat"], h["lon"], curated), "lat": h["lat"], "lon": h["lon"], "source": f"overture:{h['id']}",
        })  # fmt: skip
    return out


# ----------------------------------------------------------------------------- restaurants
_CUISINE_BY_CAT = {
    **dict.fromkeys(["japanese_restaurant", "sushi_restaurant", "ramen_restaurant"], "日料"),
    "korean_restaurant": "韩餐",
    **dict.fromkeys(
        [
            "thai_restaurant",
            "vietnamese_restaurant",
            "malaysian_restaurant",
            "indian_restaurant",
            "pakistani_restaurant",
            "turkish_restaurant",
            "singaporean_restaurant",
        ],
        "东南亚及异国菜",
    ),
    **dict.fromkeys(
        [
            "italian_restaurant",
            "pizza_restaurant",
            "french_restaurant",
            "american_restaurant",
            "steakhouse",
            "burger_restaurant",
            "german_restaurant",
            "european_restaurant",
            "spanish_restaurant",
            "mexican_restaurant",
            "bar_and_grill_restaurant",
        ],
        "西餐",
    ),
    "seafood_restaurant": "海鲜",
    **dict.fromkeys(["fondue_restaurant", "hot_pot_restaurant"], "火锅"),
    "barbecue_restaurant": "烧烤",
    **dict.fromkeys(["dim_sum_restaurant", "cantonese_restaurant"], "粤菜"),
    "halal_restaurant": "西北菜",
    **dict.fromkeys(["vegetarian_restaurant", "vegan_restaurant"], "素食"),
}
_STRONG_NAME = [
    (r"涮", "涮肉"), (r"火锅|串串|冒菜|麻辣烫|海底捞", "火锅"), (r"寿司|拉面|居酒屋|日本料理|鮨", "日料"),
    (r"面|饺|包子|馄饨|泡馍|肉夹馍|凉粉|小吃|豆浆|酥肉|粥", "小吃面食"), (r"烧烤|烤肉|烤串", "烧烤"), (r"素", "素食"),
    (r"杭帮|知味|外婆家|绿茶餐厅|新新饭店|楼外楼", "杭帮菜"), (r"烤鸭|全聚德", "京菜"),
]  # fmt: skip
_CUISINE_BY_NAME = [
    (r"火锅|串串|冒菜|麻辣烫", "火锅"), (r"川|蜀|麻辣|钵钵鸡|自贡|乐山|重庆|巴蜀", "川菜"), (r"湘|长沙|毛家", "湘菜"),
    (r"烤肉|烧烤|烤串|烤鱼", "烧烤"), (r"海鲜|渔|蟹|虾", "海鲜"), (r"粤|港式|茶餐厅|早茶|潮汕|广州|顺德", "粤菜"),
    (r"寿司|拉面|居酒屋|日本|和食|料理", "日料"), (r"面|饺|包子|馄饨|泡馍|小吃|夜市|食街|豆浆|粉|馍|酥肉", "小吃面食"),
    (r"烤鸭|京味|老北京", "京菜"), (r"本帮|上海老", "本帮菜"), (r"杭帮|杭州|西湖|知味", "杭帮菜"), (r"素", "素食"),
    (r"清真|回民|兰州|新疆|西北|羊|牛肉", "西北菜"), (r"陕|秦|长安|西安", "陕菜"),
]  # fmt: skip
SPICY_CUISINES = {"川菜", "湘菜", "火锅"}
SEAFOOD_CUISINES = {"海鲜", "日料", "粤菜"}
_SPICY_NAME = re.compile(r"辣|麻|椒|酸菜鱼|烤鱼|冒菜|麻辣烫|水煮|干锅")
_SEAFOOD_NAME = re.compile(r"海鲜|鱼|蟹|虾|渔|寿司|贝|蚝")
# cuisine -> per-person price choices (CNY)
_PRICE = {
    "小吃面食": [25, 30, 35, 40, 50], "烧烤": [70, 90, 110, 130], "火锅": [90, 110, 130, 160], "川菜": [60, 80, 100, 120],
    "湘菜": [60, 80, 100], "粤菜": [100, 130, 160, 220], "日料": [120, 160, 220, 320], "韩餐": [80, 100, 140],
    "西餐": [120, 160, 220, 280], "海鲜": [150, 200, 260, 320], "素食": [60, 80, 100, 120], "东南亚及异国菜": [70, 100, 130, 150],
    "西北菜": [40, 60, 80, 90], "京菜": [80, 120, 160, 200], "本帮菜": [90, 120, 160, 200], "杭帮菜": [80, 110, 150, 190],
    "陕菜": [50, 70, 90, 120], "涮肉": [80, 100, 130], "中餐": [60, 80, 100, 130, 160],
}  # fmt: skip


def cuisine_of(city: str, name: str, cat: str) -> str:
    for pat, c in _STRONG_NAME:
        if re.search(pat, name):
            return c
    if cat in _CUISINE_BY_CAT:
        return _CUISINE_BY_CAT[cat]
    for pat, c in _CUISINE_BY_NAME:
        if re.search(pat, name):
            return c
    return "中餐"  # unknown: no guess (a city default like 川菜 would wrongly mark it spicy)


_MILD_HOTPOT = re.compile(r"椰子鸡|潮汕|牛肉火锅|粥底|菌汤|清汤|澳门|港式|台湾|海鲜火锅")


def dietary_tags(cuisine: str, name: str) -> list[str]:
    tags = []
    mild = cuisine == "火锅" and _MILD_HOTPOT.search(name)
    if (cuisine not in SPICY_CUISINES or mild) and not _SPICY_NAME.search(name):
        tags.append("no_spicy")
    if cuisine not in SEAFOOD_CUISINES and not _SEAFOOD_NAME.search(name):
        tags.append("no_seafood")
    return tags


def build_restaurants(city: str, src: dict[str, Any], curated: list[dict[str, Any]], rng: random.Random) -> list[dict[str, Any]]:
    slug, center = CITY_SLUG[city], tuple(src["center"])
    pool = []
    taken = {norm(a["name"]) for a in curated}
    brands: set[str] = set()
    for r in src["overture"]:
        n = r["name"]
        if r["kind"] != "restaurant" or _FAST_FOOD.search(n) or _NOT_A_PLACE.search(n) or _FOREIGN.search(n) or _ov_category(r) == "food":
            continue
        if any(t and (t in norm(n) or norm(n) in t) for t in taken) or (re.search(r"(店|寺|园)$", n) and len(n) <= 4):
            continue
        if brand(n) in brands:  # one branch per chain
            continue
        brands.add(brand(n))
        if _km((r["lat"], r["lon"]), center) > 15:
            continue
        cuisine = cuisine_of(city, r["name"], r["category"])
        pool.append((r, cuisine, dietary_tags(cuisine, r["name"])))
    # make each restriction actually bind: keep some restaurants that violate it, and some that satisfy it
    picked: list[tuple[dict[str, Any], str, list[str]]] = []
    quotas = [
        (lambda t: "no_spicy" not in t, 8),
        (lambda t: "no_seafood" not in t, 6),
        (lambda t: "no_spicy" in t, 10),
        (lambda t: "no_seafood" in t, 10),
    ]
    for cond, n in quotas:
        for x in pool:
            if sum(cond(p[2]) for p in picked) >= n:
                break
            if cond(x[2]) and x not in picked:
                picked.append(x)
    per_cuisine: dict[str, int] = {}
    for _, c, _t in picked:
        per_cuisine[c] = per_cuisine.get(c, 0) + 1
    for x in pool:  # then variety: at most 10 per cuisine
        if len(picked) >= N_RESTAURANTS:
            break
        if x not in picked and per_cuisine.get(x[1], 0) < 10:
            picked.append(x)
            per_cuisine[x[1]] = per_cuisine.get(x[1], 0) + 1
    out = []
    for i, (r, cuisine, tags) in enumerate(picked):
        upscale = re.search(r"酒家|饭庄|大酒楼|会所|私房|宴", r["name"]) is not None
        price = rng.choice(_PRICE[cuisine]) * (1.4 if upscale else 1.0)
        out.append({
            "id": f"{slug}-r{i:02d}", "name": display_name(r["name"]), "city": city, "cuisine": cuisine,
            "avg_price_per_person": float(round(price / 5) * 5), "rating": round(rng.uniform(3.9, 4.8), 1),
            "dietary_tags": tags, "area": _area(r["lat"], r["lon"], curated), "lat": r["lat"], "lon": r["lon"],
            "source": f"overture:{r['id']}",
        })  # fmt: skip
    return out


def build_city(city: str, rng: random.Random) -> dict[str, Any]:
    src = json.loads((SRC / f"{CITY_SLUG[city]}.json").read_text(encoding="utf-8"))
    attractions = build_attractions(city, src, rng)
    curated = [a for a in attractions if a["source"] == "curated"]
    return {
        "city": city,
        "hotels": build_hotels(city, src, curated, rng),
        "restaurants": build_restaurants(city, src, curated, rng),
        "attractions": attractions,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(20261105)
    for city in CURATED:
        data = build_city(city, rng)
        path = OUT / f"{CITY_SLUG[city]}.json"
        path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        cats = {c: sum(a["category"] == c for a in data["attractions"]) for c in MIN_PER_CATEGORY}
        tiers = {t: sum(h["tier"] == t for h in data["hotels"]) for t in TIER_PRICES}
        diet = {t: sum(t in r["dietary_tags"] for r in data["restaurants"]) for t in ("no_spicy", "no_seafood")}
        print(f"{path.relative_to(ROOT)}: {len(data['attractions'])} attractions {cats}; {len(data['hotels'])} hotels {tiers}; "
              f"{len(data['restaurants'])} restaurants {diet}")  # fmt: skip


if __name__ == "__main__":
    main()
