"""Generate the fixed mock dataset under data/mock/.

Attraction names are real landmarks with approximate coordinates / prices.
Hotels and restaurants are fictional and generated with a fixed seed.
All numbers are illustrative only -- this data exists so the demo, tests and
evals run fully offline and deterministically.

Usage:  uv run python scripts/build_mock_data.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "data" / "mock"

# name, category, ticket, hours, open, close, area, lat, lon
ATTRACTIONS: dict[str, list[tuple]] = {
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
CITY_CUISINES = {
    "北京": ["京菜", "烤鸭", "涮羊肉", "炸酱面", "素斋", "粤菜", "清真菜"],
    "上海": ["本帮菜", "生煎小笼", "粤菜", "日料", "素斋", "西餐", "清真菜"],
    "杭州": ["杭帮菜", "面馆", "素斋", "江浙菜", "茶点", "西餐", "清真菜"],
    "成都": ["川菜", "火锅", "串串", "小吃", "素斋", "粤菜", "清真菜"],
    "西安": ["陕菜", "泡馍", "面食", "清真菜", "素斋", "火锅", "西餐"],
}
SPICY_CUISINES = {"川菜", "火锅", "串串", "小吃", "陕菜"}
PORK_FREE_CUISINES = {"清真菜", "素斋", "涮羊肉", "泡馍"}
HOTEL_SUFFIX = ["精选酒店", "快捷酒店", "青年旅舍", "花园酒店", "国际大酒店", "设计酒店", "公寓酒店", "宾馆", "度假酒店", "臻选酒店"]
REST_SUFFIX = ["老店", "食堂", "小馆", "酒家", "坊", "私房菜", "餐厅", "大排档", "楼", "馆"]
TIER_PRICES = {"budget": (150, 320), "midrange": (380, 750), "luxury": (950, 2200)}


def build_city(city: str, rng: random.Random) -> dict:
    slug = CITY_SLUG[city]
    attractions = [
        {
            "id": f"{slug}-a{i:02d}",
            "name": n,
            "city": city,
            "category": c,
            "ticket_price": t,
            "recommended_hours": h,
            "open_time": o,
            "close_time": cl,
            "rating": round(rng.uniform(4.2, 4.9), 1),
            "area": area,
            "lat": lat,
            "lon": lon,
        }
        for i, (n, c, t, h, o, cl, area, lat, lon) in enumerate(ATTRACTIONS[city])
    ]
    lats = sorted(a["lat"] for a in attractions)
    lons = sorted(a["lon"] for a in attractions)
    mlat, mlon = lats[len(lats) // 2], lons[len(lons) // 2]
    central = [a for a in attractions if abs(a["lat"] - mlat) < 0.08 and abs(a["lon"] - mlon) < 0.08]
    areas = sorted({a["area"] for a in central})
    clat = sum(a["lat"] for a in central) / len(central)
    clon = sum(a["lon"] for a in central) / len(central)

    hotels = []
    tiers = ["budget"] * 3 + ["midrange"] * 4 + ["luxury"] * 3
    for i, tier in enumerate(tiers):
        area = rng.choice(areas)
        lo, hi = TIER_PRICES[tier]
        hotels.append(
            {
                "id": f"{slug}-h{i:02d}",
                "name": f"{area}{HOTEL_SUFFIX[i]}(虚构)",
                "city": city,
                "tier": tier,
                "price_per_night": float(round(rng.uniform(lo, hi), -1)),
                "rating": round(rng.uniform(3.8, 4.9), 1),
                "area": area,
                "lat": round(clat + rng.uniform(-0.03, 0.03), 4),
                "lon": round(clon + rng.uniform(-0.03, 0.03), 4),
            }
        )

    restaurants = []
    cuisines = CITY_CUISINES[city]
    for i in range(15):
        cuisine = cuisines[i % len(cuisines)]
        tags: list[str] = []
        if cuisine == "素斋":
            tags = ["vegetarian", "no_pork", "no_seafood"]
        elif cuisine == "清真菜":
            tags = ["halal", "no_pork"]
        elif cuisine in PORK_FREE_CUISINES:
            tags = ["no_pork"]
        if cuisine not in SPICY_CUISINES:
            tags.append("no_spicy")
        if cuisine not in ("日料", "粤菜", "本帮菜", "杭帮菜", "江浙菜") and "no_seafood" not in tags and rng.random() < 0.5:
            tags.append("no_seafood")
        if cuisine not in ("素斋",) and rng.random() < 0.2:
            tags.append("vegetarian")  # has a decent vegetarian menu
        area = rng.choice(areas)
        price = rng.choice([35, 50, 70, 90, 120, 160, 220, 300])
        restaurants.append(
            {
                "id": f"{slug}-r{i:02d}",
                "name": f"{area}{cuisine}{REST_SUFFIX[i % len(REST_SUFFIX)]}(虚构)",
                "city": city,
                "cuisine": cuisine,
                "avg_price_per_person": float(price),
                "rating": round(rng.uniform(3.9, 4.9), 1),
                "dietary_tags": sorted(set(tags)),
                "area": area,
                "lat": round(clat + rng.uniform(-0.03, 0.03), 4),
                "lon": round(clon + rng.uniform(-0.03, 0.03), 4),
            }
        )
    return {"city": city, "hotels": hotels, "restaurants": restaurants, "attractions": attractions}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(20261102)
    for city in ATTRACTIONS:
        data = build_city(city, rng)
        path = OUT / f"{CITY_SLUG[city]}.json"
        path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        print(
            f"wrote {path.relative_to(OUT.parents[1])}: "
            f"{len(data['hotels'])} hotels, {len(data['restaurants'])} restaurants, {len(data['attractions'])} attractions"
        )


if __name__ == "__main__":
    main()
