from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import requests



def save_api_response(data: Dict, region_id: str, query: str, page_num: int, sub_id: str = "full"):
    cache_dir = Path("./api_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    safe_query = query.replace("/", "_")
    filename = f"{region_id}__{sub_id}__{safe_query}__page{page_num}.json"
    filepath = cache_dir / filename

    with open(filepath, "w", encoding="utf-8") as f:
        import json
        json.dump(data, f, ensure_ascii=False, indent=2)


BAIDU_AROUND_URL = "https://api.map.baidu.com/place/v3/around"

DEFAULT_QUERIES = [
    "美食",
    "住宿",
    "购物",
    "教育",
    "培训",
    "棋艺",
    "医疗",
    "交通",
    "金融",
    "公司企业",
    "生活",
    "休闲娱乐",
    "旅游景点",
    "政府机构",
    "文化场馆",
    "房地产",
]


# ----- Coordinate conversion: GCJ-02 <-> WGS84 -----

_PI = math.pi
_A = 6378245.0
_EE = 0.00669342162296594323


def _out_of_china(lon: float, lat: float) -> bool:
    return not (73.66 < lon < 135.05 and 3.86 < lat < 53.55)


def _transformlat(lon: float, lat: float) -> float:
    ret = -100.0 + 2.0 * lon + 3.0 * lat + 0.2 * lat * lat + 0.1 * lon * lat + 0.2 * math.sqrt(abs(lon))
    ret += (20.0 * math.sin(6.0 * lon * _PI) + 20.0 * math.sin(2.0 * lon * _PI)) * 2.0 / 3.0
    ret += (20.0 * math.sin(lat * _PI) + 40.0 * math.sin(lat / 3.0 * _PI)) * 2.0 / 3.0
    ret += (160.0 * math.sin(lat / 12.0 * _PI) + 320 * math.sin(lat * _PI / 30.0)) * 2.0 / 3.0
    return ret


def _transformlon(lon: float, lat: float) -> float:
    ret = 300.0 + lon + 2.0 * lat + 0.1 * lon * lon + 0.1 * lon * lat + 0.1 * math.sqrt(abs(lon))
    ret += (20.0 * math.sin(6.0 * lon * _PI) + 20.0 * math.sin(2.0 * lon * _PI)) * 2.0 / 3.0
    ret += (20.0 * math.sin(lon * _PI) + 40.0 * math.sin(lon / 3.0 * _PI)) * 2.0 / 3.0
    ret += (150.0 * math.sin(lon / 12.0 * _PI) + 300.0 * math.sin(lon / 30.0 * _PI)) * 2.0 / 3.0
    return ret


def gcj02_to_wgs84(lon: float, lat: float) -> Tuple[float, float]:
    if _out_of_china(lon, lat):
        return lon, lat
    dlat = _transformlat(lon - 105.0, lat - 35.0)
    dlon = _transformlon(lon - 105.0, lat - 35.0)
    radlat = lat / 180.0 * _PI
    magic = math.sin(radlat)
    magic = 1 - _EE * magic * magic
    sqrtmagic = math.sqrt(magic)
    dlat = (dlat * 180.0) / ((_A * (1 - _EE)) / (magic * sqrtmagic) * _PI)
    dlon = (dlon * 180.0) / (_A / sqrtmagic * math.cos(radlat) * _PI)
    mglat = lat + dlat
    mglon = lon + dlon
    return lon * 2 - mglon, lat * 2 - mglat


# ----- Tile helpers -----

@dataclass(frozen=True)
class TileBBox:
    tile_x: int
    tile_y: int
    zoom: int
    lat_s: float
    lat_n: float
    lon_w: float
    lon_e: float

    @property
    def center_lat(self) -> float:
        return (self.lat_s + self.lat_n) / 2.0

    @property
    def center_lon(self) -> float:
        return (self.lon_w + self.lon_e) / 2.0

    @property
    def corners(self) -> Dict[str, Tuple[float, float]]:
        return {
            "NW": (self.lat_n, self.lon_w),
            "NE": (self.lat_n, self.lon_e),
            "SW": (self.lat_s, self.lon_w),
            "SE": (self.lat_s, self.lon_e),
        }


@dataclass(frozen=True)
class SubBBox:
    parent_region_id: str
    sub_id: str
    row_idx: int
    col_idx: int
    rows: int
    cols: int
    lat_s: float
    lat_n: float
    lon_w: float
    lon_e: float

    @property
    def center_lat(self) -> float:
        return (self.lat_s + self.lat_n) / 2.0

    @property
    def center_lon(self) -> float:
        return (self.lon_w + self.lon_e) / 2.0


def lon_from_tile_x(x: int, z: int) -> float:
    return x / (2 ** z) * 360.0 - 180.0


def lat_from_tile_y(y: int, z: int) -> float:
    n = math.pi - (2.0 * math.pi * y) / (2 ** z)
    return math.degrees(math.atan(math.sinh(n)))


def tile_to_bbox(tile_x: int, tile_y: int, zoom: int) -> TileBBox:
    lon_w = lon_from_tile_x(tile_x, zoom)
    lon_e = lon_from_tile_x(tile_x + 1, zoom)
    lat_n = lat_from_tile_y(tile_y, zoom)
    lat_s = lat_from_tile_y(tile_y + 1, zoom)
    return TileBBox(tile_x, tile_y, zoom, lat_s, lat_n, lon_w, lon_e)


def parse_tile_from_filename(sat_file: str) -> Tuple[int, int]:
    # sat_file format: tile_y_tile_x.png
    stem = Path(sat_file).stem
    parts = stem.split("_")
    if len(parts) != 2:
        raise ValueError(f"Unexpected sat_file format: {sat_file}")
    tile_y = int(parts[0])
    tile_x = int(parts[1])
    return tile_x, tile_y


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371008.8
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlmb / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def bbox_cover_radius_m(
    lat_s: float,
    lat_n: float,
    lon_w: float,
    lon_e: float,
    buffer_m: float = 40.0
) -> int:
    center_lat = (lat_s + lat_n) / 2.0
    center_lon = (lon_w + lon_e) / 2.0
    corners = [
        (lat_s, lon_w),
        (lat_s, lon_e),
        (lat_n, lon_w),
        (lat_n, lon_e),
    ]
    max_d = max(haversine_m(center_lat, center_lon, lat, lon) for lat, lon in corners)
    return int(math.ceil(max_d + buffer_m))


def tile_cover_radius_m(bbox: TileBBox, buffer_m: float = 80.0) -> int:
    return bbox_cover_radius_m(
        bbox.lat_s, bbox.lat_n, bbox.lon_w, bbox.lon_e, buffer_m=buffer_m
    )


def point_in_bbox(lat: float, lon: float, lat_s: float, lat_n: float, lon_w: float, lon_e: float, eps: float = 1e-9) -> bool:
    return (lat_s - eps) <= lat <= (lat_n + eps) and (lon_w - eps) <= lon <= (lon_e + eps)


def point_in_tile_bbox(lat: float, lon: float, bbox: TileBBox, eps: float = 1e-9) -> bool:
    return point_in_bbox(lat, lon, bbox.lat_s, bbox.lat_n, bbox.lon_w, bbox.lon_e, eps=eps)


def split_tile_bbox(tile_bbox: TileBBox, region_id: str, sub_rows: int, sub_cols: int) -> List[SubBBox]:
    if sub_rows < 1:
        raise ValueError("--sub-rows must be >= 1")
    if sub_cols < 1:
        raise ValueError("--sub-cols must be >= 1")

    lat_step = (tile_bbox.lat_n - tile_bbox.lat_s) / sub_rows
    lon_step = (tile_bbox.lon_e - tile_bbox.lon_w) / sub_cols

    sub_boxes: List[SubBBox] = []
    for r in range(sub_rows):
        for c in range(sub_cols):
            sub_lat_n = tile_bbox.lat_n - r * lat_step
            sub_lat_s = tile_bbox.lat_n - (r + 1) * lat_step
            sub_lon_w = tile_bbox.lon_w + c * lon_step
            sub_lon_e = tile_bbox.lon_w + (c + 1) * lon_step

            sub_id = f"r{r+1}c{c+1}"
            sub_boxes.append(
                SubBBox(
                    parent_region_id=region_id,
                    sub_id=sub_id,
                    row_idx=r + 1,
                    col_idx=c + 1,
                    rows=sub_rows,
                    cols=sub_cols,
                    lat_s=sub_lat_s,
                    lat_n=sub_lat_n,
                    lon_w=sub_lon_w,
                    lon_e=sub_lon_e,
                )
            )
    return sub_boxes


def normalize_input_tiles(df: pd.DataFrame) -> pd.DataFrame:
    required = {"city", "sat_file", "foursquare_exist"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV missing columns: {sorted(missing)}")

    out = df.copy()
    if "area" in out.columns:
        out["region_id"] = out["area"].fillna(out["sat_file"].map(lambda s: Path(s).stem))
    else:
        out["region_id"] = out["sat_file"].map(lambda s: Path(s).stem)

    out = out[["city", "sat_file", "region_id", "foursquare_exist"]].drop_duplicates()
    return out


def baidu_around_search(
    session: requests.Session,
    ak: str,
    query: str,
    center_lat: float,
    center_lon: float,
    radius_m: int,
    page_num: int,
    region_id: str,
    sub_id: str,
    page_size: int = 20,
    timeout: int = 20,
) -> Dict:
    params = {
        "query": query,
        "location": f"{center_lat},{center_lon}",
        "radius": radius_m,
        "radius_limit": "true",
        "ak": ak,
        "scope": 2,
        "coord_type": 1,
        "ret_coordtype": "gcj02ll",
        "extensions_adcode": "true",
        "page_num": page_num,
        "page_size": page_size,
        "output": "json",
    }
    r = session.get(BAIDU_AROUND_URL, params=params, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    save_api_response(data, region_id, query, page_num, sub_id=sub_id)
    return data


def extract_rows(
    resp: Dict,
    city: str,
    region_id: str,
    sat_file: str,
    tile_bbox: TileBBox,
    sub_bbox: SubBBox,
    query: str,
) -> Tuple[List[Dict], int]:
    rows: List[Dict] = []
    raw_api_count = 0

    for item in resp.get("results", []) or []:
        loc = item.get("location") or {}
        lat_gcj = loc.get("lat")
        lon_gcj = loc.get("lng")
        if lat_gcj is None or lon_gcj is None:
            continue

        raw_api_count += 1
        lon_wgs, lat_wgs = gcj02_to_wgs84(float(lon_gcj), float(lat_gcj))

        # 只按原始瓦片矩形筛，不按子矩形筛
        if not point_in_tile_bbox(lat_wgs, lon_wgs, tile_bbox):
            continue

        detail = item.get("detail_info") or {}
        category = (
            detail.get("classified_poi_tag")
            or detail.get("tag")
            or item.get("classified_poi_tag")
            or item.get("tag")
            or item.get("type")
            or query
        )

        rows.append({
            "city": city,
            "region_id": region_id,
            "sat_file": sat_file,

            "tile_x": tile_bbox.tile_x,
            "tile_y": tile_bbox.tile_y,
            "tile_zoom": tile_bbox.zoom,
            "tile_lat_s": tile_bbox.lat_s,
            "tile_lat_n": tile_bbox.lat_n,
            "tile_lon_w": tile_bbox.lon_w,
            "tile_lon_e": tile_bbox.lon_e,
            "tile_center_lat": tile_bbox.center_lat,
            "tile_center_lon": tile_bbox.center_lon,
            "tile_cover_radius_m": tile_cover_radius_m(tile_bbox),

            "tile_nw_lat": tile_bbox.corners["NW"][0],
            "tile_nw_lon": tile_bbox.corners["NW"][1],
            "tile_ne_lat": tile_bbox.corners["NE"][0],
            "tile_ne_lon": tile_bbox.corners["NE"][1],
            "tile_sw_lat": tile_bbox.corners["SW"][0],
            "tile_sw_lon": tile_bbox.corners["SW"][1],
            "tile_se_lat": tile_bbox.corners["SE"][0],
            "tile_se_lon": tile_bbox.corners["SE"][1],

            "sub_id": sub_bbox.sub_id,
            "sub_row": sub_bbox.row_idx,
            "sub_col": sub_bbox.col_idx,
            "sub_rows": sub_bbox.rows,
            "sub_cols": sub_bbox.cols,
            "sub_lat_s": sub_bbox.lat_s,
            "sub_lat_n": sub_bbox.lat_n,
            "sub_lon_w": sub_bbox.lon_w,
            "sub_lon_e": sub_bbox.lon_e,
            "sub_center_lat": sub_bbox.center_lat,
            "sub_center_lon": sub_bbox.center_lon,
            "sub_cover_radius_m": bbox_cover_radius_m(
                sub_bbox.lat_s, sub_bbox.lat_n, sub_bbox.lon_w, sub_bbox.lon_e, buffer_m=40.0
            ),

            "query": query,
            "uid": item.get("uid"),
            "name": item.get("name"),
            "latitude": lat_wgs,
            "longitude": lon_wgs,
            "latitude_gcj02": lat_gcj,
            "longitude_gcj02": lon_gcj,
            "category": category,
            "address": item.get("address"),
            "province": item.get("province"),
            "area": item.get("area"),
            "adcode": item.get("adcode"),
            "detail": item.get("detail"),
            "overall_rating": detail.get("overall_rating"),
            "price": detail.get("price"),
            "shop_hours": detail.get("shop_hours"),
            "brand": detail.get("brand"),
            "distance": detail.get("distance") if "distance" in detail else item.get("distance"),
            "source": "baidu_place_v3_around_gcj02_to_wgs84_then_tile_bbox_cut_after_subcircles",
        })
    return rows, raw_api_count


def deduplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    has_uid = df["uid"].notna() & (df["uid"].astype(str).str.len() > 0)
    with_uid = df[has_uid].drop_duplicates(subset=["region_id", "uid"])
    without_uid = df[~has_uid].copy()

    if not without_uid.empty:
        without_uid["lat_round"] = without_uid["latitude"].round(6)
        without_uid["lon_round"] = without_uid["longitude"].round(6)
        without_uid = without_uid.drop_duplicates(
            subset=["region_id", "name", "lat_round", "lon_round"]
        ).drop(columns=["lat_round", "lon_round"])

    return pd.concat([with_uid, without_uid], ignore_index=True)


def query_one_subcircle(
    session: requests.Session,
    ak: str,
    city: str,
    sat_file: str,
    region_id: str,
    tile_bbox: TileBBox,
    sub_bbox: SubBBox,
    query: str,
    sleep_sec: float,
    max_pages: int,
    page_size: int = 20,
) -> Tuple[List[Dict], int]:
    radius_m = bbox_cover_radius_m(
        sub_bbox.lat_s, sub_bbox.lat_n, sub_bbox.lon_w, sub_bbox.lon_e, buffer_m=40.0
    )

    all_rows: List[Dict] = []
    raw_total = 0

    first = baidu_around_search(
        session=session,
        ak=ak,
        query=query,
        center_lat=sub_bbox.center_lat,
        center_lon=sub_bbox.center_lon,
        radius_m=radius_m,
        page_num=0,
        region_id=region_id,
        sub_id=sub_bbox.sub_id,
        page_size=page_size,
    )
    status = first.get("status")
    if status != 0:
        raise RuntimeError(
            f"Baidu API error for {region_id}/{sub_bbox.sub_id} query={query}: "
            f"status={status}, message={first.get('message')}"
        )

    total = int(first.get("total", 0) or 0)
    rows, raw_count = extract_rows(first, city, region_id, sat_file, tile_bbox, sub_bbox, query)
    all_rows.extend(rows)
    raw_total += raw_count

    n_pages = min(max_pages, math.ceil(total / page_size)) if total else 0
    for page_num in range(1, n_pages):
        time.sleep(sleep_sec)
        resp = baidu_around_search(
            session=session,
            ak=ak,
            query=query,
            center_lat=sub_bbox.center_lat,
            center_lon=sub_bbox.center_lon,
            radius_m=radius_m,
            page_num=page_num,
            region_id=region_id,
            sub_id=sub_bbox.sub_id,
            page_size=page_size,
        )
        status = resp.get("status")
        if status != 0:
            raise RuntimeError(
                f"Baidu API error for {region_id}/{sub_bbox.sub_id} query={query} page={page_num}: "
                f"status={status}, message={resp.get('message')}"
            )
        rows, raw_count = extract_rows(resp, city, region_id, sat_file, tile_bbox, sub_bbox, query)
        all_rows.extend(rows)
        raw_total += raw_count

    time.sleep(sleep_sec)
    return all_rows, raw_total


def query_tile(
    session: requests.Session,
    ak: str,
    city: str,
    sat_file: str,
    region_id: str,
    zoom: int,
    queries: List[str],
    sleep_sec: float,
    max_pages: int,
    sub_rows: int,
    sub_cols: int,
) -> Tuple[List[Dict], int]:
    tile_x, tile_y = parse_tile_from_filename(sat_file)
    tile_bbox = tile_to_bbox(tile_x, tile_y, zoom)
    sub_boxes = split_tile_bbox(tile_bbox, region_id, sub_rows=sub_rows, sub_cols=sub_cols)

    all_rows: List[Dict] = []
    raw_total = 0

    for query in queries:
        for sub_bbox in sub_boxes:
            sub_rows_data, sub_raw_total = query_one_subcircle(
                session=session,
                ak=ak,
                city=city,
                sat_file=sat_file,
                region_id=region_id,
                tile_bbox=tile_bbox,
                sub_bbox=sub_bbox,
                query=query,
                sleep_sec=sleep_sec,
                max_pages=max_pages,
            )
            all_rows.extend(sub_rows_data)
            raw_total += sub_raw_total

    return all_rows, raw_total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Beijing/Shanghai tile CSV")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--ak", required=True, help="Baidu Map API AK")
    parser.add_argument("--zoom", type=int, default=15, help="Tile zoom level, default=15")
    parser.add_argument(
        "--queries",
        type=str,
        default=",".join(DEFAULT_QUERIES),
        help="Comma-separated query keywords"
    )
    parser.add_argument(
        "--only-foursquare-no",
        action="store_true",
        help="Only keep rows where foursquare_exist == 'no'"
    )
    parser.add_argument("--limit", type=int, default=None, help="Only process first N unique tiles")
    parser.add_argument("--sleep-sec", type=float, default=0.2, help="Delay between requests")
    parser.add_argument("--max-pages", type=int, default=8, help="Safety cap for pagination loops")

    # 用户自定义拆分：任意正整数行列
    parser.add_argument("--sub-rows", type=int, default=3, help="Split each tile into N rows (>=1)")
    parser.add_argument("--sub-cols", type=int, default=3, help="Split each tile into N cols (>=1)")

    args = parser.parse_args()

    if args.sub_rows < 1:
        raise ValueError("--sub-rows must be >= 1")
    if args.sub_cols < 1:
        raise ValueError("--sub-cols must be >= 1")

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)
    tiles = normalize_input_tiles(df)

    if args.only_foursquare_no:
        tiles = tiles[tiles["foursquare_exist"].astype(str).str.lower() == "no"].copy()

    tiles = tiles.sort_values(["city", "region_id"]).reset_index(drop=True)

    if args.limit is not None:
        tiles = tiles.head(args.limit).copy()

    queries = [q.strip() for q in args.queries.split(",") if q.strip()]
    if not queries:
        raise ValueError("No valid queries provided.")

    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 baidu-poi-tile-script/2.0"})

    rows: List[Dict] = []
    errors: List[Dict] = []

    for i, rec in tiles.iterrows():
        city = str(rec["city"])
        sat_file = str(rec["sat_file"])
        region_id = str(rec["region_id"])

        try:
            tile_rows, raw_total = query_tile(
                session=session,
                ak=args.ak,
                city=city,
                sat_file=sat_file,
                region_id=region_id,
                zoom=args.zoom,
                queries=queries,
                sleep_sec=args.sleep_sec,
                max_pages=args.max_pages,
                sub_rows=args.sub_rows,
                sub_cols=args.sub_cols,
            )
            rows.extend(tile_rows)
            print(
                f"[{i+1}/{len(tiles)}] OK {region_id}: "
                f"api_raw={raw_total}, kept_after_tile_bbox_cut={len(tile_rows)}, "
                f"subcircles={args.sub_rows * args.sub_cols} ({args.sub_rows}x{args.sub_cols})"
            )
        except Exception as e:
            errors.append({
                "city": city,
                "sat_file": sat_file,
                "region_id": region_id,
                "error": str(e),
            })
            print(f"[{i+1}/{len(tiles)}] ERROR {region_id}: {e}", file=sys.stderr)

    out_df = pd.DataFrame(rows)
    if out_df.empty:
        print("No rows kept. Check AK / zoom / tile parsing / query settings / API limit.")
        out_df.to_csv(out_path, index=False)
    else:
        out_df = deduplicate_rows(out_df)
        front = [
            "city", "region_id", "name", "latitude", "longitude", "category",
            "sat_file", "tile_x", "tile_y", "tile_zoom",
            "tile_lat_s", "tile_lat_n", "tile_lon_w", "tile_lon_e",
            "tile_nw_lat", "tile_nw_lon", "tile_ne_lat", "tile_ne_lon",
            "tile_sw_lat", "tile_sw_lon", "tile_se_lat", "tile_se_lon",
            "tile_center_lat", "tile_center_lon", "tile_cover_radius_m",
            "sub_id", "sub_row", "sub_col", "sub_rows", "sub_cols",
            "sub_lat_s", "sub_lat_n", "sub_lon_w", "sub_lon_e",
            "sub_center_lat", "sub_center_lon", "sub_cover_radius_m",
            "latitude_gcj02", "longitude_gcj02",
            "uid", "query", "address", "province", "area", "adcode", "source"
        ]
        remaining = [c for c in out_df.columns if c not in front]
        out_df = out_df[front + remaining]
        out_df.to_csv(out_path, index=False, quoting=csv.QUOTE_MINIMAL)
        print(f"Saved {len(out_df)} deduplicated rows to {out_path}")

    if errors:
        err_path = out_path.with_suffix(".errors.csv")
        pd.DataFrame(errors).to_csv(err_path, index=False)
        print(f"Saved {len(errors)} errors to {err_path}")


if __name__ == "__main__":
    main()