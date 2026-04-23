# Baidu POI Tile Crawler

This project provides a scalable pipeline to collect **Points of Interest (POIs)** from the **Baidu Maps Place API** using a **tile-based spatial querying strategy**.

It is designed for **large-scale geospatial data collection**, especially for research tasks such as:

- Urban representation learning
- Recommender systems with geographic context
- Spatial data mining

---

## Features

- Tile-based querying using Web Mercator tiles
- Multi-query POI retrieval with customizable categories
- Automatic subdivision of tiles to improve coverage under API limits
- Coordinate conversion from GCJ-02 to WGS84
- Bounding-box filtering to remove out-of-range points
- Deduplication by UID or spatial proximity
- API response caching for debugging and reproducibility

---

## Project Structure


```
├── main.py
├── api_cache/
│   ├── region_id__sub_id__query__pageN.json
│   ├── ...
│   └── ...
├── input.csv
└── output.csv
```

---

## Input Format

The input CSV must contain the following columns:

| Column | Description |
|--------|-------------|
| city | City name |
| sat_file | Tile filename in the format tile_y_tile_x.png |
| foursquare_exist | Flag used for filtering |

Example:

city,sat_file,foursquare_exist
Beijing,12394_26972.png,no

---

## Usage

Basic example:

python Tile2POI.py \
  --input input.csv \
  --output output.csv \
  --ak YOUR_BAIDU_API_KEY \
  --zoom 15

---

## Optional Arguments

--queries: Comma-separated POI category queries  
--sub-rows: Number of row splits per tile  
--sub-cols: Number of column splits per tile  
--max-pages: Maximum number of pages per query  
--sleep-sec: Delay between API calls  
--limit: Limit the number of tiles processed  
--only-foursquare-no: Only process rows where foursquare_exist == no  

---

## How It Works

1. Tile is converted to bounding box  
2. Tile is split into subregions  
3. Each subregion is queried using circular API search  
4. Coordinates converted from GCJ-02 to WGS84  
5. Points filtered by tile bounding box  
6. Results deduplicated  

---

## Output

The output CSV includes POI information, coordinates, tile metadata, subregion metadata, and query information.

---

## License

MIT License

---

## Author

David Feng  
University of Minnesota – Twin Cities
