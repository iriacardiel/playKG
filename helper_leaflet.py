"""
Neo4j -> Leaflet map (no APOC needed)

- Connects to Neo4j
- Reads nodes with `n.location` (point in WGS-84)
- Generates a GeoJSON FeatureCollection
- Writes a standalone Leaflet HTML file: neo4j_locations_map.html
"""

from neo4j import GraphDatabase
import json
from pathlib import Path
import os
from textwrap import dedent
from dotenv import load_dotenv  


load_dotenv()  # Load local environment variables


# -----------------------------
# 1) CONFIG
# -----------------------------
URI = "bolt://localhost:" + os.environ.get("URI_PORT")
NEO4J_USER = os.environ.get("NEO4J_USER")
NEO4J_PWD = os.environ.get("NEO4J_PASSWORD")
NEO4J_DB = os.getenv("NEO4J_DATABASE", "neo4j")    # 👈 choose DB here
# Optional: default map center if fitting bounds fails (e.g., no points)
DEFAULT_CENTER = (40.4168, -3.7038)  # Madrid
DEFAULT_ZOOM = 12

def rows_to_geojson(rows):
    """
    Convert query rows to a GeoJSON FeatureCollection (as a Python dict).
    """
    features = []
    for row in rows:
        labels = row["labels"] or []
        name = row["name"] or "Unknown"
        lon, lat =row["lon"], row["lat"]

        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [lon, lat],  # GeoJSON expects [lon, lat]
            },
            "properties": {
                "name": name,
                "labels": labels,
            },
        })

    return {
        "type": "FeatureCollection",
        "features": features,
    }


def build_leaflet_html(geojson_str):
    """
    Returns a complete, standalone HTML document string embedding the GeoJSON.
    """
    # Escape only for closing </script> edge case
    # (Leaflet data is JSON; we embed as a JS string then JSON.parse)
    geojson_js = geojson_str.replace("</script>", "<\\/script>")

    html_doc = f"""\
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Neo4j Locations</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">

  <!-- Leaflet CSS -->
  <link
    rel="stylesheet"
    href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
    integrity="sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY="
    crossorigin=""
  />
  <style>
    html, body, #map {{ height: 100%; margin: 0; }}
    .popup-name {{ font-weight: 600; margin-bottom: 4px; }}
    .legend {{
      position: absolute;
      z-index: 1000;
      bottom: 16px; left: 16px;
      background: white; padding: 8px 10px; border-radius: 8px;
      box-shadow: 0 2px 10px rgba(0,0,0,.15);
      font: 14px/1.2 system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
    }}
    .legend span.badge {{
      display: inline-block; padding: 2px 6px; margin-right: 6px; border-radius: 999px; background: #eee;
    }}
  </style>
</head>
<body>
  <div id="map"></div>

  <!-- Leaflet JS -->
  <script
    src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"
    integrity="sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo="
    crossorigin="">
  </script>

  <script>
    // GeoJSON exported from Neo4j (as a JSON string to keep the file standalone)
    const geojson = `{geojson_js}`;
    const data = JSON.parse(geojson);

    // Build map
    const map = L.map('map').setView([{DEFAULT_CENTER[0]}, {DEFAULT_CENTER[1]}], {DEFAULT_ZOOM});

    // OSM tiles
    L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
      maxZoom: 19,
      attribution: '&copy; OpenStreetMap contributors'
    }}).addTo(map);

    // Simple icon color by label (first label wins). You can refine this.
    const labelColors = {{
      Person: '#2e7d32',
      Company: '#1565c0'
    }};

    function circleMarker(feature, latlng) {{
      const labels = (feature.properties && feature.properties.labels) || [];
      const primary = labels.length ? labels[0] : 'Other';
      const color = labelColors[primary] || '#6d4c41';
      return L.circleMarker(latlng, {{
        radius: 7,
        weight: 2,
        opacity: 1,
        color: color,
        fillOpacity: 0.15,
        fillColor: color
      }});
    }}

    const layer = L.geoJSON(data, {{
      pointToLayer: circleMarker,
      onEachFeature: (feature, layer) => {{
        const p = feature.properties || {{}};
        const name = p.name || 'Unknown';
        const labels = (p.labels || []).join(', ');
        layer.bindPopup(`<div class="popup-name">${{name}}</div>${{labels}}`);
      }}
    }}).addTo(map);

    // Fit bounds if we have features
    try {{
      const bounds = layer.getBounds();
      if (bounds.isValid()) {{
        map.fitBounds(bounds, {{ padding: [24, 24] }});
      }}
    }} catch (e) {{
      // Fall back to default center/zoom
    }}

    // Tiny legend
    const legend = L.control({{position: 'bottomleft'}});
    legend.onAdd = function() {{
      const div = L.DomUtil.create('div', 'legend');
      div.innerHTML = `
        <div><span class="badge" style="background:#2e7d32;"></span>Person</div>
        <div><span class="badge" style="background:#1565c0;"></span>Company</div>
      `;
      return div;
    }};
    legend.addTo(map);
  </script>
</body>
</html>
"""
    return html_doc


def create_map_from_rows(rows:list=[]):
  filename = "neo4j_locations_map_leaflet.html"
  fc = rows_to_geojson(rows) # Convert to GeoJSON (Python dict -> JSON string)
  geojson_str = json.dumps(fc, ensure_ascii=False)
  html_out = build_leaflet_html(geojson_str) # Build Leaflet HTML
  Path(filename).write_text(html_out, encoding="utf-8")
  print(f"Map saved to {filename}")