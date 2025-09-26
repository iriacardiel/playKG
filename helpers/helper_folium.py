import folium

def create_map_from_rows(filename:str="",rows:list=[], center_coordinates:list=[]):
    m = folium.Map(location=center_coordinates, zoom_start=12)
    for r in rows:
        folium.Marker([r.get("lat",0.0), r.get("lon",0.0)],
                        popup=str({k: v for k, v in r.items()})).add_to(m)
    
    m.save(filename)
    print(f"Map saved to {filename}")