import folium

def create_map_from_rows(rows:list=[]):
    m = folium.Map(location=[40.4168, -3.7038], zoom_start=12)
    filename= "data/friends/friends_map_folium.html"
    for r in rows:
        folium.Marker([r.get("lat",0.0), r.get("lon",0.0)],
                        popup=f"{r.get("name","")} - {r.get("labels",[])}").add_to(m)
    
    m.save(filename)
    print(f"Map saved to {filename}")