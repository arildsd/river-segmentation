import json


def get_rivers(geojson_dict):
    rivers = []
    for feature in geojson_dict["features"]:
        properties = feature["properties"]
        geometry = feature["geometry"]
        if geometry["type"] == "MultiPolygon":
            if "water" in properties:
                if "river" == properties["water"]:
                    rivers.append(feature)
    return rivers


def get_sand(geojson_dict):
    sands = []
    for feature in geojson_dict["features"]:
        properties = feature["properties"]
        geometry = feature["geometry"]
        if geometry["type"] == "MultiPolygon":
            if "natural" in properties:
                if "sand" == properties["natural"]:
                    sands.append(feature)
    return sands


if __name__ == '__main__':
    FILE_PATH = r"../data/OSM/planet_9.533,62.856_11.384,63.381.osm.geojson"
    file_object = open(FILE_PATH, "r")
    geo_dict = json.load(file_object)
    file_object.close()

    sand = get_sand(geo_dict)
    sand_dict = {"type": "FeatureCollection", "features": sand}
    sand_1965 = []
    for s in sand:
        print(s["properties"])
        if "source:date" in s["properties"]:
            if "1965" in s["properties"]["source:date"][:4]:
                sand_1965.append(s)
    sand_1965_writer = open(r"../data/sand_1965_gaula++.geojson", "w+")
    json.dump({"type": "FeatureCollection", "features": sand_1965},
              sand_1965_writer)
    sand_1965_writer.close()

    #sand_writer = open(r"../data/sand_gaula++.geojson", "w+")
    #json.dump(sand_dict, sand_writer)
    #sand_writer.close()


    #rivers = get_rivers(geo_dict)
    #rivers_writer = open(r"../data/rivers_gaula++.geojson", "w+")
    #rivers_dict = {"type": "FeatureCollection", "features": rivers}
    #json.dump(rivers_dict, rivers_writer)

# prop: 'natural': 'water'
# prop: 'water': 'river'