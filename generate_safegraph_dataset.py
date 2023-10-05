import pandas as pd
import numpy as np
import json
import os
import time
from tqdm import tqdm
from haversine import haversine


CITY = 'Los Angeles'
CACHED_BYSGNN_DATA_PATH = './data/data-0-400-400-Los Angeles.pkl'
num_nearest_neighbors = 300
output_path = './data/safegraph_neighborhood_data_{}_{}.json'.format(CITY, num_nearest_neighbors)


class stat_collector:
    def __init__(self):
        self.parquet_file_count=0
        self.data_record_count = 0
        self.memory_usage_in_GB = 0		#gives an estimate of the total RAM usage if all files were read into memory at the same time.
        self.unique_device_count = 0
        self.avg_pos_acc = 0
        self.starting_time = time.process_time()
        self.elapsed_time = time.process_time()
        self.unique_geohash_count = 0

def load_poi_db(city):
    poi_folder = "/storage/dataset/poi_haowen/CoreRecords-CORE_POI-2019_03-2020-03-25/"
    poi_columns = ["safegraph_place_id", "parent_safegraph_place_id", "location_name", "safegraph_brand_ids", "brands",
                   "top_category", "sub_category", "naics_code", "latitude", "longitude", "street_address", "city",
                   "region", "postal_code", "iso_country_code", "phone_number", "open_hours", "category_tags"]
    files = os.listdir(poi_folder)


    poi_s = stat_collector()
    poi_db = pd.DataFrame(columns=poi_columns)
    for f in files:
        if f[-3:] == 'csv' and 'brand' not in f:
            print(f)
            df = pd.read_csv(poi_folder + f)
            df = df.loc[df['city']==city]
            poi_db = pd.concat([poi_db, df], ignore_index=True, sort=False)
            poi_s.memory_usage_in_GB += df.memory_usage(deep=True).sum() / 1000000000
            poi_s.data_record_count += df.shape[0]
            poi_s.parquet_file_count += 1
    return poi_db, poi_s

def nearest_neighbors(df, lat, lon, n=300):
    # Calculate the distance to every other POI
    df['distance'] = df.apply(lambda row: haversine((lat, lon), (row['latitude'], row['longitude'])), axis=1)
    
    # Sort dataframe by distance and take the top n rows
    neighbors = df.sort_values(by='distance').iloc[1:n+1]
    
    return neighbors

poi_db, poi_s = load_poi_db(CITY)
cached_bysgnn_data = pd.read_pickle(CACHED_BYSGNN_DATA_PATH)

# keep the rows from poi_db that are in cached_bysgnn_data in a new dataframe
target_poi_db = poi_db.loc[poi_db['safegraph_place_id'].isin(cached_bysgnn_data['safegraph_place_id'])]


result = []

for index, row in tqdm(target_poi_db.iterrows(), total=target_poi_db.shape[0]):
    neighbors = nearest_neighbors(poi_db, row['latitude'], row['longitude'], n=num_nearest_neighbors)
    
    poi_json = {
        "info": {
            "name": row['location_name'],
            "geometry": {
                "coordinates": [row['longitude'], row['latitude']]
            },
            'safegraph_place_id': row['safegraph_place_id'],
        },
        "neighbor_info": {
            "name_list": neighbors['location_name'].tolist(),
            "geometry_list": [{"coordinates": [lon, lat]} for lat, lon in zip(neighbors['latitude'], neighbors['longitude'])]
        }
    }
    
    result.append(poi_json)
result_str = '\n'.join(json.dumps(j) for j in result)

output_path = './data/safegraph_neighborhood_data_{}_{}.json'.format(CITY, num_nearest_neighbors)
# save the result_str to output_path
with open(output_path, 'w') as f:
    f.write(result_str)