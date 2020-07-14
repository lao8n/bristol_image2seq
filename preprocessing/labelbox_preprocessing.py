import json
import datetime as dt
from shapely import wkt 
import pandas as pd 

def from_json(labeled_data):
  # read labelbox JSON output 
  with open(labeled_data, 'r') as f:
    label_data = json.loads(f.read())

  # create pd dataframe
  columns = ['filepath', 'x1', 'y1', 'x2', 'y2', 'class_name']
  df = pd.DataFrame(columns=columns)

  for data in label_data:
    labels = data['Label']
    if not callable(getattr(labels, 'keys', None)):
      continue
    
    for category_name, list_dicts_xy in labels.items():
      for xy_geom in list_dicts_xy:
        xy = xy_geom['geometry']
        print(xy)
        x_list = [xy[0]['x'], xy[1]['x'], xy[2]['x'], xy[3]['x']]
        y_list = [xy[0]['y'], xy[1]['y'], xy[2]['y'], xy[3]['y']]
        min_x = min(x_list)
        min_y = min(y_list)
        max_x = max(x_list)
        max_y = max(y_list)
        print("min {},{}".format(min_x, min_y))
        print("max {},{}".format(max_x, max_y))

        df = df.append({
          'filepath': 'exam_script_data/' + data['External ID'], 
          'x1': int(min_x),
          'y1': int(min_y),
          'x2': int(max_x),
          'y2': int(max_y),
          'class_name': category_name
        }, ignore_index = True)
  
  return df

if __name__ == '__main__':
  df = from_json('exam_script_data/labelbox.json')
  df.to_csv('exam_script_data/labelbox.txt', header=None, index=False)