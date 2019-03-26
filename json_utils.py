import os
import json

# for utterly stupid json serialization not working for int32 numpy datatype.. goddam python!
class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)

def ParseJsonFile(filepath):
    try:
        with open(filepath) as json_file:
            data = json.load(json_file)
        return data    
    except FileNotFoundError:
        print("File Not found while loading Json file {}".format(filepath))


def WriteJsonFile(filepath, json_data):
    try:
        with open(filepath,'w') as json_file:
            json.dump(json_data,json_file)
    except IOError:
        print("IO Error Writing Json file {}".format(filepath))

def IsJsonValueTrue(json,valuename):
    if valuename in json:
        if json[valuename] == "True":
            return True
        else:
            return False
    else:
        return False