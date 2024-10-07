import os
import json

def read_src_marker(src_marker_file):
    '''
    read src_maker_file and return its data

    src_marker is a json file marks the timestamp in millisec of its corresponding video
    example of source marker data:

    [ {"name":"start_end","begin":5595,"end":17413},
      {"name":"node_39","begin":5595},
      {"name":"node_40","begin":17413}
    ]

    Args: name of source marker file
    Return: dictionary of source marker data

    '''
    with open(src_marker_file, 'r') as f:
        data_src_marker = json.load(f)
        return data_src_marker
    
def get_src_marker_info(src_marker):
    '''
    parse source maker data stored in src_dict
    src_marker is a json file marks the timestamp in millisec of its corresponding video
    example of source marker data:

    [ {"name":"start_end","begin":5595,"end":17413},
      {"name":"node_39","begin":5595},
      {"name":"node_40","begin":17413}
    ]
    stored the information parsed from source markder and stored 
    in src_dict with entries:
    there can be multiple entries of nodeID corresponding to different nodeIDs
    {
        'ss': start time in second
        'to': end time in second
        'sodeID': node time stamp in second
    }
    example: 
    {
        'ss': 2.33
        'to': 103.41
        '1': 2.51
        '10': 3.33
        '28': 99.12
    }

    Args: 
    src_marker:  dict that contains information from source marker
    Return: src_dict 

    '''
    src_dict = {}
    for entry in src_marker:
        name = entry['name']
        if (name.startswith("start_end")):
            src_dict['ss'] = int(entry['begin'])/1000
            src_dict['to'] = int(entry['end'])/1000

        elif(name.startswith("node")):
            node_id = entry['name'].split('_')[1]
            node_time_stamp = int(entry['begin'])/1000
            src_dict[node_id] = node_time_stamp
    return src_dict


