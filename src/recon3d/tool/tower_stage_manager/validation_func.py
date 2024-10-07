import yaml



def yaml_load(fp):
    with open(fp, 'r') as f:
        yaml_dict = yaml.safe_load(f)
    return yaml_dict

def get_src_marker_info(src_marker):
        """
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

        """
        src_dict = {}
        for entry in src_marker:
            name = entry['name']
            if (name.startswith("start_end")):
                src_dict['ss'] = float(entry['begin'])/1000
                src_dict['to'] = float(entry['end'])/1000
 
            elif(name.startswith("node")):
                node_id = entry['name'].split('_')[1]
                node_time_stamp = float(entry['begin'])/1000
                src_dict[node_id] = node_time_stamp
        
        node_time_list = []
        for key in src_dict.keys():
            if key == 'ss':
                start_time = src_dict[key]
            elif key == 'to':
                end_time = src_dict[key]
            else:
                node_time_list.append(src_dict[key])
 
        return start_time, end_time, node_time_list


def check_src_marker(src_mrk_file):
    """
    check if node_time exceeds start_time or end_time
    """
    src_marker = yaml_load(src_mrk_file)
    start_time, end_time, node_time_list = get_src_marker_info(src_marker)
    # check if node time exceeds ss and to
    if any(node_time < start_time for node_time in node_time_list) or \
            any(node_time > end_time for node_time in node_time_list):
        return False
    else: 
        return True


VALIDATION_DICT = {'marker': check_src_marker}
