import networkx as nx
import json
import os
import glob
from tool.tree_to_nested_dict import Tree2NestedDict

class CrossNodeSIDPair():
    '''
    Keep Info from intersection node SP between two routes
    Args:
    node_ID: int 
        the node SP ID of intersection between two routes 
    sliceID_parent: int
        the sliceID corresponding to node SP ID from parent route
    sliceID_child: int
        the sliceID correspongding to node SP ID from child route
    '''
    def __init__(self, node_ID, SID_parent, SID_child):
        self.node_ID = node_ID
        self.sliceID_parent = SID_parent
        self.sliceID_child = SID_child
    def get_sliceID_parent(self):
        return self.sliceID_parent
    def get_sliceID_child(self):
        return self.sliceID_child

class NodeID2SliceID():
    '''
    keep Info of node SP ID and its corresponding slice ID
    on a single route
    nodeID2sliceID: {'node_ID_1': slice_ID_1,
                     'node_ID_2': slice_ID_2,
                     'node_ID_3': slice_ID_3...}
    '''
    def __init__(self):
        self.nodeID2sliceID = {}
    def add_node_slice_ID_pair(self,node_ID, slice_ID):
        self.nodeID2sliceID[node_ID] = slice_ID
    def get_node_ID_set(self):
        return set(self.nodeID2sliceID.keys())
    def fetch_slice_ID(self,node_ID):
        return self.nodeID2sliceID[node_ID]

def load_src_markers(src_mrk_path):
    '''
    The nameing of source markers is {route_name}_source_marker.json
    Args: path of source markers 
    Return: 
    src_marker_data_list: list of data read from source markers
    route_name_list: list of route_name parsed from file name of source markers
    '''
    src_marker_list = sorted(glob.glob(src_mrk_path+"/*json"))
    src_marker_data_list = []
    route_name_list = []
    for src_marker in src_marker_list:
        filename = os.path.basename(src_marker)
        route_name = filename.replace("_source_marker.json","")
        with open(src_marker, 'r') as f:
            src_marker_data = json.load(f)
            src_marker_data_list.append(src_marker_data)
            route_name_list.append(route_name)
    return src_marker_data_list, route_name_list

def gen_NID2SID(src_marker_data_list, slice_time_list):
    '''
    src_marker is a json file marks the timestamp in millisec of its corresponding video
    example of source marker data:

    [ {"name":"start_end","begin":5595,"end":17413},
      {"name":"node_39","begin":5595},
      {"name":"node_40","begin":17413}
    ]

    from the marked time, and slice_time (the time a slice covers(in second), ex. 15 secs)
    one can convert the relation of  (nodeID : video_time) to (nodeID : sliceID)
    Args: 
    list of source marker data
    list of route name 
    list of slice time: slice time is the time interval one slice covers (ex. 15 secs) 
    Return:
    route_src_path_list: list of route source path, the path one can find info of SfM route
    nid2sid_list : list of NID2SID object
    '''
    route_src_path_list = []
    nid2sid_list = []
    for src_marker_data, slice_time in zip(src_marker_data_list, slice_time_list): 
        node_id_list = []
        node_time = []
        slice_id_list = []
        for entry in src_marker_data:
            name = entry['name']
            if (name.startswith('start_end')):
                time_begin = int(entry['begin'])
            if (name.startswith('node')):
                node_id_list.append(entry['name'].split('_')[1])
                node_time.append(entry['begin'])

        for nt, nid in zip(node_time, node_id_list):
            node_time_stamp = nt - time_begin
            slice_id = int(node_time_stamp / slice_time / 1e3)
            slice_id_list.append(slice_id)

        nid2sid = NodeID2SliceID()
        for nodeID, sliceID in zip(node_id_list, slice_id_list):
            nid2sid.add_node_slice_ID_pair(nodeID, sliceID) 
        nid2sid_list.append(nid2sid)
    return nid2sid_list


class TreeGenerator():
    '''
    Given the NodeID2SliceID on each route within a stage, proceed the steps: 
    1: find the graph that prepresent the connection between the routes
    2: detect the nodes with highest degrees (with max number of outward edges) as root 
    3: use root to expand the BFS tree
    4: export the tree with the structure as nested dictionary
    The desired output is a nested dictionary contains SubTreeUnit
    For a single route with src at repo_src_path/repo_name, the item in dictionary look like:
    { 
        "repo_src_path/repo_name": {
            'parent_to_own_slice': ['003','000'],
            'child_node': []
            } 
    }

    parent_to_own_slice:
        [SliceID_parent, SliceID_child] on a cross node SP between parent route and child route.
    child_node: list of child routes
    
    For example: 
    { 
        "repo_src_path/repo_name": {
            'parent_to_own_slice': ['003','000'],
            'child_node': [    
                {
                    "repo_src_path/repo_name": {
                    'parent_to_own_slice': ['002','001'],
                    'child_node': []
                    }
                },
                {
                    "repo_src_path/repo_name": {
                    'parent_to_own_slice': ['000','000'],
                    'child_node': []
                    },
                }    
            ]

        }
    }
 
    Args:
    stage_routes_to_NID2SID: a dictionary maps routes in stage to their corresponding NodeID2SliceID object
    '''
    def __init__(self, route_src_path_list, NID2SID_list):
        self.stage_routes_to_NID2SID = self.gen_stage_routes_to_NID2SID(route_src_path_list, NID2SID_list)
        self.G = self.create_graph()
        self.root_nodes = self.find_root_nodes()
        self.max_root = len(self.root_nodes)
        self.root_node_counter = 0

    def get_graph(self):
        return self.G
    
    def get_root_node(self):
        '''
        whenever this function get called, it will return a element 
        from self.root_nodes according to the value of counter.
        counter get incremented by 1 whenever the function is calles.
        If the number of calls exceeds the lenght of self.root_nodes, 
        it will retrun None
        
        user should not give the defalut variable counter anyvalue, 
        it will be incremented by 1 everytime this function gets called.
        Return:
            root_node: root_node in self.G
            conter: just for the use in this function, not a real return
        
        '''
        if self.root_node_counter < self.max_root:
            root_node = self.root_nodes[self.root_node_counter]
            self.root_node_counter += 1
            return root_node
        else:
            print('exceed maximun')
            return None

    def gen_stage_routes_to_NID2SID(self, route_src_path_list, NID2SID_list):
        stage_routes_to_NID2SID = {}
        for route_src_path, NID2SID in zip(route_src_path_list, NID2SID_list):
            stage_routes_to_NID2SID[route_src_path] = NID2SID 
        return stage_routes_to_NID2SID

    def create_graph(self):
        '''
        assign node G to graph
        node name: path of the tower src repo
        node attribute: NodeID2SliceID  
        Return:
            networkx.MultiDiGraph()
        '''
        G = nx.MultiDiGraph()
        for route, nid2sid in self.stage_routes_to_NID2SID.items():
            G.add_node(route, NID2SID = nid2sid)
        self.assign_edges(G)
        return G

    def assign_edges(self, G):
        '''
        Loop through nodes in G, detect if there is intersection with other nodes.
        If there is intersection, assign edges between these two nodes in G.
        edge direction: parent to child
        edge attribute: CrossNodeSIDPair
        '''
        routes_to_NID2SID = nx.get_node_attributes(G, 'NID2SID')
        for idx1, route_1 in enumerate(routes_to_NID2SID.keys()):
            for idx2, route_2 in enumerate(routes_to_NID2SID.keys()):
                if idx2 <= idx1: 
                    continue
                NID2SID_1 = routes_to_NID2SID[route_1]
                NID2SID_2 = routes_to_NID2SID[route_2]
                node_id_set_1 =  NID2SID_1.get_node_ID_set()
                node_id_set_2 =  NID2SID_2.get_node_ID_set()
                cross_nodes = node_id_set_1.intersection(node_id_set_2)
                if len(cross_nodes) == 0:
                    continue
                for cross_node in sorted(cross_nodes):
                    SID_1 = NID2SID_1.fetch_slice_ID(cross_node) 
                    SID_2 = NID2SID_2.fetch_slice_ID(cross_node) 
                    cross_node_SID_pair_1_2 = CrossNodeSIDPair(cross_node, SID_1, SID_2)
                    cross_node_SID_pair_2_1 = CrossNodeSIDPair(cross_node, SID_2, SID_1)
                    G.add_edge(route_1, route_2, cross_node_SID_pair = cross_node_SID_pair_1_2)
                    G.add_edge(route_2, route_1, cross_node_SID_pair = cross_node_SID_pair_2_1)

    def find_root_nodes(self):
        '''
        Detect the node in self.G with maximum outward edges and report the node list

        Args: 
            networkx.MultiDiGraph()
        Return:
            list of nodes in networkx.MultiDiGraph() with maximum outward edges
        '''
        nodes = []
        degrees = []
        for x in self.G.nodes():
            nodes.append(x)
            degrees.append(self.G.out_degree(x))
        max_degree = max(degrees)
        root_nodes = []
        for node, degree in zip(nodes, degrees):
            if degree < max_degree:
                continue
            root_nodes.append(node)
        return root_nodes

    def get_BFS_tree(self):
        '''
        Expand BFS tree from root_node in G.
        Export different tree each time called.
        Retrun None if no further BFS tree can be exported from TreeGenerator
        Args:
            G: networkx.MultiDiGraph()
        Return:
            list of tuples(parent_node, child_node) that describes a tree
            Example:
            [(A,B),(B,C),(B,D),(D,E)]
                      A
                      |
                      B
                     / \  
                    C   D      
                        |
                        E    
            root_node: node in networkx.MultiDiGraph()
        '''
        root_node  = self.get_root_node()
        if root_node == None:
            return None, None
        else:
            return list(nx.bfs_edges(self.G, root_node)), root_node
