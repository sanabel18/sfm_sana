
class SubTreeUnit():
    '''
    Args:
    SliceID_parent: int
    SliceID_child: int
    '''
    def __init__(self, SliceID_parent, SliceID_child):
        slice_ID_pair_str = [str(SliceID_parent).zfill(3),
                             str(SliceID_child).zfill(3)]
        self.sub_tree_unit = {
            'parent_to_own_slice': slice_ID_pair_str,
            'child_node': None
        }         
    def get_subtree_unit(self):
        return self.sub_tree_unit

class Tree2NestedDict():
    def __init__(self, G, bfs_tree, key_root):
        subtrees = self.gen_subtree_dict(G, bfs_tree)
        self.nested_dict = self.assemble_subtrees_to_nested_dict( key_root, bfs_tree, subtrees)
    
    def get_nested_dict(self):
        return self.nested_dict
    
    def gen_subtree_dict(self, G, bfs_tree):
        '''
        Generate dict of SubtreeUnit from bfs_tree tuples:
        bfs_tree is constructed with (parent_node, child_node) tuples
        bfs_tree example: 
            [(A,B),(B,C),(B,D),(D,E)]
                      A
                      |
                      B
                     / \  
                    C   D      
                        |
                        E       
        Args:
            G: networkx.MultiDiGraph()
            bfs_tree: list of tuples of nodes in G
        Return:
            dict of SubtreeUnit with child node as key
        '''
        subtrees = {}
        for edge in bfs_tree:
            parent = edge[0]
            child = edge[1]
            slice_ID_pair = G[parent][child][0]['cross_node_SID_pair']
            subtrees[child] = SubTreeUnit(slice_ID_pair.sliceID_parent, slice_ID_pair.sliceID_child) 
 
        return subtrees

    def assemble_subtrees_to_nested_dict(self, key_root, bfs_tree, subtrees):
        '''
        Assamble subtree unit to a full tree according to their relations in bfs_tree
        Args: 
        key_root: str
            path of the tower src repo of the root node of the tree
        bfs_tree: list of tuples 
            list of tuples of node in G that describes a tree
            bfs_tree example: 
                [(A,B),(B,C),(B,D),(D,E)]
        subtrees: dict of SubtreeUnit
        Return:
            nested dict constructed by SubtreeUnit

        '''
        full_tree = {}
        # loop over bfg_tree
        self.create_root_tree(key_root, full_tree)
        # create children 
        while(len(bfs_tree) > 0):
            bfs_pair = bfs_tree[0]
            parent = bfs_pair[0]
            child = bfs_pair[1]
            find_parent = self._finditem(full_tree, parent)
            find_child = self._finditem(full_tree, child)
            if (find_parent != None):
                if (find_child !=None):
                    bfs_tree.pop(0)
                    subtrees.pop(0)
                else:
                    subtree = {}
                    subtree[child] = subtrees[child].get_subtree_unit()
                    if (find_parent[parent]['child_node'] == None):
                        find_parent[parent]['child_node'] = []
                    find_parent[parent]['child_node'].append(subtree)
                    bfs_tree.pop(0)
         
        return full_tree

    def _finditem(self, dict_obj, key):
        '''
        see if certain key in a nested dict exist
        if exist, return the dict{} that contains key at first level.
        if not, return None
        Args:
            dict_obj: nested dict assmbled with SubTreeUnit
            key: 
        '''
        if key in dict_obj.keys():
            return dict_obj
        for obj_key in dict_obj.keys():
            child_list = dict_obj[obj_key]['child_node']
            if child_list == None: 
                continue
            for child_dict in child_list:
                child_dict_obj = self._finditem(child_dict, key)
                if child_dict_obj != None:
                    return child_dict_obj

    def create_root_tree(self, key_root, full_tree):
        full_tree[key_root] = {
                'parent_to_own_slice' :[None, None],
                'child_node': []
                }


