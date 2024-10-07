from tool.tree_generator import TreeGenerator, load_src_markers, gen_stage_route_2_NID2SID
from tool.tree_to_nested_dict import Tree2NestedDict
import yaml

def test():
    '''
    run test and compare the output tree_lilly_test.yaml to tree_lilly_base.yaml
    the result should be the same
    '''
    src_mrk_path = './src_marker_lilly'
    src_root = 'root'
    src_marker_data_list, route_name_list = load_src_markers(src_mrk_path)
    slice_time_list = [15]*len(route_name_list)

    route_src_path_list, NID2SID_list  =  gen_stage_route_2_NID2SID(src_root, src_marker_data_list, route_name_list, slice_time_list) 
   
    tree_generator = TreeGenerator(route_src_path_list, NID2SID_list)
    max_root_num = tree_generator.get_max_root_num()
    G = tree_generator.get_graph()
    
    fulltrees = []
    for i in range(max_root_num):
        bfs_tree, root_node = tree_generator.get_BFS_tree()
        key_root = root_node
        tree_2_nested_dict = Tree2NestedDict(G, bfs_tree, key_root)
        nested_dict = tree_2_nested_dict.get_nested_dict()
        fulltrees.append(nested_dict)
    
    
    for fulltree in fulltrees:
        if list(fulltree.keys())[0] == "root/5_58_4":
            fulltree_export = fulltree
    with open('tree_lilly_test.yaml','w') as outfile:
        yaml.dump(fulltree_export, outfile,default_flow_style=None)


if __name__ == "__main__":
    test()
