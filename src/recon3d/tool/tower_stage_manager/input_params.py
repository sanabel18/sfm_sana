import copy
class DownloadParam:
    """
    Args:
    input_data: dict 
        structure that contains the info of downloaded files
        each commit-tag will have dict that marks file labels and its
        corresponding file number. After clone, and download process, 
        the file numbers will be checked. 
        key: commit-tag:string, value:dict{key:file-label:string, value: file-number:int}
        example:
            {'marker':{'marker':1}, 'cmr-inpaint-mask': {'mask':1, 'stitched_video':1}}
    """
    def __init__(self, input_data):
        self.input_data = input_data
   
    def get_input_data(self):
        return self.input_data

class CommitParam:
    """
    Args:
    commit_msg: str, commit message
    """
    def __init__(self, commit_msg):
        self.commit_msg = commit_msg
    def get_commit_msg(self):
        return self.commit_msg

class LabelParam:
    """
    dict says which label goes to which files
    {'label':[file1, file2], 'label2':[file3]}
    Args: 
    label_to_file: dict: key: string, val: list of string
    """
    def __init__(self, label_to_file):
        self.label_to_file = label_to_file
    
    def get_label_to_file_dict(self):
        return self.label_to_file

class AddTagParam:
    """
    put dst_commit_tag on the new commit generated
    Args: 
    dst_commit_tag: str, distination commit tag
    """
    def __init__(self, dst_commit_tag):
        self.dst_commit_tag = dst_commit_tag
    
    def get_dst_commit_tag(self):
        return self.dst_commit_tag

