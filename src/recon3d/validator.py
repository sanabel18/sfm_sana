'''
1. All toml has to go through the validator at the level of project class, There should be 3 stages:
    - Critical: If this parameter is missing, then it throws error and forces the user to define it by him/herself.
    - Warning: If this parameter is missing, then the validator defines the default parameter for the user and throws warning so that the user can check if this default parameter is okay.
    - Nothing: Optional parameter or anything else without which the pipeline can still run with no risks.
   The developer should know that the validator has to be modified when the pipeline is updated.
'''