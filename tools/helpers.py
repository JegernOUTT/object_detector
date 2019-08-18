

def list_if_not_list(obj):
    return obj if type(obj) == list else [obj]
