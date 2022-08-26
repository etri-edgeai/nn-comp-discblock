""" 
    Configuration parser

"""



def get_embedding_options(config):
    """Get embedding options """

    if "diff_embedding" in config["embedding"]:
        options = config["diff_embedding"]
    elif "block" in config["embedding"]:
        options = config["block_options"]
    elif config["embedding"] == "svd":
        options = config["svd_options"]
    elif config["embedding"] == "word2ket":
        options = config["word2ket_options"]
    elif config["embedding"] == "smallfry":
        options = config["smallfry_options"]
    elif config["embedding"] == "tt":
        options = config["tt_options"]
    else:
        options = None
    return options
