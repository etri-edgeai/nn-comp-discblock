""" 
    Configuration parser

Copyright 2022. ETRI all rights reserved

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

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
