import glob
import logging
import time
from typing import List, Tuple, Iterator, Optional

import numpy as np
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from index_utils.faiss_indexers import (
    DenseFlatIndexer
)

import random

logger = logging.getLogger()
logger.setLevel(logging.INFO)
if logger.hasHandlers():
    logger.handlers.clear()
console = logging.StreamHandler()
logger.addHandler(console)

class DenseRetriever(object):
    """
    Does passage retrieving over the provided index and question encoder
    """

    def __init__(
        self,
        ctx_files_pattern
    ):
        self.index = self._create_index(ctx_files_pattern)

    def _create_index(self, ctx_files_pattern):
        index = DenseFlatIndexer(768, 50000, num_threads=5)
        input_paths = glob.glob(ctx_files_pattern)
        logger.info(f"Number of index files found: {len(input_paths)}") 
        logger.info("Reading all passages data from files: %s", input_paths)
        start_time = time.time()
        index.index_data(input_paths)
        logger.info(f"Done. Took {time.time() - start_time:.0f} seconds to build index.")
        return index       

    def get_top_docs(
        self, query_vectors: np.array, top_docs: int = 100
    ) -> List[Tuple[List[object], List[float]]]:
        results = self.index.search_knn(query_vectors, top_docs)
        return results
    
    def get_random_docs(self, number, black_list):
        ids_to_choose = [key[0] for key in self.index.id_to_db_id.keys() if key[0] not in black_list]
        rand_indices = random.sample(ids_to_choose, number)
        data_and_vectors = [self.index.id_to_db_id[idx] for idx in rand_indices]
        return data_and_vectors
    
    def get_embeds_from_docs_ids(self, docs_ids):
        embeds = []
        for id in docs_ids:
            embeds.append(self.index.id_to_db_id[id][1])
        return embeds
    
    def get_input_ids_from_docs_id(self, docs_id):
        return self.index.id_to_db_id[docs_id][0][1]

