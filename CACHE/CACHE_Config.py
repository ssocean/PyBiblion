import os.path



import os
import sys

base_dir = os.path.dirname(os.path.abspath(__file__))


sys.path.append(base_dir)
def generate_cache_file_name(url='',force_file_name=None):
    if not force_file_name:
        if 'authors?' in url:
            return os.path.join(base_dir, '.authorsCache')
        if 'citations?' in url:
            return os.path.join(base_dir, '.citationsCache')
        if 'references?' in url:
            return os.path.join(base_dir, '.references')

        if 'batch' in url:
            return os.path.join(base_dir,'.batchCache')
        if 'bulk' in url:
            return os.path.join(base_dir,'.bulkCache')
        if 'paper/search?' in url:
            return os.path.join(base_dir, '.relevantCache')
    else:
        return os.path.join(base_dir, force_file_name)
    return os.path.join(base_dir, '.generalCache')