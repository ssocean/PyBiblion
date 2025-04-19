import os
from diskcache import Cache

# 保证缓存文件夹在当前这个脚本文件所在的目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_PATH = os.path.join(BASE_DIR, ".request_cache")

disk_cache = Cache(CACHE_PATH)