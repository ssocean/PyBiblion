import threading
import os
import sys
from requests.exceptions import RequestException
import requests
import requests_cache
from datetime import timedelta

base_dir = os.path.dirname(os.path.abspath(__file__))

sys.path.append(base_dir)

def generate_cache_file_name(url='', force_file_name=None):
    if not force_file_name:
        if 'authors?' in url:
            return os.path.join(base_dir, 'authorsCache.sqlite')
        if 'citations?' in url:
            return os.path.join(base_dir, 'citationsCache.sqlite')
        if 'references?' in url:
            return os.path.join(base_dir, 'referencesCache.sqlite')
        if 'batch' in url:
            return os.path.join(base_dir, 'batchCache.sqlite')
        if 'bulk' in url:
            return os.path.join(base_dir, 'bulkCache.sqlite')
        if 'paper/search?' in url:
            return os.path.join(base_dir, 'relevantCache.sqlite')
    else:
        return os.path.join(base_dir, force_file_name)
    return os.path.join(base_dir, 'generalCache.sqlite')

# Dictionary to hold CachedSession objects per cache file
cache_sessions = {}
cache_sessions_lock = threading.Lock()

def get_cached_session(cache_file):
    """
    Retrieve a CachedSession for the given cache file.
    Ensures that only one session is created per cache file.
    """
    with cache_sessions_lock:
        if cache_file not in cache_sessions:
            # Create a new CachedSession with SQLite backend
            cache_sessions[cache_file] = requests_cache.CachedSession(
                cache_name=cache_file,
                backend='sqlite',
                expire_after=timedelta(days=30)  # Adjust cache expiration as needed
            )
        return cache_sessions[cache_file]

def cached_get(url, headers=None):
    """
    Thread-safe GET request with caching.
    """
    cache_file = generate_cache_file_name(url)
    session = get_cached_session(cache_file)
    try:
        response = session.get(url, headers=headers)
        return response
    except RequestException as e:
        # Handle exceptions as needed
        print(f"Error fetching {url}: {e}")
        return None
