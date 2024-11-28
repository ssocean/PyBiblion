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



def cached_get(url, params=None, headers=None):
    """
    Thread-safe GET request with caching.
    Supports URL query parameters (params) and headers.
    """
    # Generate the cache file name based on the URL
    cache_file = generate_cache_file_name(url)

    # Retrieve the cached session for the specified cache file
    session = get_cached_session(cache_file)

    try:
        # Perform the GET request, passing both params and headers if they are provided
        response = session.get(url, params=params, headers=headers)

        # Optionally, check if the response is valid (e.g., status code 200)
        if response.status_code == 200:
            return response
        else:
            # Optionally, handle different status codes (e.g., 404, 500)
            print(f"Request failed with status code {response.status_code}")
            return None
    except RequestException as e:
        # Handle exceptions (e.g., network error, timeout) and re-raise as needed
        print(f"Error fetching {url}: {e}")
        raise RequestException(f'{e}')



def cached_post(url, params=None, json=None, headers=None):
    """
    Thread-safe POST request with caching. This handles both `params` and `json` arguments.
    """
    cache_file = generate_cache_file_name(url)
    session = get_cached_session(cache_file)

    try:
        # Ensure the POST request uses both params and json correctly
        response = session.post(url, params=params, json=json, headers=headers)

        # Optionally, cache the response if it's a successful request (e.g., status code 200)
        if response.status_code == 200:
            return response
        else:
            # Optionally, handle different status codes as needed
            print(f"Request failed with status code {response.status_code}")
            return None
    except RequestException as e:
        # Handle exceptions as needed
        print(f"Error fetching {url}: {e}")
        raise RequestException(f'{e}')





