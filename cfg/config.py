import os
import json

CONFIG_FILE_NAME = ".service_api_config.json"

# Define the config schema
_config_schema = {
    "s2api": {
        "require": False,
        "description": "Semantic Scholar API key (Free). Avoids rate-limiting."
    },
    "eskey": {
        "require": False,
        "description": "Easy Scholar API key (Free). For journal IF and conference levels."
    },
    "SERPER_API_KEY": {
        "require": False,
        "description": "Serper API key (Free). Enables search engine queries like Google."
    },
    "LLM_SERVICE_KEY": {
        "require": True,
        "description": "Main API key for authentication (like OpenAI api key)."
    },
    "BASE_URL": {
        "require": True,
        "description": "Base URL for the LLM API endpoint."
    },
    # OSS keys
    "OSS_KEY_ID": {
        "require": True,
        "description": "Aliyun OSS Key ID"
    },
    "OSS_KEY_SECRET": {
        "require": True,
        "description": "Aliyun OSS Key Secret"
    },
    "OSS_ENDPOINT": {
        "require": True,
        "description": "Aliyun OSS Endpoint"
    },
    "OSS_BUCKET_NAME": {
        "require": True,
        "description": "Aliyun OSS Bucket Name"
    },

    # MySQL
    "MYSQL_URL": {
        "require": True,
        "description": "SQLAlchemy-compatible MySQL connection URL"
    }
}


def _get_config_path():
    return os.path.join(os.path.expanduser("~"), CONFIG_FILE_NAME)


def _create_if_missing():
    path = _get_config_path()
    if not os.path.exists(path):
        with open(path, "w") as f:
            json.dump({k: "" for k in _config_schema}, f, indent=4)
        print(f"🔧 Created default config at {path}. Please fill in your API keys.")


def _load_config():
    with open(_get_config_path(), "r") as f:
        return json.load(f)


def _validate(config):
    missing = []
    for key, meta in _config_schema.items():
        val = config.get(key)
        if meta["require"] and not val:
            missing.append(key)
        elif not meta["require"] and not val:
            print(f"ℹ️ Optional config `{key}` is not set — {meta['description']}")
    if missing:
        raise ValueError(
            f"❌ Missing required config keys: {', '.join(missing)}\n To avoid this error, please EDIT config at: {_get_config_path()}")


# Load and validate config
_create_if_missing()
_config_data = _load_config()
_validate(_config_data)

# === Explicit variable exports ===
s2api = _config_data.get("s2api") or None
eskey = _config_data.get("eskey") or None
SERPER_API_KEY = _config_data.get("SERPER_API_KEY") or None
LLM_SERVICE_KEY = _config_data.get("LLM_SERVICE_KEY") or None
BASE_URL = _config_data.get("BASE_URL") or None

os.environ["OPENAI_API_KEY"] = LLM_SERVICE_KEY
os.environ["OPENAI_API_BASE"] = BASE_URL
os.environ["SERPER_API_KEY"] = SERPER_API_KEY

OSS_KEY_ID = _config_data.get("OSS_KEY_ID") or None
OSS_KEY_SECRET = _config_data.get("OSS_KEY_SECRET") or None
OSS_ENDPOINT = _config_data.get("OSS_ENDPOINT") or None
OSS_BUCKET_NAME = _config_data.get("OSS_BUCKET_NAME") or None

MYSQL_URL = _config_data.get("MYSQL_URL") or None
