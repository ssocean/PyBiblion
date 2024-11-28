# CONFIG FOR API KEYS
import os
import openai
openai.api_base = 'https://flag.smarttrot.com/'#"https://api.chatanywhere.cn"
# semantic scholar API || Free || non-mandatory || If not provided, you may get blocked by S2 for a little while. Try solve this by setting a longer waiting time period. ||You can request a Semantic Scholar API Key via https://www.semanticscholar.org/product/api#api-key-form
s2api = None
# OPENAI API || NOT Free || non-mandatory ||  You can avoid providing this by specifying the args.field parameter. || If you have trouble accessing OpenAI, try this-> https://github.com/chatanywhere/GPT_API_free
openai_key =None
# easy scholar API || Free || non-mandatory || If not provided, then no conference lever, journal IF will be presented on PDF || A key used for searching IF or CCF level for journal or conference, you can request one via https://www.easyscholar.cc/console/user/open
eskey = None
API_SECRET_KEY = None
BASE_URL = "https://flag.smarttrot.com/v1/"
os.environ["OPENAI_API_KEY"] = API_SECRET_KEY
os.environ["OPENAI_API_BASE"] = BASE_URL
os.environ["SERPER_API_KEY"] = None
S2_PAPER_URL = "https://api.semanticscholar.org/v1/paper/"
S2_QUERY_URL = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"
from .local_config import *