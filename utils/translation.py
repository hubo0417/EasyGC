import requests
import random
from hashlib import md5
from configs.base_config import BASE_CONFIG


class Translation_Baidu:
    appid = BASE_CONFIG["baidu_appid"]
    appkey = BASE_CONFIG["baidu_app_key"]
    from_lang = 'zh'
    to_lang = 'en'
    endpoint = 'http://api.fanyi.baidu.com'
    path = '/api/trans/vip/translate'
    url = endpoint + path
    salt = random.randint(32768, 65536)

    # Generate salt and sign

    @classmethod
    def excute_translation(cls, query: str):

        def _make_md5(s, encoding='utf-8'):
            return md5(s.encode(encoding)).hexdigest()

        def _sign(query: str):
            sign = _make_md5(cls.appid + query + str(cls.salt) + cls.appkey)
            return sign

        headers = {'Content-Type': 'application/x-www-form-urlencoded'}
        payload = {
            'appid': cls.appid,
            'q': query,
            'from': cls.from_lang,
            'to': cls.to_lang,
            'salt': cls.salt,
            'sign': _sign(query)
        }
        r = requests.post(cls.url, params=payload, headers=headers)
        result = r.json()
        return result["trans_result"][0]["dst"]
