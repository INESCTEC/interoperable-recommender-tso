# flake8: noqa

from dataclasses import dataclass
from collections import namedtuple
from conf import settings

api_basepath = settings.ENERGYAPP["basepath"]
fields = ('GET', 'POST', 'uri')
endpoint = namedtuple('endpoint', fields)

# HTTP methods
http_methods = "GET", "POST"

# Authentication
post_actions = endpoint(*http_methods, f"{api_basepath}/recommendations")


@dataclass(frozen=True)
class Endpoint:
    http_method: str
    uri: str
