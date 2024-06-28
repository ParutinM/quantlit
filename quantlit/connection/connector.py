import json
import requests
import logging

from quantlit.connection.error import ClientError, ServerError
from quantlit.connection.market import Market


class Connector:
    def __init__(self,
                 base_url: str,
                 api_key: str = None,
                 api_secret: str = None):
        self.base_url = base_url
        self.api_key = api_key
        self.api_secret = api_secret

        self.session = requests.Session()
        self._logger = logging.getLogger(__name__)

    @property
    def market(self) -> Market:
        return Market(self)

    def get(self, url_path: str, params: dict = None):
        if params is None:
            params = {}
        url = self.base_url + url_path
        self._logger.debug(f"url: {url}")
        response = self.session.get(url, params=params)
        self._logger.debug(f"raw response: {response.text}")
        self._handle_exception(response)
        return response.json()

    def _handle_exception(self, response: requests.Response):
        if response.status_code < 400:
            self._logger.debug(f"valid status code: {response.status_code}")
            return
        elif 400 <= response.status_code < 500:
            self._logger.debug(f"not valid status code: {response.status_code}")
            try:
                err = response.json()
            except json.JSONDecodeError:
                raise ClientError(response.status_code, None, response.text)
            else:
                raise ClientError(response.status_code, err.get("code"), err.get("msg"), err.get("data"))
        self._logger.debug(f"not valid status code: {response.status_code}")
        raise ServerError(response.status_code, response.text)
