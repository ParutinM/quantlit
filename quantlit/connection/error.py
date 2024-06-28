class Error(Exception):
    pass


class ClientError(Error):
    def __init__(self,
                 status_code: int,
                 code: int = None,
                 message: str = None,
                 data=None):
        self.status_code = status_code
        self.code = code
        self.message = message
        self.data = data


class ServerError(Error):
    def __init__(self, status_code, message):
        self.status_code = status_code
        self.message = message
