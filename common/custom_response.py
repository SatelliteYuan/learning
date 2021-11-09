'''
Author: xujiawei
Date: 2021-03-10 15:44:35
Description: description
'''
from common.error import Error


class CustomResponse(object):
    rsp = {}

    def __init__(self, error: Error, **kwargs):
        self.rsp['errorCode'] = error.code.value
        self.rsp['errorMsg'] = error.msg

        if 'results' in kwargs:
            self.rsp['results'] = kwargs['results']

    def to_json(self):
        return self.rsp
