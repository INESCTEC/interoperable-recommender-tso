from time import time
from loguru import logger
from http import HTTPStatus

from .Endpoint import Endpoint, post_actions
from .RequestController import RequestController
from .exception import LoginException, PostActionsException


class Controller(RequestController):
    def __init__(self):
        RequestController.__init__(self)
        self.access_token = ""

    def __check_if_token_exists(self):
        if self.access_token is None:
            e_msg = "Access token is not yet available. Login first."
            logger.error(e_msg)
            raise ValueError(e_msg)

    def set_access_token(self, token):
        self.access_token = token

    def login(self, email: str, password: str):
        raise NotImplementedError("Method not implemented.")

    def __request_template(self,
                           endpoint_cls: Endpoint,
                           log_msg: str,
                           exception_cls,
                           data: dict = None,
                           params: dict = None,
                           url_params: list = None,
                           raise_exception: bool = True,
                           ) -> dict:
        t0 = time()
        rsp = self.request(
            endpoint=endpoint_cls,
            data=data,
            params=params,
            url_params=url_params,
            auth_token=self.access_token
        )

        # -- Inspect response:
        if (rsp.status_code == HTTPStatus.OK) or \
                (rsp.status_code == HTTPStatus.CREATED):
            logger.debug(f"{log_msg} ... Ok! ({time() - t0:.2f})")
            return rsp.json()
        elif rsp.status_code == HTTPStatus.INTERNAL_SERVER_ERROR:
            log_msg_ = f"{log_msg} ... Failed! ({time() - t0:.2f})"
            if raise_exception:
                msg = "Internal Server Error."
                raise exception_cls(message=log_msg_, errors={"message": msg})
            else:
                return rsp.json()
        else:
            if raise_exception:
                log_msg_ = f"{log_msg} ... Failed! ({time() - t0:.2f})"
                logger.error(log_msg_ + f"\n{rsp.json()}")
                raise exception_cls(message=log_msg_, errors=rsp.json())
            else:
                return rsp.json()

    def post_actions_data(self, payload: dict):
        """
        Post countries actions data to EnergyAPP backend
        """
        country_code = payload["metadata"]["country_code"]
        country_name = payload["metadata"]["country_name"]
        # request data:
        response = self.__request_template(
            endpoint_cls=Endpoint(post_actions.POST, post_actions.uri),
            log_msg=f"Posting coordinated actions data for {country_name} ({country_code})",  # noqa
            exception_cls=PostActionsException,
            data=payload,
        )
        return response
