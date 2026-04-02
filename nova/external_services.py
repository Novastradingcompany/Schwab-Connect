class InputError(ValueError):
    pass


class ExternalServiceError(RuntimeError):
    def __init__(self, service, action, user_message, original=None):
        super().__init__(user_message)
        self.service = service
        self.action = action
        self.user_message = user_message
        self.original = original


def exception_status_code(exc):
    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None)
    if status_code is not None:
        return status_code
    return getattr(exc, "status_code", None)


def public_error_message(exc, service=None, action=None):
    if isinstance(exc, InputError):
        return str(exc)
    if isinstance(exc, ExternalServiceError):
        return exc.user_message

    status_code = exception_status_code(exc)
    service_label = service or "External service"
    action_suffix = f" while {action}" if action else ""

    if status_code == 401:
        return f"{service_label} authentication failed{action_suffix}. Reconnect or check credentials."
    if status_code == 403:
        return f"{service_label} denied access{action_suffix}. Check account permissions."
    if status_code == 404:
        return f"{service_label} could not find the requested data{action_suffix}."
    if status_code == 429:
        return f"{service_label} rate-limited the request{action_suffix}. Try again shortly."
    if status_code is not None and 500 <= int(status_code) <= 599:
        return f"{service_label} is temporarily unavailable{action_suffix}. Try again shortly."

    text = str(exc).lower()
    if service == "OpenAI":
        if "api key" in text or "authentication" in text:
            return "OpenAI authentication failed. Check the API key."
        if "timeout" in text:
            return "OpenAI timed out. Try again."
        return "OpenAI request failed. Try again."
    if service == "Schwab":
        if "token" in text or "auth" in text or "login" in text:
            return "Schwab authentication failed. Log in again."
        if "timeout" in text:
            return "Schwab request timed out. Try again."
        return f"Schwab request failed{action_suffix}. Try again."

    return "Something went wrong. Check the error log for details."
