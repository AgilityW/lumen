from lumen.exceptions import LumenError, ConfigError, APIError, ParseError, UserInterrupt


class TestLumenExceptions:
    def test_hierarchy(self):
        assert issubclass(ConfigError, LumenError)
        assert issubclass(APIError, LumenError)
        assert issubclass(ParseError, LumenError)
        assert issubclass(UserInterrupt, LumenError)

    def test_config_error_message(self):
        err = ConfigError("test message")
        assert str(err) == "test message"

    def test_api_error_message(self):
        err = APIError("API failed")
        assert str(err) == "API failed"

    def test_parse_error_message(self):
        err = ParseError("parse failed")
        assert str(err) == "parse failed"

    def test_user_interrupt(self):
        err = UserInterrupt("user quit")
        assert str(err) == "user quit"
