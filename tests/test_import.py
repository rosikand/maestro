import maestro


def test_version():
    assert isinstance(maestro.__version__, str)


def test_hello(capsys):
    maestro.hello()
    assert capsys.readouterr().out == "hello, this is maestro\n"
