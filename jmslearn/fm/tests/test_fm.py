
from jmslearn.fm.fm import FM
from ..fm import FM


def test1():
    assert FM().check() == 1
