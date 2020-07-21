from main import *


def test_q1():
    result = q1()

    assert type(result) == list

def test_q2():
    result = q2()

    assert type(result) == int

def test_q3():
    result = q3()

    assert type(result) == int

def test_q4():
    result = q4()

    assert type(result) == float

def test_q5():
    result = q5()

    assert type(result) == tuple
    assert len(result) == 3
    assert type(result[0]) == int
    assert type(result[1]) == int
    assert type(result[2]) == bool

def test_q6():
    result = q6()

    assert type(result) == int

def test_q7():
    result = q7()

    assert type(result) == float
