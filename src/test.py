from predict_rehapp import rehappClassifier
import pytest

def test_answer():
    myClassifer = rehappClassifier()
    assert myClassifer.predict("../output/test.csv") == [4]
