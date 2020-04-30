import pytest
from Typical_M_Year import convert_to_df
import numpy as np


def test_input_dictionary():
    with pytest.raises(TypeError):
        convert_to_df(np.linspace(0, 40, 30))


def test_input_dict_keys():
    with pytest.raises(Exception):
        time_temp = json.load(open("Lancaster_temps.json"))
        time_temp['2002-09-27'] = 0
        df = convert_to_df(time_temp)


def test_no_data():
    with pytest.raises(Exception):
        time_temp = {}
        time_temp['2002-09-27'] = {}
        time_temp['2002-09-27']['time'] = []
        time_temp['2002-09-27']['temperature'] = []
        df = convert_to_df(time_temp)
        assert len(df) > 0

# def test_answer():
#     assert func(3) == 5

#
# def testing():
#     with pytest.raises(Exception):
#         time_temp = json.load(open("Lancaster_temps.json"))
#         convert_to_df(np.linspace(0,40,30),-1)
