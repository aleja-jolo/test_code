from datetime import date
from functions_api_weather_module import get_data_from_api, get_hourly_temp, load_apidata_from_json, get_coordinates

"""
Notes:
- currently code uses only local time based on geographical coordinates. But printed time (time.localtime())
  convert UNIX time according to system's local time
- use package version for pandas & numpy based on project requirements.txt
Powered by DarkSky"  - Do not remove this line
"""


if __name__ == '__main__':

    # API Time Machine request inputs
    address = "Bahnhofplatz 1, Erlangen, Germany"
    latitude, longitude = get_coordinates(address)

    #latitude = 49.6158        # °N -> positive, °S -> negative
    #longitude = 10.8516        # °E -> positive, °W -> negative

    start_date = date(1990, 1, 1)  # yyyy-mm-dd # here '01' does not work
    end_date = date(1990, 1, 3)

    # Extract API data as py dicts
    output = load_apidata_from_json(file='test.json')

    # output = get_data_from_api(latitude=latitude, longitude=longitude, start_date=start_date, end_date=end_date)
    time_temp, max_temp = get_hourly_temp(output_data_dict=output, extract_daily_max_temp=False)
    pass
