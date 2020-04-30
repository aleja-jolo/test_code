import requests
import json
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pytz
from geopy.geocoders import Nominatim

# todo : convert different types of user date formats to python acceptable one
# for ex. - 2019-01-01 to be converted into 2019-1-1


# API call
def get_data_from_api(latitude, longitude, start_date, end_date):
    """
    fetch raw data from API as JSON based on inputs and convert JSON to python dict objects
    :param latitude:
    :param longitude:
    :param start_date:
    :param end_date:
    :return: output_data_dict: dicts with {date1:data1, date2:data2,...} based on start and end date
    """

    api_key = "6c464f06a6ae386f9dd2d9d8cb43095b"
    output_data_dict = dict()
    processing_date = start_date
    delta = timedelta(days=1)
    num_of_days = end_date - start_date

    for i in range(0, num_of_days.days + 1):
        location_time = str(processing_date) + "T00:00:00"
        api_url = f'https://api.darksky.net/forecast/{api_key}/{latitude},{longitude},{location_time}' \
                  f'?exclude=currently,flags&units=si'
        response = requests.get(api_url)
        output_data_dict[str(processing_date)] = response.json()
        processing_date += delta

    return output_data_dict


def load_apidata_from_json(file):
    """
    TO BE USED IN FUTURE
    loads the JSON file based on raw API data into a dict similar to get_data_from_api()
    :param file:
    :return: output_data_dict: dict, same as output of get_data_from_api()
    """
    with open(file, 'r') as f:
        output_data_dict = json.load(f)
    return output_data_dict


def get_hourly_temp(output_data_dict, extract_daily_max_temp=False):
    """
    processes dict to extract hourly temperature data for all days in input dict
    :param extract_daily_max_temp: set to true if max daily temp is needed separately
    :param output_data_dict:
    :return: time_temp: dict containing hourly data for all days in input dict
    """
    time_temp = dict()
    max_temp_data = dict()

    for day in output_data_dict.keys():
        # Save max daily data in a new dict
        time_zone = output_data_dict[day]['timezone']

        # Check if data is available from api
        try:
            hourly_data = output_data_dict[day]['hourly']['data']
        except KeyError:
            print(f"{day}: Hourly data (completely) not available for given inputs")

        time_temp[day] = {}
        time_temp[day]['time'] = []
        time_temp[day]['temperature'] = []

        for hd in hourly_data:

            # Extract time and convert to human readble
            local_time = datetime.fromtimestamp(hd['time'], tz=pytz.timezone(time_zone))
            time_temp[day]['time'].append(local_time.strftime('%d/%m/%Y %H:%M:%S'))

            # te only replaced if new values exist, else old temperature
            # this algo fills data gaps with previous hour temperature
            try:
                te = hd['temperature']
	        time_temp[day]['temperature'].append(te)

            except KeyError:
                print(f"{local_time}: Temperature data gaps here")


        # Save max daily data in a new dict
        if extract_daily_max_temp:
            max_temp_data[day] = dict()
            ts_max = output_data_dict[day]['daily']['data'][0]['temperatureMaxTime']
            local_ts_max = datetime.fromtimestamp(ts_max, tz=pytz.timezone(time_zone))
            max_temp_data[day]['time'] = local_ts_max
            max_temp_data[day]['temperatureMax'] = output_data_dict[day]['daily']['data'][0]['temperatureMax']

    return time_temp, max_temp_data


def get_coordinates(user_address):
    """
    Convert given address into coordinates
    :param user_address: string
    :return: coordinates as floats
    """
    input_address = Nominatim(user_agent="userLocation")
    coordinates = input_address.geocode(user_address)
    return coordinates.latitude, coordinates.longitude


def api_error_msg():
    print("Data does not exist in API for given inputs. Try:")
    print("- Check coordinates and the N/W/E/S signs")
    print("- New coordinates (nearby town,city, etc.)")
    print("- Another date range (near past)")


def plot_temps(max_temp_data, time_temp):
    """
    DATA VISUALISATION
    :param max_temp_data:
    :param time_temp:
    :return:
    """
    dates = []
    times = []
    temps = []
    for key, value in max_temp_data.items():
        times.append(value['time'])
        temps.append(value['temperatureMax'])
        dates.append(key)

    for key, value in time_temp.items():
        plt.plot(value['time'], value['temperature'], 'r-o')
    # plt.show()
    # x = range(0, len(max_temp_data))
    # plt.xticks(x, dates)
    plt.plot(times, temps, 'bx')
    plt.xlabel('Time')
    plt.ylabel('Temperature')
    plt.show()
    pass
