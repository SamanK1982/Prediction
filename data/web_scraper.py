import requests
import json
import time

# set the start and end date for the historical data
start_date = "2020-01-01"
end_date = "2023-03-15"

# set the granularity of the data (in seconds)
granularity = 86400 # 1 day

# set the URL for the API endpoint
url = f"https://api.coinbase.com/v2/prices/BTC-USD/spot?start={start_date}&end={end_date}&granularity={granularity}"

# create an empty list to store the price data
prices = []

# loop through the API responses, paging through the results if necessary
while url:
    response = requests.get(url)
    data = json.loads(response.text)
    prices += data["data"]["prices"]
    url = data["pagination"]["next_url"]
    time.sleep(1) # sleep for 1 second to avoid hitting rate limits

# print the collected data
print(prices)