import requests

url = 'http://localhost:9696/predict'

customer_id = '11'
customer = {'battery_power': 441,
  'blue': 1,
  'clock_speed': 2,
  'dual_sim': 0,
  'fc': 0,
  'four_g': 0,
  'int_memory': 8,
  'm_dep': 0.4,
  'mobile_wt': 100,
  'n_cores': 6,
  'pc': 6,
  'px_height': 200,
  'px_width	': 300,
  'ram': 2500,
  'sc_h': 8,
  'sc_w': 8,
  'talk_time': 10,
  'three_g': 1,
  'touch_screen	': 1,
  'wifi': 1}

response = requests.post(url, json=customer).json()
print(response)

if response['churn'] == True:
    print('sending promo email to {}'.format(customer_id))
else:
    print('not sending promo email to {}'.format(customer_id))