#%%
from utils.params import params as var
import twitter, csv

api = twitter.Api(consumer_key=var.twitter_api['APP_CONSUMER_KEY'],
                  consumer_secret=var.twitter_api['APP_CONSUMER_SECRET'],
                  access_token_key=var.twitter_api['ACCESS_TOKEN'],
                  access_token_secret=var.twitter_api['ACCESS_TOKEN_SECRET'],
                  sleep_on_rate_limit=True)

interestUsers = []
with open('utils/twitterUsers.txt', 'r') as file:
  interestUsers = [i[:-1] for i in file]

with open('data/tweets_tech.csv', 'wt', newline='', encoding="utf-8") as csvfile:
  
  spamwriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
  spamwriter.writerow(['user', 'user_id', 'tweet'])

  for usrs in interestUsers:
    statuses = api.GetUserTimeline(user_id = usrs, exclude_replies=True, count=2)
    spamwriter.writerows(map(lambda i: [i.AsDict()['user']['screen_name'], i.AsDict()['user']['id'], i.AsDict()['text']], statuses))
    print(f'User {usrs} retrieved!')

# %%

# %%
