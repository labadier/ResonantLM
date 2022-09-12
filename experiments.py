import os

phrases = ["Women's Harley Davidson Jacket",
"White pure linen curtain",
"Restaurant LE CONFIDENTIAL MARRAKECH international meals",
"Papaya body cream",
"5 nights all inclusive hotel accommodation at Disneyland Paris", 
"Natural cream", 
"cleaning products",
"The LED display is 28% brighter",
"Electric scooters the electric transition is underway",
"Executive backpack",
"Move freely while listening to your favorite music",
"Sound bar speaker for pc and tv",
"The lg led monitor/tv for tv has a dual purpose",
"Completely removes dirt and grime with",
"Your dark clothes should stay dark wash after wash",
"Fast ironing and outstanding results",
"Thank you for choosing the sports gps watch",
"Electric barbecue with powergrill technology",
"This chair is quite comfortable to rest in after a hard day's work",
"Are you looking for a high-performance cell phone ",
"Enjoy the most popular games and playstation exclusives ",
"The reno 34 shoulder bag (orange) is perfect for carrying your equipment",
"Mini camera for kids equipped with a high-definition display",
"Take a selfie with the mirror and light ring",
"Powerful and easy to use zoom with realistic details",
"Samsung's most epic technology is here",
"A revolutionary camera system",
"The neo qled TVs feature the new mini LEDs, which control the light",
"The c81 series combines an ultra-slim design, picture quality, and a high quality image",
"Combining premium materials and advanced audio technology",
"Sony wireless headphones",
"At 90 cm wide, haier's side-by-side 90 refrigerators offer large capacity",
"A dishwasher that seats up to 13 guests",
"Front-loading washing machine",
"This electric stove offers you everything you could ever need",
"The tristar ac-5531 portable air conditioner is efficient, portable and easy to use",
"Humidifier that increases and regulates the relative humidity of the room",
"The purline bio ethanol barbecues ",
"The new galaxy book2 features a 15-inch screen",
"Visionary Imac like never before",
"The new macbook pro offers out-of-the-box performance for professionals",
]


gm_scales = [0.2, 0.4, 0.6, 0.8] 
cf = {'O':0.25, 'C':0.2, 'E':0.2, 'A':0.2, 'N':0.25}
for p in ['O']:
  print("========== Factea O ==============\n")
  # for gm in gm_scales:
  print("=========== gm: {gm} =============\n")
  for i in phrases:
      os.system(f'python main_generation.py -seed "{i}" -dt {p} -gm {cf[p]}')