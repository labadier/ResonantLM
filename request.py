import csv, pandas as pd, re, string
import urllib.parse, requests, json
import argparse, sys

def check_params(args=None):

	parser = argparse.ArgumentParser(description='Language Model Encoder')

	parser.add_argument('-seed', metavar='seed', required = True, help='Seed')
	parser.add_argument('-gender', metavar='gender', required = True, help='Gender')

	return parser.parse_args(args)

if __name__ == '__main__':


	parameters = check_params(sys.argv[1:])

	seed = parameters.seed
	gender = 1 if parameters.gender[0] == 'm' else 0


	query = f"http://localhost:5201/api/v1/generator/?seed={urllib.parse.quote(seed)}&lang=en&faceta=g&classlabel={gender}"

	try:
		response = requests.request("POST", query)
		response.raise_for_status()
		result = json.loads(response.text)

	except requests.exceptions.RequestException as e: 
		exit(0)

	ans = []
	for i in result["facets"][0].keys():
		if "sente" in i:
			ans += [(int(i[-1]), result["facets"][0][i])]

	for i in sorted(ans):
		print(i[1])

	exit(0)



