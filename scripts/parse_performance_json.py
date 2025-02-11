"""
	Script to parse directory with performance metrics stored in json's. Basically turn [{'k': v}, ..] in {'k': [v,..]} 
"""

import json
import os
import math
import statistics as stat
import argparse
from typing import List, Dict
from collections import defaultdict

def parse_dir(json_dir: str=None)->Dict[str,List]:
	jsons = [f for f in os.listdir(json_dir) if f.endswith('.json')]

	res_dict = defaultdict(list)
	if jsons:
		for _json in jsons:
			floc = os.path.join(json_dir, _json)
			d = json.load(open(floc, 'rb'))
			for k,v in d.items():
				if isinstance(v, dict):
					for _k, _v in v.items():
						res_dict[f"{k}_{_k}"].append(_v)
				else:
					res_dict[k].append(v)
	return res_dict


def get_aggregates(res_dict: dict=None)->Dict[str,Dict]:
	out_string = "Aggregate results \n"
	out_string += "="*50+"\n"
	out_string += "class\tmean\tmedian\tstdev\n"

	print(res_dict)
	agg_dict = defaultdict(dict)
	for k,v in res_dict.items():
		print(v)
		_mean, _median, _stdev = sum(v)/len(v), stat.median(v), stat.stdev(v)

		out_string += f"{k}\t{round(_mean,3)}\t{round(_median,3)}\t{round(_stdev,3)}\n"
		out_string += "-"*50+"\n"

		agg_dict[k] = {
			'mean': round(_mean,3),
			'median': round(_median,3),
			'stdev': round(_stdev,3)
		}
	print(out_string)
	return agg_dict


if __name__=="__main__":
	argparser = argparse.ArgumentParser()
	argparser.add_argument('--dir', type=str, required=True)

	args = argparser.parse_args()

	collected_dict = parse_dir(args.dir)
	json.dump(collected_dict, 
		open(os.path.join(args.dir, 'collected_results.json'), 'w', encoding='latin1'))
	
	aggregated_dict = get_aggregates(collected_dict)
	json.dump(aggregated_dict, 
		open(os.path.join(args.dir, 'aggregated_results.json'), 'w', encoding='latin1'))

