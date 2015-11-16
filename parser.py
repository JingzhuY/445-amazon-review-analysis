# -*- coding: utf-8 -*-
import json
import itertools

names=["product/productId","review/userId","review/profileName","review/helpfulness","review/score","review/time","review/summary","review/text"]

with open('finefoods.txt', 'rU') as f_in, open('finefoods_short.json', 'w') as f_out:
	f_out.write("[")
	n=0
	for pid, uid, name, helpfulness, score, time, summary, text, linebreak in itertools.izip_longest(*[f_in]*9):
		n+=1
		print n
		print helpfulness
		record = {
            names[0]: pid.strip(names[0]+": ").strip(),
            names[1]: uid.strip(names[1]+": ").strip(),
            names[2]: name.strip(names[2]+": ").strip().decode('utf-8',"ignore"),
            names[3]: helpfulness.strip(names[3]+": ").strip(),
            names[4]: score.strip(names[4]+": ").strip(),
            names[5]: time.strip(names[5]+": ").strip(),
            names[6]: summary.strip(names[6]+": ").strip().decode('utf-8',"ignore"),
            names[7]: text.strip(names[7]+": ").strip().decode('utf-8',"ignore")
            }
		f_out.write(json.dumps(record))
		if n!=568454:
			f_out.write(",")
	f_out.write("]")