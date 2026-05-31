import json
def r(f): return open(f,encoding="utf-8").read()
NL=chr(10)+chr(10)
s={}
s["Introduction"]=r("_i1.txt")+NL+r("_i2.txt")+NL+r("_i3.txt")
s["Data and Backtest Setup"]=r("_d1.txt")+NL+r("_d2.txt")
s["Methodology"]=r("_m1.txt")+NL+r("_m2.txt")+NL+r("_m3.txt")
s["Results"]=r("_r1.txt")+NL+r("_r2.txt")+NL+r("_r3.txt")+NL+r("_r4.txt")
s["Robustness Analysis"]=r("_rb1.txt")+NL+r("_rb2.txt")+NL+r("_rb3.txt")
s["Discussion"]=r("_di1.txt")+NL+r("_di2.txt")+NL+r("_di3.txt")+NL+r("_di4.txt")
s["Conclusion"]=r("_c1.txt")+NL+r("_c2.txt")+NL+r("_c3.txt")
print(json.dumps(s,ensure_ascii=False,indent=2))
