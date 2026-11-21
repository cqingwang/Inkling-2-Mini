import gzip, json, sys, collections, re
t = json.load(gzip.open(sys.argv[1]))
ev = t["traceEvents"]
cats = collections.Counter((e.get("ph"), e.get("cat")) for e in ev)
print(cats.most_common(20))
k = [e for e in ev if e.get("cat")=="kernel"]
print(len(k)); print(json.dumps(k[0])[:1500])
cpu = [e for e in ev if e.get("cat") in ("cpu_op","user_annotation","python_function")]
print(len(cpu)); 
for e in cpu[:5]: print(json.dumps(e)[:600])
