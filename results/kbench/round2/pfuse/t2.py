import gzip, json, sys, collections, re
t = json.load(gzip.open(sys.argv[1]))
ev = t["traceEvents"]
for c in ("python_function","cpu_op","kernel","user_annotation","cuda_runtime"):
    x=[e for e in ev if e.get("cat")==c and e.get("ph")=="X"]
    print(c, len(x), (min(e["ts"] for e in x)%1e8)/1e3, (max(e["ts"]+e["dur"] for e in x)%1e8)/1e3)
for e in ev:
    if e.get("cat") in ("kernel","gpu_memcpy"): print(e["cat"], (e["ts"]%1e8)/1e3, e["dur"], e["name"][:100], e["args"].get("graph id"))
for e in ev:
    if e.get("cat") in ("cuda_runtime",): print(e["cat"], (e["ts"]%1e8)/1e3, e["dur"], e["name"][:100])
pf = collections.Counter(re.sub(r"\(\d+\)","",e["name"]) for e in ev if e.get("cat")=="python_function")
for n,c in pf.most_common(60): print(c,n[:150])
