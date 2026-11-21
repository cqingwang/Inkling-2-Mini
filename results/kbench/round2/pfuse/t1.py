import gzip, json, sys, collections, re
t = json.load(gzip.open(sys.argv[1]))
ev = t["traceEvents"]
for e in ev:
    if e.get("cat") in ("kernel","gpu_memcpy","cuda_runtime","cuda_driver","overhead","gpu_user_annotation","cpu_op","Trace"):
        print(e["cat"], round(e["ts"]%1e8,1), round(e.get("dur",0),1), e["name"][:110])
print([k for k in t if k!="traceEvents"])
print({k:(t[k] if not isinstance(t[k],(list,dict)) else str(t[k])[:300]) for k in t if k!="traceEvents"})
ua = collections.Counter(e["name"] for e in ev if e.get("cat")=="user_annotation")
print(ua.most_common(40))
