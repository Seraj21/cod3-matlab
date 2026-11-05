"""Recreate reviewed artifacts entirely from immutable GitHub Actions outputs."""
import base64,concurrent.futures,csv,hashlib,io,json,os,re,shutil,subprocess,sys,zipfile
from pathlib import Path
import numpy as np
from PIL import Image

ROOT=Path("artifacts");ROOT.mkdir(exist_ok=True)
REPO="Seraj21/cod3-matlab"
def api(path):
    return json.loads(subprocess.check_output(["gh","api",f"repos/{REPO}/{path}"]))
original_half={f"v2-evaluation-{bs}-{b}" for bs,bs_batches in [(32,[0,1,7]),(64,[0,3,4,7])] for b in bs_batches}
plans=[
 (41946464299,lambda n:n=="v3-refined-frozen-model" or n.startswith(("v3-ls-evaluation-","v3-final-evaluation-"))),
 (41943264421,lambda n:n.startswith("v3-evaluation-")),
 (41942634465,lambda n:n.startswith("v3-receiver-")),
 (41908344154,lambda n:n=="v2-frozen-model" or n in original_half),
 (41933436597,lambda n:n.startswith("v2-evaluation-")),
 (41920713983,lambda n:n.startswith("v2-robust-")),
 (41937575835,lambda n:n.startswith("v2-phase-control-")),
 (41937575516,lambda n:n=="v2-pzf-pilot-control"),
 (41940416732,lambda n:n.startswith("v2-gauge-control-")),
 (41944581302,lambda n:n=="v3-reference-control"),
 (41945557567,lambda n:n=="v3-reference-residual-control"),
]
selected={}
for run,accept in plans:
    state=api(f"actions/runs/{run}")
    assert state["conclusion"]=="success" or (run==41908344154 and state["status"]=="completed" and state["head_sha"]=="ec00ae08f0e470071158768d92b8320c490784c3"),(run,state["conclusion"])
    rows=api(f"actions/runs/{run}/artifacts?per_page=100")["artifacts"]
    for m in rows:
        if accept(m["name"]):
            assert not m["expired"]
            m["_run"]=run;m["_source"]=state["head_sha"]
            assert m["name"] not in selected,m["name"]
            selected[m["name"]]=m

def download(m):
    data=subprocess.check_output(["gh","api",f"repos/{REPO}/actions/artifacts/{m['id']}/zip"])
    digest=hashlib.sha256(data).hexdigest()
    assert digest==m["digest"].split(":")[-1],m["name"]
    path=ROOT/(m["name"]+".zip");path.write_bytes(data)
    with zipfile.ZipFile(path) as archive:assert archive.testzip() is None
    return dict(name=m["name"],artifact_id=m["id"],run_id=m["_run"],source=m["_source"],sha256=digest,bytes=len(data))
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    catalog=list(pool.map(download,selected.values()))
(ROOT/"GitHub-artifact-catalog.json").write_text(json.dumps(catalog,indent=2))
print("All immutable source artifact digests and ZIP CRCs verified:",len(catalog),flush=True)

def csv_in(archive,base):
    name=next(n for n in archive.namelist() if Path(n).name==base)
    return list(csv.DictReader(io.StringIO(archive.read(name).decode("utf-8-sig"))))
training={}
for scheme in ["MR","ZF","PZF"]:
    with zipfile.ZipFile(ROOT/f"v3-receiver-{scheme}.zip") as z:
        c=csv_in(z,"rx_conditions.csv");v=csv_in(z,"rx_channel_validation.csv")
        tr=[r for r in c if r["Training"].lower() in ["true","1"]]
        va=[r for r in c if r["Training"].lower() in ["false","0"]]
        assert len(tr)==640 and len(va)==80 and len(v)==640
        assert len({r["Seed"] for r in tr})==640 and len({r["Seed"] for r in va})==80
        assert {r["Seed"] for r in tr}.isdisjoint({r["Seed"] for r in va})
        means={k:float(np.mean([float(r[k]) for r in v])) for k in ["MSE_LS","MSE_LMMSE","MSE_CNN"]}
        hard={int(r["Seed"]) for r in va if r["HardCase"].lower() in ["true","1"]}
        hv=[r for r in v if int(r["ChannelSeed"]) in hard]
        training[scheme]=dict(train_channels=640,validation_channels=80,channel_MSE=means,
          hard_validation=dict(channels=len(hard),MSE_LMMSE=float(np.mean([float(r["MSE_LMMSE"]) for r in hv])),
            MSE_CNN=float(np.mean([float(r["MSE_CNN"]) for r in hv]))))
Path("tmp").mkdir(exist_ok=True)
Path("tmp/V3-training-review.json").write_text(json.dumps(training,indent=2))

profiles=[
 ("reviewed_half",["--run-id","41908344154","--source","ec00ae08f0e470071158768d92b8320c490784c3",
  "--recovery-run-id","41933436597","--rate",".5","--artifact-prefix","v2-evaluation",
  "--archive-name","MU-V2-LDPC-Half-Results.zip"]),
 ("reviewed_eighth",["--run-id","41920713983","--source","42a0941d6c033ec426c8f7f6cf0ed09449685e3e",
  "--rate",".125","--artifact-prefix","v2-robust","--archive-name","MU-V2-LDPC-Eighth-Results.zip"]),
 ("reviewed_v3",["--version","3","--frames-per-batch","4","--seed-base","50000000",
  "--run-id","41946464299","--source","a0583f0cd2347a9e0a93bfbc873a6ce623648fb6",
  "--rate",".125","--artifact-prefix","v3-final-evaluation",
  "--model-artifact","v3-refined-frozen-model.zip","--model-summary","v3_model_summary.json",
  "--archive-name","MU-V3-Reviewed-Results.zip"]),
]
for folder,args in profiles:
    with open(folder+".log","w") as log:
        subprocess.run([sys.executable,"tools/analyze_v2.py","--artifacts",str(ROOT),"--output",folder]+args,
          stdout=log,stderr=subprocess.STDOUT,check=True)
    print("Completed independent count/configuration/model/curve audit:",folder,flush=True)
    figures=list(Path(folder).glob("*.png"));assert len(figures)==12
    for path in figures:
        with Image.open(path) as im:im.verify()
    print("All 12 plot PNG files decoded successfully:",folder,flush=True)

target=Path("reviewed_v3/training");target.mkdir(exist_ok=True)
shutil.copy2("tmp/V3-training-review.json",target/"V3-training-review.json")
shutil.copy2("V3-Reproduce.md","reviewed_v3/V3-Reproduce.md")

summary=json.loads(Path("reviewed_v3/V3-summary.json").read_text())
brief={k:summary[k] for k in ["run_id","source","network_fingerprint","payload_bits_per_user",
 "maximum_goodput_Mbps","frames_per_batch","batches_per_array","model","checks"]}
brief["arrays"]={}
for bs,info in summary["arrays"].items():
    brief["arrays"][bs]={k:v for k,v in info.items() if k!="points"}
    brief["arrays"][bs]["points20"]=[r for r in info["points"] if r["snr"]==20]
brief["LS_revision"]=summary.get("LS_revision")
Path("review_metadata.json").write_text(json.dumps(brief,indent=2))
print("REVIEW_SUMMARY_BEGIN",flush=True)
print(json.dumps(brief,ensure_ascii=False),flush=True)
print("REVIEW_SUMMARY_END",flush=True)

for bs in [32,64]:
    canvas=Image.new("RGB",(1320,1350),"white")
    for p,prec in enumerate(["MR","ZF","PZF"]):
        for j,kind in enumerate(["BER","Goodput"]):
            path=Path("reviewed_v3")/f"V3-{prec}-{kind}-BS{bs}.png"
            with Image.open(path) as opened:
                thumb=opened.convert("RGB");thumb.thumbnail((660,450))
                canvas.paste(thumb,(j*660,p*450))
    buffer=io.BytesIO();canvas.save(buffer,format="PNG",optimize=True)
    encoded=base64.b64encode(buffer.getvalue()).decode()
    print(f"QA_PREVIEW_BS{bs}_BEGIN",flush=True)
    for start in range(0,len(encoded),12000):print(encoded[start:start+12000],flush=True)
    print(f"QA_PREVIEW_BS{bs}_END",flush=True)
print("FINAL_REVIEW_PASS",flush=True)
