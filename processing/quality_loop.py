from __future__ import annotations
import argparse, json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from processing.provenance import write_json
from processing.quality_followup import CULLING_FLAGS, WINNERS, run_culling_sweep

DEFAULT_POLICY={
    "max_psnr_drop":0.5,
    "max_ssim_drop":0.01,
    "max_lpips_increase":0.02,
    "max_runtime_ratio":1.5,
    "max_vram_ratio":1.25,
    "max_artifact_regression":0.002,
    "min_artifact_improvement":0.001,
}
ARTIFACT_METRICS=("low_opacity_primitive_ratio","scale_anisotropy_above_10_ratio")

def _number(row: Mapping[str,Any], key: str) -> float | None:
    value=row.get(key)
    if isinstance(value,bool) or not isinstance(value,(int,float)): return None
    return float(value)

def _ratio(candidate: float | None, baseline: float | None) -> float | None:
    if candidate is None or baseline is None or baseline<=0: return None
    return candidate/baseline

def evaluate_candidate(baseline: Mapping[str,Any], candidate: Mapping[str,Any], policy: Mapping[str,float]=DEFAULT_POLICY) -> dict[str,Any]:
    reasons=[]
    if baseline.get("status")!="success": reasons.append("baseline_failed")
    if candidate.get("status")!="success": reasons.append("candidate_failed")
    b_psnr,c_psnr=_number(baseline,"psnr"),_number(candidate,"psnr")
    b_ssim,c_ssim=_number(baseline,"ssim"),_number(candidate,"ssim")
    b_lpips,c_lpips=_number(baseline,"lpips"),_number(candidate,"lpips")
    for key,b,c in (("psnr",b_psnr,c_psnr),("ssim",b_ssim,c_ssim),("lpips",b_lpips,c_lpips)):
        if b is None or c is None: reasons.append(f"unmeasured_{key}")
    if b_psnr is not None and c_psnr is not None and c_psnr < b_psnr-policy["max_psnr_drop"]: reasons.append("psnr_regression")
    if b_ssim is not None and c_ssim is not None and c_ssim < b_ssim-policy["max_ssim_drop"]: reasons.append("ssim_regression")
    if b_lpips is not None and c_lpips is not None and c_lpips > b_lpips+policy["max_lpips_increase"]: reasons.append("lpips_regression")
    runtime_ratio=_ratio(_number(candidate,"wall_clock_seconds"),_number(baseline,"wall_clock_seconds"))
    vram_ratio=_ratio(_number(candidate,"peak_gpu_memory_bytes"),_number(baseline,"peak_gpu_memory_bytes"))
    if runtime_ratio is not None and runtime_ratio>policy["max_runtime_ratio"]: reasons.append("runtime_regression")
    if vram_ratio is not None and vram_ratio>policy["max_vram_ratio"]: reasons.append("vram_regression")
    gains={}
    for key in ARTIFACT_METRICS:
        b,c=_number(baseline,key),_number(candidate,key)
        if b is None or c is None:
            reasons.append(f"unmeasured_{key}"); continue
        gain=b-c; gains[key]=gain
        if gain < -policy["max_artifact_regression"]: reasons.append(f"{key}_regression")
    score=sum(gains.values())
    if not gains or max(gains.values(),default=0.0)<policy["min_artifact_improvement"]: reasons.append("no_measured_artifact_improvement")
    return {"eligible":not reasons,"score":score,"artifactGains":gains,"runtimeRatio":runtime_ratio,"vramRatio":vram_ratio,"reasons":sorted(set(reasons))}

def _value_key(value: float) -> str:
    return format(float(value),".12g")

def run_bounded_culling_loop(
    data_dir: str|Path,
    source_video: str|Path,
    output_root: str|Path,
    *,
    winner: str,
    parameter: str,
    values: Sequence[float],
    iterations: int=30000,
    holdout_count: int|None=None,
    max_iterations: int=4,
    no_improvement_limit: int=2,
    policy: Mapping[str,float]=DEFAULT_POLICY,
    runner: Callable[...,dict]=run_culling_sweep,
) -> dict:
    if winner not in WINNERS: raise ValueError(f"winner must be one of {WINNERS}")
    if parameter not in CULLING_FLAGS: raise ValueError(f"unknown culling parameter: {parameter}")
    if max_iterations<1 or no_improvement_limit<1: raise ValueError("loop bounds must be positive")
    candidates=tuple(dict.fromkeys(float(v) for v in values))
    if not candidates: raise ValueError("at least one candidate value is required")
    root=Path(output_root).expanduser().resolve(); root.mkdir(parents=True,exist_ok=True)
    manifest_path=root/"quality-loop.json"
    prior={}
    if manifest_path.is_file():
        prior=json.loads(manifest_path.read_text(encoding="utf-8"))
        if prior.get("winner")!=winner or prior.get("parameter")!=parameter: raise ValueError("existing loop manifest belongs to a different search")
    attempted={_value_key(item["value"]) for item in prior.get("attempts",[]) if isinstance(item,dict) and isinstance(item.get("value"),(int,float))}
    summary={
        "schema_version":1,"experiment_type":"bounded-culling-loop","winner":winner,"parameter":parameter,
        "iterations":iterations,"max_iterations":max_iterations,"no_improvement_limit":no_improvement_limit,
        "policy":dict(policy),"dataset_id":prior.get("dataset_id"),"attempts":list(prior.get("attempts",[])),
        "last_good":prior.get("last_good",{"kind":"baseline","value":None,"score":0.0,"metrics":None}),
        "status":"running","stop_reason":None,
    }
    no_improvement=0
    remaining=[v for v in candidates if _value_key(v) not in attempted]
    remaining=remaining[:max(0,max_iterations-len(summary["attempts"]))]
    for value in remaining:
        index=len(summary["attempts"])+1
        result=runner(data_dir,source_video,root/f"iteration-{index:02d}-{_value_key(value).replace('.','p')}",winner=winner,parameter=parameter,values=[value],iterations=iterations,holdout_count=holdout_count)
        dataset_id=result.get("dataset_id") or (result.get("comparison") or {}).get("dataset_id")
        if not dataset_id: raise ValueError("sweep did not report dataset identity")
        if summary["dataset_id"] is None: summary["dataset_id"]=dataset_id
        elif summary["dataset_id"]!=dataset_id: raise ValueError("dataset identity changed inside bounded loop")
        rows=(result.get("comparison") or {}).get("results")
        if not isinstance(rows,list) or len(rows)!=2: raise ValueError("bounded loop requires baseline + one candidate comparison")
        baseline,candidate=rows
        decision=evaluate_candidate(baseline,candidate,policy)
        improved=decision["eligible"] and decision["score"]>float(summary["last_good"].get("score") or 0.0)+policy["min_artifact_improvement"]
        if improved:
            summary["last_good"]={"kind":"candidate","value":value,"score":decision["score"],"metrics":dict(candidate)}
            no_improvement=0
        else:
            no_improvement+=1
        summary["attempts"].append({"value":value,"dataset_id":dataset_id,"comparison_path":result.get("comparison_path"),"candidate":dict(candidate),"decision":decision,"adoptedAsLastGood":improved})
        write_json(manifest_path,summary)
        if no_improvement>=no_improvement_limit:
            summary["status"]="stopped"; summary["stop_reason"]="no_improvement"; break
    if summary["status"]=="running":
        if len(summary["attempts"])>=max_iterations:
            summary["status"]="stopped"; summary["stop_reason"]="max_iterations"
        elif not remaining:
            summary["status"]="stopped"; summary["stop_reason"]="candidate_space_exhausted"
        else:
            summary["status"]="complete"; summary["stop_reason"]="candidate_space_exhausted"
    summary["manifest_path"]=str(manifest_path)
    write_json(manifest_path,summary)
    return summary

def main() -> None:
    p=argparse.ArgumentParser(description="Bounded one-parameter quality loop using same-holdout culling sweeps.")
    p.add_argument("--data",required=True); p.add_argument("--source-video",required=True); p.add_argument("--output-root",required=True)
    p.add_argument("--winner",choices=WINNERS,required=True); p.add_argument("--parameter",choices=tuple(CULLING_FLAGS),required=True)
    p.add_argument("--value",action="append",type=float,required=True); p.add_argument("--iterations",type=int,default=30000); p.add_argument("--holdout-count",type=int)
    p.add_argument("--max-iterations",type=int,default=4); p.add_argument("--no-improvement-limit",type=int,default=2)
    args=p.parse_args()
    result=run_bounded_culling_loop(args.data,args.source_video,args.output_root,winner=args.winner,parameter=args.parameter,values=args.value,iterations=args.iterations,holdout_count=args.holdout_count,max_iterations=args.max_iterations,no_improvement_limit=args.no_improvement_limit)
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=="__main__": main()
