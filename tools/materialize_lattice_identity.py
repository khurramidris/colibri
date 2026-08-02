#!/usr/bin/env python3
"""Apply the reviewed Lattice execution-identity hardening as one tested commit."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_once(path: str, old: str, new: str) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{path}: expected one replacement, found {count}: {old[:80]!r}")
    target.write_text(text.replace(old, new), encoding="utf-8", newline="\n")


# Colibri context: bind the frozen plan and exact native replay argument.
replace_once(
    "lattice/colibri.py",
    '    storage_topology: dict[str, Any] = field(default_factory=dict)\n',
    '    storage_topology: dict[str, Any] = field(default_factory=dict)\n'
    '    plan_fingerprint: str = ""\n'
    '    replay_cap: int = 0\n',
)
replace_once(
    "lattice/colibri.py",
    '    qualification_overrides: dict[str, str] | None = None,\n) -> ColibriContext:\n',
    '    qualification_overrides: dict[str, str] | None = None,\n'
    '    frozen_plan: dict[str, Any] | None = None,\n) -> ColibriContext:\n',
)
replace_once(
    "lattice/colibri.py",
    '        plan = resource_plan.build_plan(str(model), context=context_length, policy="quality")\n'
    '        report = doctor_module.run_doctor(\n',
    '        computed_plan = resource_plan.build_plan(str(model), context=context_length, policy="quality")\n'
    '        if frozen_plan is not None and not isinstance(frozen_plan, dict):\n'
    '            raise LatticeError("frozen execution plan must be an object")\n'
    '        plan = frozen_plan if frozen_plan is not None else computed_plan\n'
    '        report = doctor_module.run_doctor(\n',
)
replace_once(
    "lattice/colibri.py",
    '    controlled_snapshot = qualification_environment(controlled)\n'
    '    execution_fingerprint = sha256_bytes(canonical_json({\n'
    '        "schema": 1,\n'
    '        "context": context_length,\n',
    '    controlled_snapshot = qualification_environment(controlled)\n'
    '    plan_fingerprint = sha256_bytes(canonical_json(plan))\n'
    '    replay_cap = int(plan.get("tiers", {}).get("ram", {}).get("cache_slots_per_layer", 0) or 0)\n'
    '    if replay_cap < 0:\n'
    '        raise LatticeError("execution plan has a negative replay cache cap")\n'
    '    execution_fingerprint = sha256_bytes(canonical_json({\n'
    '        "schema": 2,\n'
    '        "context": context_length,\n'
    '        "plan_fingerprint": plan_fingerprint,\n'
    '        "native_replay_arguments": [str(replay_cap)],\n',
)
replace_once(
    "lattice/colibri.py",
    '        storage_topology=topology,\n    )\n',
    '        storage_topology=topology,\n'
    '        plan_fingerprint=plan_fingerprint,\n'
    '        replay_cap=replay_cap,\n'
    '    )\n',
)
replace_once(
    "lattice/colibri.py",
    '    cap = context.plan.get("tiers", {}).get("ram", {}).get("cache_slots_per_layer", 0)\n'
    '    command = [str(context.engine), str(int(cap or 0))]\n',
    '    command = [str(context.engine), str(context.replay_cap)]\n',
)
replace_once(
    "lattice/colibri.py",
    '            "Lattice v0.1 qualification currently supports the GLM/colibri "\n',
    '            "Lattice numerical qualification currently supports the GLM/colibri "\n',
)

# Project/session/run evidence includes the frozen plan identity and argument.
replace_once(
    "lattice/experiment.py",
    '        "execution_fingerprint": context.execution_fingerprint,\n'
    '        "qualification_context": context.qualification_context,\n',
    '        "execution_fingerprint": context.execution_fingerprint,\n'
    '        "plan_fingerprint": context.plan_fingerprint,\n'
    '        "replay_cap": context.replay_cap,\n'
    '        "qualification_context": context.qualification_context,\n',
)
replace_once(
    "lattice/experiment.py",
    '        "execution_fingerprint": context.execution_fingerprint,\n'
    '    }\n'
    '    run_id = short_id',
    '        "execution_fingerprint": context.execution_fingerprint,\n'
    '        "plan_fingerprint": context.plan_fingerprint,\n'
    '        "replay_cap": context.replay_cap,\n'
    '    }\n'
    '    run_id = short_id',
)
# The first occurrence above is create_project; insert session fields separately.
needle = '            "execution_fingerprint": context.execution_fingerprint,\n            "qualification_context": context.qualification_context,\n'
replace_once(
    "lattice/experiment.py",
    needle,
    '            "execution_fingerprint": context.execution_fingerprint,\n'
    '            "plan_fingerprint": context.plan_fingerprint,\n'
    '            "replay_cap": context.replay_cap,\n'
    '            "qualification_context": context.qualification_context,\n',
)

replace_once(
    "lattice/workspace.py",
    '            "execution_fingerprint", "qualification_context",\n',
    '            "execution_fingerprint", "plan_fingerprint", "replay_cap",\n'
    '            "qualification_context",\n',
)
replace_once(
    "lattice/workspace.py",
    '        if not isinstance(data["qualification_environment"], dict):\n',
    '        if not isinstance(data.get("plan_fingerprint"), str) or len(data["plan_fingerprint"]) != 64:\n'
    '            raise LatticeError("project.json has an invalid plan_fingerprint")\n'
    '        if (isinstance(data.get("replay_cap"), bool) or not isinstance(data.get("replay_cap"), int)\n'
    '                or data["replay_cap"] < 0):\n'
    '            raise LatticeError("project.json has an invalid replay_cap")\n'
    '        if not isinstance(data["qualification_environment"], dict):\n',
)

replace_once(
    "lattice/integrity.py",
    '    if session.get("execution_fingerprint") != project.get("execution_fingerprint"):\n'
    '        raise LatticeError("session execution fingerprint does not match project")\n',
    '    if session.get("execution_fingerprint") != project.get("execution_fingerprint"):\n'
    '        raise LatticeError("session execution fingerprint does not match project")\n'
    '    if session.get("plan_fingerprint") != project.get("plan_fingerprint"):\n'
    '        raise LatticeError("session plan fingerprint does not match project")\n'
    '    if session.get("replay_cap") != project.get("replay_cap"):\n'
    '        raise LatticeError("session replay cache cap does not match project")\n',
)
replace_once(
    "lattice/integrity.py",
    '        if run.get("execution_fingerprint") != project.get("execution_fingerprint"):\n'
    '            raise LatticeError(f"execution identity mismatch in run {run.get(\'id\')}")\n',
    '        if run.get("execution_fingerprint") != project.get("execution_fingerprint"):\n'
    '            raise LatticeError(f"execution identity mismatch in run {run.get(\'id\')}")\n'
    '        if run.get("plan_fingerprint") != project.get("plan_fingerprint"):\n'
    '            raise LatticeError(f"plan identity mismatch in run {run.get(\'id\')}")\n'
    '        if run.get("replay_cap") != project.get("replay_cap"):\n'
    '            raise LatticeError(f"replay cache cap mismatch in run {run.get(\'id\')}")\n',
)

# Hash runtime files before importing them, and execute the recorded plan.
replace_once(
    "lattice/cli.py",
    'from .colibri import create_context, hardware_summary\n',
    'from .colibri import create_context, fingerprint_runtime, hardware_summary\n',
)
replace_once(
    "lattice/cli.py",
    'def _context_from_project(workspace: Workspace, deep: bool = False):\n'
    '    project = workspace.load_project()\n'
    '    return create_context(\n',
    'def _context_from_project(workspace: Workspace, deep: bool = False):\n'
    '    project = workspace.load_project()\n'
    '    repo_root = Path(project["repo_root"])\n'
    '    engine = Path(project["engine_path"])\n'
    '    current_runtime = fingerprint_runtime(repo_root / "c", repo_root / "c" / "coli", engine)\n'
    '    if current_runtime != project.get("runtime_fingerprint"):\n'
    '        raise LatticeError("runtime changed before support modules could be executed; reinitialize")\n'
    '    return create_context(\n',
)
replace_once(
    "lattice/cli.py",
    '        engine=Path(project["engine_path"]),\n'
    '        deep=deep,\n'
    '        context_length=int(project["qualification_context"]),\n'
    '        qualification_overrides=project.get("qualification_environment"),\n'
    '    )\n',
    '        engine=engine,\n'
    '        deep=deep,\n'
    '        context_length=int(project["qualification_context"]),\n'
    '        qualification_overrides=project.get("qualification_environment"),\n'
    '        frozen_plan=project.get("plan"),\n'
    '    )\n',
)
replace_once(
    "lattice/cli.py",
    '    if project.get("execution_fingerprint") != context.execution_fingerprint:\n'
    '        mismatches.append("execution environment")\n',
    '    if project.get("execution_fingerprint") != context.execution_fingerprint:\n'
    '        mismatches.append("execution environment")\n'
    '    if project.get("plan_fingerprint") != context.plan_fingerprint:\n'
    '        mismatches.append("execution plan")\n'
    '    if project.get("replay_cap") != context.replay_cap:\n'
    '        mismatches.append("native replay arguments")\n',
)

replace_once(
    "lattice/verify.py",
    'from .colibri import create_context\n',
    'from .colibri import create_context, fingerprint_runtime\n',
)
replace_once(
    "lattice/verify.py",
    '    context = create_context(\n'
    '        Path(project["repo_root"]),\n'
    '        Path(project["model_path"]),\n'
    '        engine=Path(project["engine_path"]),\n',
    '    repo_root = Path(project["repo_root"])\n'
    '    engine = Path(project["engine_path"])\n'
    '    current_runtime = fingerprint_runtime(repo_root / "c", repo_root / "c" / "coli", engine)\n'
    '    if current_runtime != project.get("runtime_fingerprint"):\n'
    '        raise LatticeError("runtime changed before support modules could be executed")\n'
    '    context = create_context(\n'
    '        repo_root,\n'
    '        Path(project["model_path"]),\n'
    '        engine=engine,\n',
)
replace_once(
    "lattice/verify.py",
    '        qualification_overrides=project.get("qualification_environment"),\n'
    '    )\n',
    '        qualification_overrides=project.get("qualification_environment"),\n'
    '        frozen_plan=project.get("plan"),\n'
    '    )\n',
)
replace_once(
    "lattice/verify.py",
    '        "execution_fingerprint": context.execution_fingerprint == project.get("execution_fingerprint"),\n'
    '        "qualification_context": context.qualification_context == project.get("qualification_context"),\n',
    '        "execution_fingerprint": context.execution_fingerprint == project.get("execution_fingerprint"),\n'
    '        "plan_fingerprint": context.plan_fingerprint == project.get("plan_fingerprint"),\n'
    '        "replay_cap": context.replay_cap == project.get("replay_cap"),\n'
    '        "qualification_context": context.qualification_context == project.get("qualification_context"),\n',
)
replace_once(
    "lattice/verify.py",
    '            "profile_execution_fingerprint": profile.get("execution_fingerprint") == context.execution_fingerprint,\n'
    '            "profile_qualification_context": profile.get("qualification_context") == context.qualification_context,\n',
    '            "profile_execution_fingerprint": profile.get("execution_fingerprint") == context.execution_fingerprint,\n'
    '            "profile_plan_fingerprint": profile.get("plan_fingerprint") == context.plan_fingerprint,\n'
    '            "profile_replay_cap": profile.get("replay_cap") == context.replay_cap,\n'
    '            "profile_qualification_context": profile.get("qualification_context") == context.qualification_context,\n',
)

replace_once(
    "lattice/recommend.py",
    '        "execution_fingerprint": project["execution_fingerprint"],\n'
    '        "qualification_context": project["qualification_context"],\n',
    '        "execution_fingerprint": project["execution_fingerprint"],\n'
    '        "plan_fingerprint": project["plan_fingerprint"],\n'
    '        "replay_cap": project["replay_cap"],\n'
    '        "qualification_context": project["qualification_context"],\n',
)

# deployment_environment must use the recorded plan too.
replace_once(
    "lattice/deploy.py",
    '        qualification_overrides=project.get("qualification_environment"),\n'
    '    )\n',
    '        qualification_overrides=project.get("qualification_environment"),\n'
    '        frozen_plan=project.get("plan"),\n'
    '    )\n',
)

# The performance pass must actually be uninstrumented.
replace_once(
    "c/colibri.c",
    '    /* Phase 1: uninstrumented performance replay. */\n'
    '    kv_alloc(m,nfull+2);\n'
    '    float *logit=step(m,full,np-1,0); free(logit);\n'
    '    m->hits=m->miss=m->ereq=m->gpu_expert_calls=0; m->hit_pin=m->hit_ecache=0; m->hit_vk=0;\n'
    '    profile_reset(m);\n'
    '    ProfBase pb; prof_base(m,&pb);\n'
    '    for(int r=0;r<MIR_REPS;r++){ atomic_store(&g_mir_bytes[r],0); atomic_store(&g_mir_nread[r],0); }\n'
    '    double t0=now_s(); int steps=0;\n'
    '    for(int i=np-1;i<nfull-1;i++){\n'
    '        double tf0=g_prof?now_s():0;\n'
    '        logit=step(m,full+i,1,i); free(logit); steps++;\n'
    '        if(g_prof){ prof_lat(now_s()-tf0); m->n_fw++; m->n_emit++; }\n'
    '    }\n'
    '    double dt=now_s()-t0, tot=m->hits+m->miss;\n'
    '    printf("REPLAY decode: %d tokens in %.3fs | %.2f tok/s | expert hit %.1f%%\\n",\n'
    '        steps,dt,steps/dt,tot?100.0*m->hits/tot:0.0);\n'
    '    profile_print(m,dt);\n'
    '    if(g_prof) prof_report(m,&pb,dt,steps,stdout);\n',
    '    /* Phase 1: disable all profiler work inside the measured loop. */\n'
    '    int saved_prof=g_prof; g_prof=0;\n'
    '    kv_alloc(m,nfull+2);\n'
    '    float *logit=step(m,full,np-1,0); free(logit);\n'
    '    m->hits=m->miss=m->ereq=m->gpu_expert_calls=0; m->hit_pin=m->hit_ecache=0; m->hit_vk=0;\n'
    '    for(int r=0;r<MIR_REPS;r++){ atomic_store(&g_mir_bytes[r],0); atomic_store(&g_mir_nread[r],0); }\n'
    '    double t0=now_s(); int steps=0;\n'
    '    for(int i=np-1;i<nfull-1;i++){\n'
    '        logit=step(m,full+i,1,i); free(logit); steps++;\n'
    '    }\n'
    '    double dt=now_s()-t0, tot=m->hits+m->miss;\n'
    '    printf("REPLAY decode: %d tokens in %.3fs | %.2f tok/s | expert hit %.1f%%\\n",\n'
    '        steps,dt,steps/dt,tot?100.0*m->hits/tot:0.0);\n'
    '    g_prof=saved_prof;\n',
)

print("execution identity hardening materialized")
