"""m7 - USD stage + memory-wake predicate resolver.

The "missing seam" from the proposal: USD composes authored opinions, but does
not itself notice that unrelated attributes satisfy a condition and wake a
memory. This module IS that seam. USD 26.08 holds the world-model as a non-3D
data layer (custom attrs + relations, zero geometry); resolve_wake() evaluates
the predicate over the composed stage and returns wake/dormant.

Nothing here reaches J-space. It decides WHETHER a memory is delivered.
Delivery -- turning a woken memory into echo-guarded prompt text -- happens in
run_aprime.py (build_prompt), the decision-probe experiment this module feeds.
"""
from __future__ import annotations
from dataclasses import dataclass
from pxr import Usd, Sdf


@dataclass(frozen=True)
class WorldModel:
    """One authored configuration. `affects` is the counterfactual lever."""
    task_target: str          # e.g. "/Systems/Payments"
    task_intent: str          # e.g. "restoreService"
    obs_subject: str
    obs_status: str
    obs_error_rate: float
    evidence_affects: str     # flip THIS for the counterfactual
    evidence_change: str      # e.g. "connectionPool"
    evidence_correlated: bool


def author_stage(wm: WorldModel) -> Usd.Stage:
    """Compose the world-model as a USD 26.08 non-3D data layer.

    Task / Observations / Evidence / Memories / Policy as custom opinions.
    This is real USD composition -- the facts are independently authored prims,
    exactly as the proposal specifies.
    """
    stage = Usd.Stage.CreateInMemory()

    task = stage.DefinePrim("/Task", "Scope")
    task.CreateAttribute("intent", Sdf.ValueTypeNames.Token, True).Set(wm.task_intent)
    task.CreateRelationship("target", True).SetTargets([Sdf.Path(wm.task_target)])

    obs = stage.DefinePrim("/Observations/System", "Scope")
    obs.CreateRelationship("subject", True).SetTargets([Sdf.Path(wm.obs_subject)])
    obs.CreateAttribute("status", Sdf.ValueTypeNames.Token, True).Set(wm.obs_status)
    obs.CreateAttribute("errorRate", Sdf.ValueTypeNames.Float, True).Set(wm.obs_error_rate)

    ev = stage.DefinePrim("/Evidence/LatestDeployment", "Scope")
    ev.CreateRelationship("affects", True).SetTargets([Sdf.Path(wm.evidence_affects)])
    ev.CreateAttribute("changeType", Sdf.ValueTypeNames.Token, True).Set(wm.evidence_change)
    ev.CreateAttribute("temporallyCorrelated", Sdf.ValueTypeNames.Bool, True).Set(wm.evidence_correlated)

    # Policy floor -- authored, but NOT enforced here (enforced in tool gateway)
    pol = stage.DefinePrim("/Policy", "Scope")
    pol.CreateAttribute("allowProductionMutation", Sdf.ValueTypeNames.Bool, True).Set(False)

    return stage


def _rel_target(prim: Usd.Prim, rel_name: str) -> str:
    rel = prim.GetRelationship(rel_name)
    t = rel.GetTargets()
    return str(t[0]) if t else ""


def resolve_wake(stage: Usd.Stage) -> bool:
    """THE PREDICATE. Reads the composed stage, returns wake/dormant.

    Wakes exactly when the three facts form a recognizable situation: task,
    observation, and evidence all concern the SAME system, the situation is
    degraded past threshold, and the change is a connection-pool deployment.
    This is a modeled configuration, NOT the label 'incident'.
    """
    task = stage.GetPrimAtPath("/Task")
    obs = stage.GetPrimAtPath("/Observations/System")
    ev = stage.GetPrimAtPath("/Evidence/LatestDeployment")

    target = _rel_target(task, "target")
    return (
        target == _rel_target(obs, "subject")
        and target == _rel_target(ev, "affects")           # <- counterfactual breaks HERE
        and task.GetAttribute("intent").Get() == "restoreService"
        and obs.GetAttribute("status").Get() == "degraded"
        and (obs.GetAttribute("errorRate").Get() or 0.0) >= 0.30
        and ev.GetAttribute("changeType").Get() == "connectionPool"
        and bool(ev.GetAttribute("temporallyCorrelated").Get())
    )


# ---- the two conditions, authored ----
ALIGNED = WorldModel(
    task_target="/Systems/Payments", task_intent="restoreService",
    obs_subject="/Systems/Payments", obs_status="degraded", obs_error_rate=0.42,
    evidence_affects="/Systems/Payments", evidence_change="connectionPool",
    evidence_correlated=True,
)
# ONE relationship changed. Everything else identical.
COUNTERFACTUAL = WorldModel(
    task_target="/Systems/Payments", task_intent="restoreService",
    obs_subject="/Systems/Payments", obs_status="degraded", obs_error_rate=0.42,
    evidence_affects="/Systems/Search", evidence_change="connectionPool",   # <- the flip
    evidence_correlated=True,
)


if __name__ == "__main__":
    for name, wm in (("ALIGNED", ALIGNED), ("COUNTERFACTUAL", COUNTERFACTUAL)):
        st = author_stage(wm)
        woke = resolve_wake(st)
        print(f"{name:<16} predicate -> {'WAKE' if woke else 'dormant'}")
        print(f"                 evidence.affects = {wm.evidence_affects}")
    print()
    print("EXPECTED: ALIGNED wakes, COUNTERFACTUAL dormant (one relation flipped)")
