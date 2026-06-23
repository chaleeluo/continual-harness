#!/usr/bin/env python3
"""
HarnessEvolver — Full harness evolution for the ContinualHarness scaffold.

Subsumes PromptOptimizer and adds subagent, skill, and memory evolution.
Runs periodically (every N steps, after a minimum warmup) to analyze
recent trajectories and improve all harness components mid-episode.

Optimizations:
- Single LLM call for all 4 evolution passes (prompt, subagents, skills, memory)
- Content-hash cache to skip redundant evolution when trajectories unchanged
"""

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from agents.prompts.paths import (
    CONTINUAL_HARNESS_BASE_ORCHESTRATOR_POLICY_PATH,
    GAME_NAME,
    CONTINUAL_HARNESS_SYSTEM_PROMPT_PATH,
    resolve_repo_path,
)
from agents.utils.prompt_optimizer import PromptOptimizer

logger = logging.getLogger(__name__)

MIN_WARMUP_STEPS = 25
EARLY_PHASE_CUTOFF = 200
EARLY_FREQUENCY = 25
STABLE_FREQUENCY = 100

_ALWAYS_AVAILABLE_TOOLS = frozenset({
    "press_buttons", "complete_direct_objective", "get_game_state", "get_map_data",
    "process_memory", "process_skill", "run_skill", "run_code",
    "process_subagent", "execute_custom_subagent", "process_trajectory_history", "replan_objectives",
})


class HarnessEvolver:
    """Evolves all harness components: prompt, subagents, skills, memory.

    Optimizations over baseline:
    1. Single LLM call for all 4 evolution passes instead of 4 separate calls.
    2. Content-hash cache to skip evolution when trajectories haven't changed.
    """

    def __init__(self, vlm, run_data_manager,
                 base_prompt_path=CONTINUAL_HARNESS_BASE_ORCHESTRATOR_POLICY_PATH,
                 system_prompt_path=CONTINUAL_HARNESS_SYSTEM_PROMPT_PATH,
                 initial_prompt_override=None):
        self.prompt_optimizer = PromptOptimizer(
            vlm, run_data_manager, base_prompt_path, system_prompt_path,
            initial_prompt_override=initial_prompt_override,
        )
        self.text_vlm = self.prompt_optimizer.vlm
        self.run_manager = run_data_manager
        self.generation = 0
        self.evolution_log: List[Dict[str, Any]] = []
        self._last_trajectory_digest: str | None = None
        self._last_evolution_result: Dict[str, Any] | None = None
        logger.info("HarnessEvolver initialized (warmup=%d steps)", MIN_WARMUP_STEPS)

    def _get_memory_store(self):
        from utils.stores.memory import get_memory_store
        return get_memory_store()

    def _get_skill_store(self):
        from utils.stores.skills import get_skill_store
        return get_skill_store()

    def _get_subagent_store(self):
        from utils.stores.subagents import get_subagent_store
        return get_subagent_store()

    def should_evolve(self, current_step: int, frequency: int) -> bool:
        if current_step < MIN_WARMUP_STEPS or current_step <= 0:
            return False
        freq = EARLY_FREQUENCY if current_step <= EARLY_PHASE_CUTOFF else STABLE_FREQUENCY
        return current_step % freq == 0

    def get_current_prompt(self) -> str:
        return self.prompt_optimizer.get_current_prompt()

    def _compute_trajectory_digest(self, trajectories: List[Dict[str, Any]]) -> str:
        d = hashlib.sha256()
        for t in trajectories:
            d.update(str(t.get("step", "")).encode())
            action = t.get("action", {})
            if isinstance(action, dict):
                for tc in action.get("tool_calls", []):
                    d.update(tc.get("name", "").encode())
                    r = tc.get("result", "")
                    if isinstance(r, dict):
                        d.update(str(r.get("success", "")).encode())
                    d.update(str(r)[:50].encode())
        return d.hexdigest()

    def evolve(self, current_step: int, num_trajectory_steps: int = 50) -> Dict[str, Any]:
        logger.info("=== HarnessEvolver generation %d at step %d ===", self.generation, current_step)
        trajectories = self.prompt_optimizer.get_recent_trajectories(num_trajectory_steps)
        if not trajectories:
            logger.warning("No trajectories — skipping evolution")
            return {"skipped": True, "reason": "no_trajectories"}

        digest = self._compute_trajectory_digest(trajectories)
        if digest == self._last_trajectory_digest and self._last_evolution_result is not None:
            logger.info("Trajectory unchanged since last evolution — reusing cached result")
            self.generation += 1
            return dict(self._last_evolution_result)

        try:
            combined = self._evolve_combined(trajectories, current_step)
        except Exception as e:
            logger.error("Combined evolution failed: %s", e, exc_info=True)
            combined = {"error": str(e), "prompt": {"rewritten": False, "error": str(e)}}

        prompt_result = combined.get("prompt", {})
        if prompt_result.get("rewritten") and prompt_result.get("new_prompt"):
            self.prompt_optimizer.current_base_prompt = prompt_result["new_prompt"]
            prompt_result.pop("new_prompt", None)

        results = {
            "prompt": prompt_result,
            "subagents": combined.get("subagents", {"error": "missing"}),
            "skills": combined.get("skills", {"error": "missing"}),
            "memory": combined.get("memory", {"error": "missing"}),
        }
        self._apply_subagent_changes(results["subagents"])
        self._apply_skill_changes(results["skills"])
        self._apply_memory_changes(results["memory"])

        self.generation += 1
        self._last_trajectory_digest = digest
        self._last_evolution_result = results
        self._save_evolution_log(current_step, results)
        return results

    def _evolve_combined(self, trajectories, current_step):
        sa_ov = self._get_subagent_store().get_tree_overview()
        sk_ov = self._get_skill_store().get_tree_overview()
        me_ov = self._get_memory_store().get_tree_overview()
        summary = self.prompt_optimizer._format_trajectories_for_analysis(trajectories)
        failures = self._extract_tool_failures(trajectories)
        cur_prompt = self.prompt_optimizer.get_current_prompt()

        locs = set()
        for t in trajectories:
            pre = t.get("pre_state", {})
            loc = pre.get("location") or t.get("location")
            if loc:
                locs.add(loc)

        rc = sum(1 for t in trajectories for tc in (t.get("action", {}).get("tool_calls", []) if isinstance(t.get("action", {}), dict) else []) if tc.get("name") == "run_code")
        rs = sum(1 for t in trajectories for tc in (t.get("action", {}).get("tool_calls", []) if isinstance(t.get("action", {}), dict) else []) if tc.get("name") == "run_skill")
        aw = f"\n## ANTIPATTERN\nThe agent called run_code {rc}x but run_skill 0x. Create executable skills from run_code patterns.\n" if rc >= 3 and rs == 0 else ""

        prompt = f"""You are a harness evolution system for an AI agent playing {GAME_NAME}.
The agent has NO walkthrough or wiki access — it learns entirely through gameplay.

Your job: analyze the recent trajectories and recommend improvements across 4 areas:
prompt, subagents, skills, and memory. **Return a single JSON object** with all 4 sections.

## Current Base Prompt
{cur_prompt[:2000]}
...(truncated)...

## Current Subagent Registry
{sa_ov}

## Current Skill Library
{sk_ov}

## Current Memory Overview
{me_ov}

## Locations Visited Recently
{sorted(locs)}

## Recent Trajectories (last {len(trajectories)} steps)
{summary}

{failures}
{aw}

Available tools for subagents: {sorted(_ALWAYS_AVAILABLE_TOOLS)}

## Output Format

Respond with ONLY a JSON object (no markdown fences):

{{
  "prompt": {{
    "rewritten": true,
    "new_prompt": "Complete new base prompt markdown (if changing) or null"
  }},
  "subagents": {{
    "analysis": "summary",
    "create": [{{ "name": "string", "description": "string", "handler_type": "one_step or looping", "max_turns": 25, "available_tools": ["press_buttons", ...], "system_instructions": "...", "directive": "...", "return_condition": "Task completed" }}],
    "update": [{{ "id": "sa_XXXX", "system_instructions": "...", "directive": "..." }}],
    "retire": ["sa_XXXX"]
  }},
  "skills": {{
    "analysis": "summary",
    "add": [{{ "name": "string", "path": "category/subcategory", "description": "...", "code": "optional Python code", "effectiveness": "low|medium|high", "importance": 3 }}],
    "update": [{{ "id": "skill_XXXX", "effectiveness": "low|medium|high", "description": "...", "code": "optional replacement code" }}]
  }},
  "memory": {{
    "analysis": "summary",
    "add": [{{ "path": "category/subcategory", "title": "string", "content": "string", "importance": 3 }}],
    "update": [{{ "id": "mem_XXXX", "content": "optional updated content", "importance": 2 }}]
  }}
}}

Only include non-empty sections. If no changes needed for a section, set it to {{}}.
"""

        response = self.text_vlm.get_text_query(prompt, "HarnessEvolver_Combined")
        rec = self._parse_json_response(response)
        if rec is None:
            return {"error": "failed_to_parse_combined_response"}
        rec.setdefault("prompt", {"rewritten": False})
        rec.setdefault("subagents", {})
        rec.setdefault("skills", {})
        rec.setdefault("memory", {})
        return rec

    def _extract_tool_failures(self, trajectories):
        failures = []
        for t in trajectories:
            action = t.get("action", {})
            if not isinstance(action, dict):
                continue
            for tc in action.get("tool_calls", []):
                r = tc.get("result", "")
                rs = str(r) if r else ""
                if not rs:
                    continue
                is_fail = False
                if isinstance(r, dict):
                    is_fail = r.get("success") is False or "error" in r
                elif '"success": false' in rs.lower() or '"error"' in rs.lower():
                    is_fail = True
                if is_fail:
                    failures.append({"step": t.get("step"), "tool": tc.get("name"), "args": str(tc.get("args", {}))[:200], "error": rs[:300]})
        if not failures:
            return ""
        lines = ["## Tool Failures Detected"]
        for f in failures:
            lines.append(f"- Step {f['step']}: `{f['tool']}` args={f['args']} => {f['error']}")
        return "\n".join(lines)

    def _apply_subagent_changes(self, spec):
        if not spec or "error" in spec:
            return
        store = self._get_subagent_store()
        for s in spec.get("create", []):
            try:
                tools = [t for t in s.get("available_tools", []) if t in _ALWAYS_AVAILABLE_TOOLS] or ["press_buttons"]
                e = store.add(path=f"evolved/{s.get('name', 'unnamed').lower().replace(' ', '_')}", name=s.get("name", "Unnamed"), description=s.get("description", ""), handler_type=s.get("handler_type", "looping"), max_turns=min(s.get("max_turns", 25), 50), available_tools=tools, system_instructions=s.get("system_instructions", "")[:12000], directive=s.get("directive", "")[:12000], return_condition=s.get("return_condition", "Task completed"), importance=3, source="evolved")
                logger.info("Created evolved subagent: %s (%s)", e.id, e.name)
            except Exception as ex:
                logger.error("Failed to create subagent: %s", ex)
        for s in spec.get("update", []):
            sid = s.get("id")
            if not sid:
                continue
            try:
                fields = {k: v for k, v in s.items() if k != "id" and v}
                if fields:
                    store.update(sid, **fields)
            except Exception as ex:
                logger.error("Failed to update subagent %s: %s", sid, ex)
        for sid in spec.get("retire", []):
            try:
                store.remove(sid)
            except Exception as ex:
                logger.error("Failed to retire subagent %s: %s", sid, ex)

    def _apply_skill_changes(self, spec):
        if not spec or "error" in spec:
            return
        store = self._get_skill_store()
        for s in spec.get("add", []):
            try:
                e = store.add(path=s.get("path", "general"), name=s.get("name", "Unnamed Skill"), description=s.get("description", ""), code=s.get("code"), effectiveness=s.get("effectiveness", "medium"), importance=s.get("importance", 3), source="evolved")
                logger.info("Created evolved skill: %s (%s)", e.id, e.name)
            except Exception as ex:
                logger.error("Failed to create skill: %s", ex)
        for s in spec.get("update", []):
            sid = s.get("id")
            if not sid:
                continue
            try:
                fields = {k: v for k, v in s.items() if k != "id" and v is not None}
                if fields:
                    store.update(sid, **fields)
            except Exception as ex:
                logger.error("Failed to update skill %s: %s", sid, ex)

    def _apply_memory_changes(self, spec):
        if not spec or "error" in spec:
            return
        store = self._get_memory_store()
        for s in spec.get("add", []):
            try:
                e = store.add(path=s.get("path", "general"), title=s.get("title", "Untitled"), content=s.get("content", ""), importance=s.get("importance", 3), source="evolved")
                logger.info("Created evolved memory: %s (%s)", e.id, e.title)
            except Exception as ex:
                logger.error("Failed to create memory: %s", ex)
        for s in spec.get("update", []):
            mid = s.get("id")
            if not mid:
                continue
            try:
                fields = {k: v for k, v in s.items() if k != "id" and v is not None}
                if fields:
                    store.update(mid, **fields)
            except Exception as ex:
                logger.error("Failed to update memory %s: %s", mid, ex)

    def _evolve_prompt(self, current_step, num_trajectory_steps):
        try:
            np = self.prompt_optimizer.optimize_prompt(current_step=current_step, num_trajectory_steps=num_trajectory_steps)
            return {"rewritten": True, "length": len(np)}
        except Exception as e:
            return {"rewritten": False, "error": str(e)}

    def _evolve_subagents(self, trajectories, current_step):
        return {"delegated_to_combined": True}

    def _evolve_skills(self, trajectories, current_step):
        return {"delegated_to_combined": True}

    def _evolve_memory(self, trajectories, current_step):
        return {"delegated_to_combined": True}

    def _parse_json_response(self, response):
        text = response.strip()
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    return json.loads(text[start:end])
                except json.JSONDecodeError:
                    pass
            logger.error("Failed to parse JSON: %s", text[:200])
            return None

    def _save_evolution_log(self, current_step, results):
        from utils.data_persistence.run_data_manager import get_cache_path
        entry = {
            "generation": self.generation, "step": current_step,
            "timestamp": datetime.now().isoformat(), "mode": "combined_4in1",
            "prompt_rewritten": results.get("prompt", {}).get("rewritten", False),
            "subagents_created": len(results.get("subagents", {}).get("create", [])),
            "subagents_retired": results.get("subagents", {}).get("retire", []),
            "skills_added": len(results.get("skills", {}).get("add", [])),
            "skills_updated": len(results.get("skills", {}).get("update", [])),
            "memory_added": len(results.get("memory", {}).get("add", [])),
            "memory_updated": len(results.get("memory", {}).get("update", [])),
        }
        try:
            entry["store_counts"] = {
                "memory": len(self._get_memory_store()._entries),
                "skills": len(self._get_skill_store()._entries),
                "subagents": len(self._get_subagent_store()._entries),
            }
        except Exception:
            pass
        self.evolution_log.append(entry)
        try:
            log_file = get_cache_path("evolution_log.jsonl")
            with open(log_file, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.error("Failed to save evolution log: %s", e)


def create_harness_evolver(vlm, run_data_manager,
                           base_prompt_path=CONTINUAL_HARNESS_BASE_ORCHESTRATOR_POLICY_PATH,
                           system_prompt_path=CONTINUAL_HARNESS_SYSTEM_PROMPT_PATH,
                           initial_prompt_override=None):
    return HarnessEvolver(vlm, run_data_manager, base_prompt_path, system_prompt_path,
                          initial_prompt_override=initial_prompt_override)
