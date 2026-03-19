if __name__ == "__main__":
    import json
    import os
    import sys
    from pathlib import Path

    PROJECT_ROOT = Path(__file__).resolve().parent
    SRC_DIR = PROJECT_ROOT / "src"
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))

    from txagent.flex_skill import FlexSkillConfig, FlexSkillTxAgent

    os.environ["MKL_THREADING_LAYER"] = "GNU"

    model_name = os.getenv("TXAGENT_MODEL", "mims-harvard/TxAgent-T1-Llama-3.1-8B")
    rag_model_name = os.getenv(
        "TXAGENT_RAG_MODEL", "mims-harvard/ToolRAG-T1-GTE-Qwen2-1.5B"
    )
    report_path = os.getenv("FLEXSKILL_REPORT_PATH", "data/flexskill_status.json")
    question = os.getenv(
        "FLEXSKILL_QUESTION",
        "Given a 50-year-old patient experiencing severe acute pain and considering the use of the newly approved medication, Journavx, how should the dosage be adjusted considering the presence of moderate hepatic impairment?",
    )

    config = FlexSkillConfig(
        raw_experience_path=os.getenv("FLEXSKILL_RAW_PATH", "data/flexskill_raw_experiences.jsonl"),
        cluster_path=os.getenv("FLEXSKILL_CLUSTER_PATH", "data/flexskill_clusters.jsonl"),
        skill_bank_path=os.getenv("FLEXSKILL_SKILL_BANK_PATH", "data/flexskill_skill_bank.jsonl"),
        online_top_k_failures=int(os.getenv("FLEXSKILL_ONLINE_TOP_K_FAILURES", "2")),
        online_top_k_skills=int(os.getenv("FLEXSKILL_ONLINE_TOP_K_SKILLS", "3")),
        cluster_min_size=int(os.getenv("FLEXSKILL_CLUSTER_MIN_SIZE", "2")),
        consolidation_interval=int(os.getenv("FLEXSKILL_CONSOLIDATION_INTERVAL", "3")),
    )

    agent = FlexSkillTxAgent(
        model_name=model_name,
        rag_model_name=rag_model_name,
        flexskill_config=config,
        skill_generator_use_llm=os.getenv("FLEXSKILL_USE_LLM_SKILL_GEN", "false").lower() == "true",
        enable_summary=False,
    )
    agent.init_model()

    response = agent.run_multistep_agent(
        question,
        temperature=float(os.getenv("TXAGENT_TEMPERATURE", "0.3")),
        max_new_tokens=int(os.getenv("TXAGENT_MAX_NEW_TOKENS", "1024")),
        max_token=int(os.getenv("TXAGENT_MAX_TOKEN", "90240")),
        call_agent=False,
        max_round=int(os.getenv("TXAGENT_MAX_ROUND", "20")),
    )

    print(f"\033[94m{response}\033[0m")

    status = agent.flexskill.build_status_snapshot()
    report_dir = os.path.dirname(report_path)
    if report_dir:
        os.makedirs(report_dir, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(status, handle, indent=2, ensure_ascii=False)
