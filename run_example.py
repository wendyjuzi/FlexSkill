def print_optimization_summary(agent, stage):
  print(f"\n=== Optimization Summary ({stage}) ===")
  plugin = getattr(agent, "optimization_plugin", None)
  if plugin is None:
    print("optimization_enabled: False")
    print("failure_count: 0")
    print("applied_optimizations: 0")
    print("cached_suggestions: 0")
    return

  try:
    failure_stats = plugin.get_failure_statistics()
    engine_stats = plugin.get_status().get("engine_stats", {})
    print("optimization_enabled: True")
    print(f"failure_count: {failure_stats.get('total_failures', 0)}")
    print(f"applied_optimizations: {engine_stats.get('applied_optimizations', 0)}")
    print(f"cached_suggestions: {engine_stats.get('cached_optimizations', 0)}")
  except Exception as e:
    print(f"optimization_summary_error: {e}")

if __name__ == '__main__':
  from txagent import TxAgent
  import os
  os.environ["MKL_THREADING_LAYER"] = "GNU"
  
  
  model_name = 'mims-harvard/TxAgent-T1-Llama-3.1-8B'
  rag_model_name = 'mims-harvard/ToolRAG-T1-GTE-Qwen2-1.5B'
  multiagent = False
  max_round = 20
  init_rag_num = 0
  step_rag_num = 10
  
  agent = TxAgent(model_name,
                  rag_model_name,
                  enable_summary=False)
  agent.init_model()
  print_optimization_summary(agent, "after_init")
  
  question = "Given a 50-year-old patient experiencing severe acute pain and considering the use of the newly approved medication, Journavx, how should the dosage be adjusted considering the presence of moderate hepatic impairment?"
  
  response = agent.run_multistep_agent(
      question,
      temperature=0.3,
      max_new_tokens=1024,
      max_token=90240,
      call_agent=multiagent,
      max_round=max_round)
  print_optimization_summary(agent, "after_run")
  
  print(f"\033[94m{response}\033[0m")
