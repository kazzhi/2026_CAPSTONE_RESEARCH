def compute_rewards(cfg, world, coverage, agent_state, t, alive_agents, newly_covered_count, agent_step_stats, spawn_info):
    rewards = {}
    
    # Global Shared Reward for progress
    shared_reward = (newly_covered_count * cfg.reward_per_cell) * 0.5
    for agent_id in alive_agents:
        s = agent_state[agent_id]
        stats = agent_step_stats.get(agent_id, {"new": 0, "overlap": 0})

        discovery_multiplier = 2.0 if "car" in agent_id else 1.0
        individual_discovery = stats["new"] * (cfg.reward_per_cell * discovery_multiplier)
        overlap_cost = stats["overlap"] * cfg.overlap_penalty
        movement_cost = cfg.step_penalty if s.is_moving else 0.0
        r = shared_reward + individual_discovery + overlap_cost + movement_cost
        
        # 1. Battery/Movement Penalties (Personal Cost)
        if not s.is_active and s.collisions > 0 and stats["new"] == 0 and stats["overlap"] == 0:
                    r += cfg.crash_penalty
        
        if "drone" in agent_id and s.battery <= 0:
            r += cfg.battery_out_penalty
            
        # Spawn rewards
        num_spawns = spawn_info.get(agent_id, 0)
        r += (cfg.car_spawn_reward * num_spawns)

        rewards[agent_id] = r


        
    return rewards
        
