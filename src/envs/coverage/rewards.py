def compute_rewards(cfg, world, coverage, agent_state, t, alive_agents, newly_covered_count, overlap_cells):
    rewards = {}
    
    # Global Shared Reward for progress
    shared_reward = (newly_covered_count * cfg.reward_per_cell) - (overlap_cells * cfg.overlap_penalty)

    for agent_id in alive_agents:
        s = agent_state[agent_id]
        
        # 1. Massive Penalty for crashing (Car)
        r = shared_reward if s.is_active else 0.0
        
        # 1. Battery/Movement Penalties (Personal Cost)
        if s.is_active:
            if "drone" in agent_id:
                # Drones are expensive to move
                cost = cfg.drone_move_cost if s.is_moving else cfg.drone_idle_cost
                r -= cost
            else:
                # Cars are cheap to move
                r -= cfg.car_move_cost if s.is_moving else 0.0

        # 2. Critical Failure Penalties
        if "car" in agent_id and not s.is_active and s.collisions > 0:
            # This only triggers on the EXACT frame the car crashes
            r += cfg.crash_penalty 

        rewards[agent_id] = r
        
    return rewards
        
