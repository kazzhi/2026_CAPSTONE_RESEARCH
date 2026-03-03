def compute_rewards(cfg, world, coverage, agent_state, t, alive_agents, newly_covered_count):
    rewards = {}
    
    # Global Shared Reward for progress
    shared_reward = newly_covered_count * cfg.reward_per_cell 

    for agent_id in alive_agents:
        s = agent_state[agent_id]
        r = shared_reward
        
        # 1. Massive Penalty for crashing (Car)
        if "car" in agent_id and not s.is_active and s.collisions > 0:
            r -= cfg.crash_penalty # e.g., -500
            
        # 2. Battery/Movement Penalty
        if "drone" in agent_id:
            # High penalty for movement encourages the drone to stay still 
            # unless the car can't reach an area.
            if s.is_active:
                move_penalty = cfg.drone_step_penalty if s.last_action != 0 else 0
                r -= move_penalty
            else:
                # If battery ran out, perhaps a small penalty for failing to finish
                if s.battery <= 0:
                    r -= cfg.battery_out_penalty 

        rewards[agent_id] = r
        
    return rewards