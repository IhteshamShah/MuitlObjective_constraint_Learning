from simglucose.analysis.risk import risk_index

def mo_risk_diff_reward(BG_last_hour: list[float]) -> tuple[float, float]:
    """
    Multi-objective reward function that returns a penalty for both hypoglycemia and hyperglycemia risk index.
    If the risk index decreases, the reward is positive, and if it increases, the reward is negative.
        
    Args:
        BG_last_hour (list): A list of blood glucose levels measured in the last hour.
    Returns:
        Tuple[float, float]: A tuple containing the hypoglycemia and hyperglycemia rewards.
    """
    if len(BG_last_hour) < 2:
        return 0, 0
    else:
        LBGI_current, HBGI_current, _ = risk_index([BG_last_hour[-1]], 1)
        LBGI_prev, HBGI_prev, _ = risk_index([BG_last_hour[-2]], 1)
        LBGI_reward = LBGI_prev - LBGI_current  # Reward for reducing hypoglycemia risk
        HBGI_reward = HBGI_prev - HBGI_current  # Reward for reducing hyperglycemia risk
        return (LBGI_reward, HBGI_reward)
    

def hyper_hypo_reward(BG_last_hour: list[float]) -> tuple[float, float]:
    """
    Multi-objective reward function that returns a penalty for both hyperglycemia and hypoglycemia risk.
    The reward is calculated based on the blood glucose levels in the last hour, with a higher penalty for more extreme values. The function uses a piecewise linear approach to assign rewards, where values within the target range receive a positive reward, while values outside the range receive a negative reward that increases with the severity of the deviation.
    
    Args:
        BG_last_hour (list): A list of blood glucose levels measured in the last hour.
    Returns:
        Tuple[float, float]: A tuple containing the hyperglycemia and hypoglycemia rewards.
    """
    if len(BG_last_hour) < 2:
        return (0, 0)
    
    hyper_reward = 0
    hypo_reward = 0
    
    for BG in BG_last_hour:
        if BG < 70:
            hypo_reward += (70 - BG) / 70  # More severe hypoglycemia gets a higher penalty
        elif BG > 180:
            hyper_reward += (BG - 180) / 180  # More severe hyperglycemia gets a higher penalty
        else:
            hyper_reward += 1  # Reward for being within the target range
            hypo_reward += 1   # Reward for being within the target range
    
    return (hyper_reward, hypo_reward)