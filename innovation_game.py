import numpy as np
import random
from enum import Enum
from typing import Dict, Any, Tuple, Optional, List, Union
import json
import os
import pandas as pd
from google import genai
from google.genai.errors import APIError
from io import StringIO

# ==============================================================================
# 0) MODEL ENUMERATIONS AND CONSTANTS
# ==============================================================================

class Strategy(Enum):
    HIGH_TEMPO = 0
    ECONOMIC_WARFARE = 1
    WAR = 2

# The 8 mutually exclusive termination conditions
class Outcome(Enum):
    # War Initiated Outcomes (W1-W4) - Game ends immediately when WAR is chosen
    CHALLENGER_WAR_DESPERATION = "W1: War Initiated (Challenger/Desp.)"
    CHALLENGER_WAR_STRENGTH = "W2: War Initiated (Challenger/Strength)"
    ADVERSARY_WAR_DESPERATION = "W3: War Initiated (Adversary/Desp.)"
    ADVERSARY_WAR_STRENGTH = "W4: War Initiated (Adversary/Strength)"
    
    # Dominance Outcomes (D1-D2) - Structural win/loss
    CHALLENGER_DOMINANCE = "D1: Challenger Achieves Dominance"
    ADVERSARY_DOMINANCE = "D2: Adversary Achieves Dominance"
    
    # Exogenous / Stalemate Outcomes (C1, S1)
    GLOBAL_COLLAPSE = "C1: Global System Collapse"
    DYNAMIC_EQUILIBRIUM = "S1: Dynamic Equilibrium (Stalemate)"

class DistributionType(Enum):
    GAMMA = 0        # High-Impact Disruptive (Positive Skew, High Variance)
    GAUSSIAN_T = 1   # Steady/Incremental (Tighter, Consistent Gains)

# Base payoffs for the Challenger's strategies
PAYOFFS = {
    Strategy.HIGH_TEMPO: 50.0,
    Strategy.ECONOMIC_WARFARE: 35.0,
    Strategy.WAR: -20.0  # Challenger's Utility Shock for initiating War
}

# --- NEW: QUALITATIVE VOLATILITY MAPPING (FOR CHALLENGER) ---
# Maps a qualitative user choice to specific numerical parameters for the distributions
VOLATILITY_PROFILES = {
    "STEADY": {
        'dist_type': DistributionType.GAUSSIAN_T,
        'noise_params': {'mu': 2.0, 'sigma': 1.0, 'offset': 2.0} # Tighter, consistent
    },
    "VOLATILE": {
        'dist_type': DistributionType.GAUSSIAN_T,
        'noise_params': {'mu': 3.0, 'sigma': 3.0, 'offset': 3.0} # Wider Gaussian, higher variance
    },
    "DISRUPTIVE": {
        'dist_type': DistributionType.GAMMA,
        'noise_params': {'shape': 2.5, 'scale': 2.5, 'offset': 6.0} # Skewed, high impact
    },
    "IMPROVISATIONAL": { # Maps to the "Ukraine_Style" high-risk/high-skew profile
        'dist_type': DistributionType.GAMMA,
        'noise_params': {'shape': 1.5, 'scale': 4.0, 'offset': 6.0} # Even higher potential for extreme shocks
    }
}
PROFILE_KEYS = {1: 'STEADY', 2: 'VOLATILE', 3: 'DISRUPTIVE', 4: 'IMPROVISATIONAL'}

# --- SYSTEM PROFILE ASYMMETRY (Model Constants) ---
PROFILE_DEFAULTS = {
    # Challenger Profiles default to a specific Volatility Profile
    'US_STYLE': {
        'vol_key': 'DISRUPTIVE',
        'gamma_base': 0.025,    # Higher Friction (Bureaucracy/Cost)
        'p_war_max': 0.12,      # Higher Risk Tolerance
    },
    'CHINA_STYLE': {
        'vol_key': 'STEADY', 
        'gamma_base': 0.010,    # Lower Friction (Centralization/Efficiency)
        'p_war_max': 0.08,      # Lower Risk Tolerance
    },
    'UKRAINE_STYLE': {
        'vol_key': 'IMPROVISATIONAL', 
        'gamma_base': 0.005,    # Very Low Friction (Survival priority, no bureaucracy)
        'p_war_max': 0.25,      # Extremely High Risk Tolerance (Existential Threat)
    }
}
PROFILE_NAMES = {1: 'US_STYLE', 2: 'CHINA_STYLE', 3: 'UKRAINE_STYLE'}


# --- NEW DESCRIPTIVE CONSTANTS FOR UX ---
CONTEXT_TEXT = {
    'initial_gap': {
        'desc': "This value sets the starting competitive advantage or deficit (ΔU) for the Challenger.",
        'meaning': "A positive number means the Challenger starts with an edge; a negative number means a deficit.",
        'scale': "Scale runs from -100 (Terminal Deficit) to +100 (Terminal Advantage). -15.0 is a common starting deficit."
    },
    'gamma_base': {
        'desc': "This is the systemic structural friction (γ_base) opposing utility growth. It models internal costs, bureaucracy, or resistance.",
        'meaning': "Higher γ_base means the system is less efficient, and Challenger's utility decays faster relative to the friction multiplier (λ_t).",
        'scale': "Scale runs from 0.005 (Very Low Friction) to 0.050 (Very High Friction)."
    },
    'p_war_max': {
        'desc': "This is the maximum probability (P_WarMax) the Challenger will choose the high-risk 'War' strategy.",
        'meaning': "It models the Challenger's inherent risk tolerance. A higher value means the Challenger is more likely to pivot to War when under pressure.",
        'scale': "Scale runs from 0.02 (Very Risk-Averse) to 0.25 (Extremely Risk-Tolerant)."
    },
    'v_h_type': {
        'desc': "This defines the volatility profile (V_H Type) of the Challenger's innovation gains.",
        'meaning': "STEADY/VOLATILE (Gaussian) = Smaller, consistent gains. DISRUPTIVE/IMPROVISATIONAL (Gamma) = Larger, less frequent, high-impact gains/losses.",
        'scale': "Choose the distribution that best fits the Challenger's R&D philosophy."
    },
    'iterations': {
        'desc': "The number of full simulations (N) the model will run for this scenario.",
        'meaning': "More iterations increase the statistical confidence in the final probabilistic outcomes.",
        'scale': "10,000 runs provides high fidelity."
    }
}


# ==============================================================================
# 1) STOCHASTIC DISTRIBUTION HELPERS 
# ==============================================================================
def get_innovation_noise(dist_type: DistributionType, noise_params: Dict[str, float]) -> float:
    # Generates stochastic noise based on the chosen innovation profile (V_H) and specific parameters.
    
    if dist_type == DistributionType.GAMMA:
        # High Variance/Disruptive: Can lead to large positive or negative shocks
        return np.random.gamma(shape=noise_params['shape'], scale=noise_params['scale']) - noise_params['offset']
    
    elif dist_type == DistributionType.GAUSSIAN_T:
        # Tighter/Consistent: Smaller variance around the mean
        mu, sigma = noise_params['mu'], noise_params['sigma']
        noise = np.random.normal(mu, sigma)
        return max(0.0, noise) - noise_params['offset']
        
    return 0.0

def get_stochastic_friction_multiplier() -> float:
    # Generates the effective structural friction multiplier (lambda_t) centered around 1.0.
    mu, sigma = 1.0, 0.20
    multiplier = np.random.normal(mu, sigma)
    return min(2.0, max(0.0, multiplier))

def get_irrationality_noise() -> float:
    # Small, random perturbation for War choice, modeling human error/irrationality.
    return random.uniform(-0.01, 0.01)

# ==============================================================================
# 2) MODEL KERNEL: DYNAMIC INNOVATION MODEL (Dual-Actor Asymmetric)
# ==============================================================================

class DynamicInnovationModel:
    def __init__(self, params: Dict[str, Any], log_path: bool = False):
        self.U = params['initial_gap']
        self.SII = 0.0
        self.CII = 0.0
        self.params = params
        self.outcome = None
        self.rounds = 0
        self.log_path = log_path
        self.path_log: List[float] = [] 
        self.final_CII = 0.0
        self.final_SII = 0.0
        
        self.C_PWarMax = params['C_p_war_max']
        self.A_GammaBase = params['A_gamma_base']
        
        # New: Store the parameters needed for the distribution function
        self.C_VHDist = params['C_v_h_type']
        self.C_NoiseParams = params['C_noise_params']
        
        self.P_CAT_BASE = 0.0001 

    def _get_war_choice_prob(self, p_max: float, actor_U: float) -> float:
        """Calculates P(War Choice) based on Desperation and adds Noise."""
        U_collapse = self.params['u_collapse']
        U_ref = -50.0
        
        if actor_U > U_ref:
            p_desperation = p_max * 0.1
        else:
            desperation_scale = (U_ref - actor_U) / (U_ref - U_collapse)
            desperation_scale = max(0.0, min(1.0, desperation_scale))
            p_desperation = p_max * (0.1 + 0.9 * desperation_scale**2)

        p_desperation += get_irrationality_noise()
        
        return max(0.0, min(p_max, p_desperation))

    def _check_war_initiation(self) -> Tuple[bool, Optional[Outcome]]:
        """Checks if either Challenger or Adversary initiates War."""
        C_P_Choice = self._get_war_choice_prob(self.C_PWarMax, self.U)
        if random.random() < C_P_Choice:
            if self.U < -50.0:
                return True, Outcome.CHALLENGER_WAR_DESPERATION
            else:
                return True, Outcome.CHALLENGER_WAR_STRENGTH

        A_PWarMax = self.params.get('A_p_war_max', 0.10) 
        
        if self.CII > 25.0: 
            A_P_Desperation = A_PWarMax * (self.CII / 100.0)
            if random.random() < A_P_Desperation:
                 return True, Outcome.ADVERSARY_WAR_DESPERATION
        
        if self.SII > 50.0: 
            A_P_Strength = A_PWarMax * 0.5
            if random.random() < A_P_Strength:
                return True, Outcome.ADVERSARY_WAR_STRENGTH

        return False, None

    def _check_global_collapse(self) -> bool:
        """Checks for the rare, exogenous Global System Collapse (C1)."""
        P_Catastrophe = self.P_CAT_BASE
        if self.U < -50.0 or self.SII > 50.0:
            P_Catastrophe *= 10

        if random.random() < P_Catastrophe:
            self.outcome = Outcome.GLOBAL_COLLAPSE
            return True
        return False
        
    def _apply_strategies(self, S_t: Strategy):
        """Applies the effects of the chosen strategy (currently always Challenger's)."""
        P_base = PAYOFFS[S_t]
        
        # MODIFIED: Pass both the type and the specific parameters
        epsilon_S = get_innovation_noise(self.C_VHDist, self.C_NoiseParams)
        
        lambda_t = get_stochastic_friction_multiplier()
        gamma_base = self.A_GammaBase
        decay_rate = 1.0 - (lambda_t * gamma_base) 
        
        if S_t == Strategy.HIGH_TEMPO:
            spill_rate = random.uniform(0.01, 0.05) 
            self.SII += epsilon_S * 0.002 * spill_rate 
        elif S_t == Strategy.ECONOMIC_WARFARE:
            EW_shock = random.uniform(5.0, 15.0)
            epsilon_S -= EW_shock * 0.5 
            decay_rate = 1.0 - (lambda_t * (gamma_base * 1.5)) 

        self.U = (self.U * decay_rate) + P_base + epsilon_S
        
        self.SII += (self.A_GammaBase * 10.0)
        self.SII -= P_base * 0.005
        
        self.CII += P_base * 0.005
        # The 'magic number' is replaced by a value based on the chosen complexity/risk
        complexity_shock = 0.0 
        if self.C_VHDist == DistributionType.GAMMA:
             complexity_shock = 5.0 
        elif self.C_VHDist == DistributionType.GAUSSIAN_T:
             complexity_shock = 3.0
             
        self.CII -= complexity_shock 
        
        self.SII = max(0.0, self.SII)
        self.CII = max(0.0, self.CII)

    def _check_decisive_outcomes(self):
        """Checks for Dominance and Stalemate (D1, D2, S1)."""
        if self.U <= self.params['u_collapse'] or self.SII >= self.params['s_thresh']:
            self.outcome = Outcome.ADVERSARY_DOMINANCE
            return
            
        if self.CII >= self.params['c_thresh']:
            self.outcome = Outcome.CHALLENGER_DOMINANCE
            return

        if self.rounds >= self.params['max_rounds']:
            self.outcome = Outcome.DYNAMIC_EQUILIBRIUM
            return

    def run_round(self):
        self.rounds += 1
        
        if self.log_path:
            self.path_log.append(self.U)
            
        if self._check_global_collapse():
            return
            
        is_war, war_outcome = self._check_war_initiation()
        if is_war:
            self.outcome = war_outcome
            self.U += PAYOFFS[Strategy.WAR]
            if self.log_path:
                self.path_log.append(self.U) 
            return
        
        S_t = Strategy.HIGH_TEMPO 
        if random.random() < 0.15:
             S_t = Strategy.ECONOMIC_WARFARE
        
        self._apply_strategies(S_t)
        
        self._check_decisive_outcomes()

    def simulate(self) -> Tuple[Optional[Outcome], float, float, List[float]]:
        if self.log_path:
            self.path_log.append(self.U)

        while not self.outcome and self.rounds < self.params['max_rounds']:
            self.run_round()
            
        if not self.outcome:
            self.outcome = Outcome.DYNAMIC_EQUILIBRIUM
            
        if self.log_path:
            while len(self.path_log) < self.params['max_rounds'] + 1:
                self.path_log.append(self.path_log[-1])
        
        self.final_CII = self.CII
        self.final_SII = self.SII
        
        return self.outcome, self.final_CII, self.final_SII, self.path_log


# ==============================================================================
# 3) UX WRAPPER: INPUT, BATCH EXECUTION, AND DATA HANDLING (UPDATED INPUTS)
# ==============================================================================

def get_profile_params(profile_key: str) -> Dict[str, Any]:
    """Helper to retrieve parameters based on profile key."""
    profile_data = PROFILE_DEFAULTS.get(profile_key, PROFILE_DEFAULTS['US_STYLE']).copy()
    
    # NEW: Inject the specific distribution parameters from the VOLATILITY_PROFILES
    vol_key = profile_data['vol_key']
    vol_data = VOLATILITY_PROFILES.get(vol_key)
    
    profile_data['C_v_h_type'] = vol_data['dist_type']
    profile_data['C_noise_params'] = vol_data['noise_params']
    
    return profile_data

def matchup_selection_screen() -> Dict[str, Any]:
    """Handles initial selection of Challenger and Adversary System Profiles."""
    print("\n=======================================================")
    print("     DYNAMIC INNOVATION WAR GAME: SYSTEM SELECTION")
    print("=======================================================")
    print("Define the Systems (Select the inherent characteristics):")
    print("1: Western/US Style (Disruptive, High Friction)")
    print("2: Chinese Style (Steady, Low Friction)")
    print("3: Ukraine Style (Improvisational, Extreme Risk)")
    print("-------------------------------------------------------")
    
    # 1. Challenger Selection
    while True:
        try:
            c_choice = int(input("Select Challenger Profile (1-3): "))
            if c_choice in PROFILE_NAMES:
                c_profile_key = PROFILE_NAMES[c_choice]
                c_params = get_profile_params(c_profile_key)
                break
            else:
                print("Invalid choice. Must be 1, 2, or 3.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    # 2. Adversary Selection
    while True:
        try:
            a_choice = int(input("Select Adversary Profile (1-3): "))
            if a_choice in PROFILE_NAMES:
                a_profile_key = PROFILE_NAMES[a_choice]
                a_params = get_profile_params(a_profile_key)
                break
            else:
                print("Invalid choice. Must be 1, 2, or 3.")
        except ValueError:
            print("Invalid input. Please enter a number.")
            
    print(f"\nMatchup Set: Challenger ({c_profile_key}) vs. Adversary ({a_profile_key})")
    
    return {
        # These parameters are now derived from the profile key via get_profile_params
        'C_v_h_type': c_params['C_v_h_type'], 
        'C_noise_params': c_params['C_noise_params'], 
        'C_p_war_max': c_params['p_war_max'],
        'A_gamma_base': a_params['gamma_base'],
        'A_p_war_max': a_params['p_war_max'],
        'C_profile_name': c_profile_key,
        'A_profile_name': a_profile_key
    }

def run_simulation_batch(params: Dict[str, Any], path_sample_size: int = 50) -> Tuple[Dict[str, float], List[Dict[str, Any]], float, float, float]:
    """
    Runs N iterations and aggregates results, now also calculating 
    the average final CII, SII, and the War Choice Desperation Ratio.
    Returns: (final_results, dynamic_paths, avg_cii, avg_sii, war_ratio)
    """
    
    iterations = params['iterations']
    results = {o.name: 0 for o in Outcome}
    dynamic_paths = []
    
    total_final_CII = 0.0
    total_final_SII = 0.0
    total_w1 = 0 # Challenger Desperation
    total_w2 = 0 # Challenger Strength
    
    for i in range(iterations):
        log_path = i < path_sample_size
        model = DynamicInnovationModel(params, log_path=log_path)
        
        outcome, final_cii, final_sii, path_log = model.simulate()
        
        if outcome:
            results[outcome.name] += 1
            total_final_CII += final_cii
            total_final_SII += final_sii
            
            if outcome == Outcome.CHALLENGER_WAR_DESPERATION:
                total_w1 += 1
            elif outcome == Outcome.CHALLENGER_WAR_STRENGTH:
                total_w2 += 1
            
        if log_path and path_log:
            dynamic_paths.append({
                'id': i + 1,
                'outcome': outcome.name,
                'path': [float(u) for u in path_log]
            })
            
    total = sum(results.values())
    final_results = {name: (count / total) * 100 for name, count in results.items()}
    
    avg_cii = total_final_CII / iterations
    avg_sii = total_final_SII / iterations
    
    war_total = total_w1 + total_w2
    war_ratio = total_w1 / war_total if war_total > 0 else 0.0 # Ratio of desperation-initiated wars to all challenger-initiated wars
    
    return final_results, dynamic_paths, avg_cii, avg_sii, war_ratio

# Input Handlers 
def get_float(prompt, default):
    while True:
        try:
            user_input = input(f"   > {prompt} (Default: {default}): ")
            if not user_input:
                return default
            return float(user_input)
        except ValueError:
            print("Invalid input. Please enter a number.")

def get_int(prompt, default):
    while True:
        try:
            user_input = input(f"   > {prompt} (Default: {default}): ")
            if not user_input:
                return default
            return int(user_input)
        except ValueError:
            print("Invalid input. Please enter an integer.")

# get_dist now maps a user choice to specific noise parameters
def get_dist(prompt: str, default_key: str) -> Tuple[DistributionType, Dict[str, float]]:
    """Prompts for a qualitative volatility profile and returns the DistributionType and its parameters."""
    
    profile_data = VOLATILITY_PROFILES[default_key]
    default_str = default_key
    
    print("\n[CHALLENGER VOLATILITY PROFILES]")
    print("1: STEADY (Consistent, Low Variance)")
    print("2: VOLATILE (Wider Gaussian, Higher Variance)")
    print("3: DISRUPTIVE (Gamma, Skewed, High-Impact)")
    print("4: IMPROVISATIONAL (Extreme Gamma, High Shock Potential)")
    
    while True:
        v_h_input = input(f"   > {prompt} [1-4] (Default: {default_str}): ")
        
        if not v_h_input:
            return profile_data['dist_type'], profile_data['noise_params']
        
        try:
            choice = int(v_h_input)
            if choice in PROFILE_KEYS:
                chosen_key = PROFILE_KEYS[choice]
                chosen_data = VOLATILITY_PROFILES[chosen_key]
                print(f"   Selected: {chosen_key} ({chosen_data['dist_type'].name})")
                return chosen_data['dist_type'], chosen_data['noise_params']
            else:
                print("Invalid choice. Please select 1, 2, 3, or 4.")
        except ValueError:
            print("Invalid input. Please enter a number.")


def get_llm_style() -> int:
    """Prompts user for desired LLM analysis style."""
    print("\n\n-- LLM REPORTING STYLE SELECTION --")
    print("1: Default (Policy Staffer - Concise & Actionable)")
    print("2: Strategic Analyst (Long-term, Structural Focus)")
    print("3: Executive Summary (Brief, High-Level Bullet Points)")
    
    while True:
        try:
            choice = input("Select LLM Style (1, 2, or 3, Default: 1): ")
            if not choice:
                return 1
            choice_int = int(choice)
            if choice_int in [1, 2, 3]:
                return choice_int
            else:
                print("Invalid choice. Must be 1, 2, or 3.")
        except ValueError:
            print("Invalid input. Please enter a number.")

def save_scenario_data(params: Dict[str, Any], results: Dict[str, float], dynamic_paths: List[Dict[str, Any]]):
    """Saves scenario results and dynamic paths to a structured JSON file."""
    
    output_dir = "simulation_data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Prepare params for JSON serialization (handling Enums and complex types)
    serializable_params = {}
    for k, v in params.items():
        if isinstance(v, (Enum, DistributionType)):
            serializable_params[k] = v.name
        elif k == 'C_noise_params': # Serialize the noise parameter dictionary
            serializable_params[k] = v
        else:
            serializable_params[k] = v

    data_to_save = {
        'scenario_params': serializable_params,
        'probabilistic_results': results,
        'dynamic_paths_sample': dynamic_paths
    }

    c_name = params['C_profile_name']
    a_name = params['A_profile_name']
    delta_u = f"DU_{params['initial_gap']:.1f}"
    
    filename = os.path.join(output_dir, f"{c_name}_vs_{a_name}_{delta_u}_N{params['iterations']}.json")
    
    try:
        with open(filename, 'w') as f:
            json.dump(data_to_save, f, indent=4)
        print(f"\n✅ Data saved successfully for plotting. File: {filename}")
    except Exception as e:
        print(f"\n❌ ERROR: Could not save data to JSON file: {e}")


def run_interactive_mode(context_defaults: Dict[str, Any]):
    """Handles terminal user input for scenario calibration (Mode 1), with enhanced UX."""
    
    def print_context(key):
        ctx = CONTEXT_TEXT[key]
        print(f"\n[SETTING: {key.replace('_', ' ').title()}]")
        print(f"  Description: {ctx['desc']}")
        print(f"  Meaning: {ctx['meaning']}")
        print(f"  Scale/Range: {ctx['scale']}")

    print("\n--- MODE 1: INTERACTIVE SCENARIO SETUP (Contextualized Defaults) ---")
    
    print("\n\n-- 1. COMPETITIVE STARTING CONDITIONS --")
    print_context('initial_gap')
    initial_gap = get_float("Initial Competitive Gap (ΔU_initial)", -15.0)
    
    print("\n\n-- 2. SYSTEMIC CONSTRAINTS & BEHAVIOR (Defaults from Profiles) --")
    print_context('gamma_base')
    gamma_base_input = get_float("Adversary Structural Friction (γ_base)", context_defaults['A_gamma_base'])
    
    print_context('p_war_max')
    p_war_max_input = get_float("Challenger Risk Tolerance (P_WarMax)", context_defaults['C_p_war_max'])
    
    print_context('v_h_type')
    
    # Get the default key from the system profile for the UX prompt
    default_vol_key = PROFILE_DEFAULTS[context_defaults['C_profile_name']]['vol_key']
    
    # UPDATED CALL: This returns the distribution type AND the noise parameters
    v_h_type_input, noise_params_input = get_dist(
        "Challenger Innovation Volatility (V_H Type)", 
        PROFILE_DEFAULTS[context_defaults['C_profile_name']]['vol_key']
    )

    print("\n\n-- 3. SIMULATION PARAMETERS --")
    print_context('iterations')
    iterations_input = get_int("Iterations (N, Total runs)", 10000)
    
    params = {
        'initial_gap': initial_gap, 
        'iterations': iterations_input, 
        # UPDATED: Pass both the type and the specific parameters
        'C_v_h_type': v_h_type_input,
        'C_noise_params': noise_params_input, 
        'C_p_war_max': p_war_max_input,
        'A_gamma_base': gamma_base_input,
        'A_p_war_max': context_defaults['A_p_war_max'], 
        'max_rounds': 100, 
        'u_collapse': -100.0, 
        's_thresh': 100.0, 
        'c_thresh': 100.0, 
        'C_profile_name': context_defaults['C_profile_name'],
        'A_profile_name': context_defaults['A_profile_name']
    }
    
    llm_style = get_llm_style()

    print("\n--- RUNNING INTERACTIVE SIMULATION ---")
    results, dynamic_paths, avg_cii, avg_sii, war_ratio = run_simulation_batch(params)
    
    # Mode 1: Individual Interpretation
    mode_context = "Interactive Scenario Deep Dive. Focus on immediate policy levers and strategic options for the Challenger."
    
    generate_summary_report(params, results, avg_cii, avg_sii, war_ratio, mode_context=mode_context, llm_style=llm_style)
    save_scenario_data(params, results, dynamic_paths)
    
    print("\n--- SCENARIO COMPLETE ---")

# get_base_params now explicitly sets the two noise parameters
def get_base_params(context_defaults: Dict[str, Any], initial_gap: float, C_p_war_max: float, iterations: int = 5000) -> Dict[str, Any]:
    """Generates the common fixed parameters for batch runs."""
    return {
        'initial_gap': initial_gap, 
        'C_p_war_max': C_p_war_max,
        'iterations': iterations, 
        'max_rounds': 100, 
        'u_collapse': -100.0, 
        's_thresh': 100.0, 
        'c_thresh': 100.0, 
        'C_profile_name': context_defaults['C_profile_name'],
        'A_profile_name': context_defaults['A_profile_name'],
        'A_p_war_max': context_defaults['A_p_war_max'],
        'A_gamma_base': context_defaults['A_gamma_base'],
        'C_v_h_type': context_defaults['C_v_h_type'],
        'C_noise_params': context_defaults['C_noise_params'] # NEW
    }

def run_extreme_batch(context_defaults: Dict[str, Any]):
    """
    Mode 2: Runs a batch of scenarios and sends results for single collective analysis.
    """
    print("\n--- MODE 2: EXTREME EDGES BATCH ANALYSIS (5000 Iterations per Scenario) ---")
    
    # --- START OF NEW INPUT STEP FOR VOLATILITY PROFILE ---
    print("\n--- BATCH-WIDE VOLATILITY PROFILE SETUP ---")
    default_key = PROFILE_DEFAULTS[context_defaults['C_profile_name']]['vol_key']
    v_h_type, noise_params = get_dist(
        "Challenger Innovation Volatility (V_H Type) for ALL batch scenarios", 
        default_key
    )
    # Overwrite the context_defaults with the user's specific choice for the batch
    context_defaults['C_v_h_type'] = v_h_type
    context_defaults['C_noise_params'] = noise_params
    # --- END OF NEW INPUT STEP ---
    
    scenarios = [
        {"title": "Extreme 1: Dominance & Risk-Averse", "initial_gap": 75.0, "C_p_war_max": 0.02},
        {"title": "Extreme 2: Desperate & Risk-Tolerant", "initial_gap": -75.0, "C_p_war_max": 0.25},
        {"title": "Extreme 3: Parity & Max Risk", "initial_gap": 0.0, "C_p_war_max": 0.25},
    ]
    
    llm_style = get_llm_style()
    batch_results_list = []
    
    for i, scenario in enumerate(scenarios):
        # get_base_params now uses the updated C_v_h_type and C_noise_params from context_defaults
        params = get_base_params(
            context_defaults=context_defaults,
            initial_gap=scenario['initial_gap'],
            C_p_war_max=scenario['C_p_war_max'],
            iterations=5000 
        )
        
        print(f"\n--- Running Scenario {i+1}: {scenario['title']} (N={params['iterations']}) ---")
        results, dynamic_paths, avg_cii, avg_sii, war_ratio = run_simulation_batch(params)
        
        # Accumulate results for batch analysis
        batch_results_list.append({
            'scenario_title': scenario['title'],
            'params': params,
            'results': results,
            'avg_cii': avg_cii,
            'avg_sii': avg_sii,
            'war_ratio': war_ratio
        })
        
        # Save individual data files
        save_scenario_data(params, results, dynamic_paths)
        
    mode_context = "Model Boundary Analysis (Extreme Edge). The task is to describe the solution surface based on the comparative data. Focus on systemic stability, structural resilience, and identifying the 'point of no return' revealed by these extreme parameters."
    
    # New: Run collective analysis
    generate_batch_analysis(batch_results_list, mode_context, llm_style, "BATCH REPORT: Extreme Edges Solution Surface")
        
    print("\n--- BATCH EXECUTION COMPLETE ---")


def run_interior_batch(context_defaults: Dict[str, Any]):
    """
    Mode 3: Runs a user-defined batch of scenarios (Interior Combinations).
    Allows defining a range and granularity for ΔU and P_WarMax.
    """
    print("\n--- MODE 3: CUSTOM INTERIOR BATCH ANALYSIS ---")
    
    # --- START OF NEW INPUT STEP FOR VOLATILITY PROFILE ---
    print("\n--- BATCH-WIDE VOLATILITY PROFILE SETUP ---")
    default_key = PROFILE_DEFAULTS[context_defaults['C_profile_name']]['vol_key']
    v_h_type, noise_params = get_dist(
        "Challenger Innovation Volatility (V_H Type) for ALL batch scenarios", 
        default_key
    )
    # Overwrite the context_defaults with the user's specific choice for the batch
    context_defaults['C_v_h_type'] = v_h_type
    context_defaults['C_noise_params'] = noise_params
    # --- END OF NEW INPUT STEP ---
    
    # --------------------------------------------------------
    # 1. VARIABLE DEFINITION
    # --------------------------------------------------------
    
    # --- Delta U Setup ---
    print("\n--- 1. Define Range for Initial Utility Gap (ΔU) ---")
    min_du = get_float("Minimum ΔU (e.g., -40.0, -5.0)", -20.0)
    max_du = get_float("Maximum ΔU (e.g., 50.0, 20.0)", 20.0)
    steps_du = get_int("Number of steps for ΔU (1 to fix, 3 for low grid)", 3)

    # --- P_WarMax Setup ---
    print("\n--- 2. Define Range for Challenger Risk Tolerance (P_WarMax) ---")
    min_p_war = get_float("Minimum P_WarMax (e.g., 0.05, 0.08)", 0.08)
    max_p_war = get_float("Maximum P_WarMax (e.g., 0.20, 0.15)", 0.15)
    steps_p_war = get_int("Number of steps for P_WarMax (1 to fix, 2 for low grid)", 2)
    
    # --------------------------------------------------------
    # 2. GENERATE SCENARIOS GRID
    # --------------------------------------------------------
    
    # Generate evenly spaced values for each parameter
    # Ensure steps is at least 1, even if max < min (linspace handles this)
    du_values = np.linspace(min_du, max_du, max(1, steps_du))
    p_war_values = np.linspace(min_p_war, max_p_war, max(1, steps_p_war))
    
    scenarios_list = []
    
    # Cartesian product of all combinations
    for du_val in du_values:
        for p_war_val in p_war_values:
            scenario = {
                "initial_gap": du_val, 
                "C_p_war_max": p_war_val
            }
            scenario['title'] = (
                f"ΔU={du_val:.1f} / P_W_C={p_war_val:.3f}"
            )
            scenarios_list.append(scenario)

    if not scenarios_list:
        print("❌ Error: No valid scenarios generated.")
        return
        
    total_scenarios = len(scenarios_list)
    print(f"\n--- Running {total_scenarios} Scenario Combination(s) (N=10000 per) ---")

    llm_style = get_llm_style()
    batch_results_list = []

    # --------------------------------------------------------
    # 3. RUN SIMULATION LOOP
    # --------------------------------------------------------

    for i, scenario in enumerate(scenarios_list):
        # Create full parameter set for the current combination
        # get_base_params now uses the updated C_v_h_type and C_noise_params from context_defaults
        params = get_base_params(
            context_defaults=context_defaults,
            initial_gap=scenario['initial_gap'],
            C_p_war_max=scenario['C_p_war_max'],
            iterations=10000 
        )

        print(f"\n--- Running Scenario {i+1}/{total_scenarios}: {scenario['title']} ---")
        results, dynamic_paths, avg_cii, avg_sii, war_ratio = run_simulation_batch(params)

        # Accumulate results for batch analysis
        batch_results_list.append({
            'scenario_title': scenario['title'],
            'params': params,
            'results': results,
            'avg_cii': avg_cii,
            'avg_sii': avg_sii,
            'war_ratio': war_ratio
        })
        
        # Save individual data files
        save_scenario_data(params, results, dynamic_paths)

    mode_context = f"Solution Surface Analysis of {total_scenarios} Points. The task is to describe the solution surface based on the comparative data. Focus on high volatility, risk management, and identifying the closest pivot points within this mid-range area. You MUST propose a single parameter and direction of change for the most critical pivot."

    # New: Run collective analysis
    generate_batch_analysis(batch_results_list, mode_context, llm_style, "BATCH REPORT: Custom Interior Solution Surface Grid")

    print("\n--- BATCH EXECUTION COMPLETE ---")


# ==============================================================================
# 4) LLM INTERPRETATION AND REPORTING FUNCTIONS
# ==============================================================================

def generate_comparative_tables(batch_data: List[Dict[str, Any]]) -> str:
    """
    Formats batch data into 2-3 comparative tables for the LLM prompt. 
    """
    
    scenario_titles = [d['scenario_title'] for d in batch_data]
    
    # --- Table 1: Input Parameters & Key Drivers ---
    
    data1 = {
        'Scenario': scenario_titles,
        'ΔU_Initial': [d['params']['initial_gap'] for d in batch_data],
        'P_WarMax_C': [d['params']['C_p_war_max'] for d in batch_data],
        'γ_A_Friction': [d['params']['A_gamma_base'] for d in batch_data],
        'Challenger_V_Type': [d['params']['C_v_h_type'].name for d in batch_data],
        'Desp. Ratio (W1/W_C)': [d['war_ratio'] for d in batch_data],
    }
    df1 = pd.DataFrame(data1).set_index('Scenario')
    # Use .map and format floats
    df1 = df1.T.map(lambda x: f"{x:.3f}" if isinstance(x, (float, np.float64)) else x)
    
    table_str_1 = "TABLE 1: INPUT PARAMETERS AND DESPERATION VALIDATION\n"
    table_str_1 += df1.to_csv(sep='|', float_format='%.3f')
    
    # --- Table 2: Structural Indicators ---
    
    data2 = {
        'Scenario': scenario_titles,
        'CII_Avg (Capacity)': [d['avg_cii'] for d in batch_data],
        'SII_Avg (Instability)': [d['avg_sii'] for d in batch_data],
    }
    df2 = pd.DataFrame(data2).set_index('Scenario')
    df2 = df2.T.map(lambda x: f"{x:.2f}" if isinstance(x, (float, np.float64)) else x)

    table_str_2 = "\n\nTABLE 2: AVERAGE FINAL STRUCTURAL INDICATORS\n"
    table_str_2 += df2.to_csv(sep='|', float_format='%.2f')

    # --- Table 3: Top Outcome Probabilities ---
    
    # Calculate W Total
    def get_w_total(results):
        w_outcomes = [Outcome.CHALLENGER_WAR_DESPERATION.name, Outcome.CHALLENGER_WAR_STRENGTH.name, 
                      Outcome.ADVERSARY_WAR_DESPERATION.name, Outcome.ADVERSARY_WAR_STRENGTH.name]
        return sum(results.get(o, 0) for o in w_outcomes)

    data3 = {
        'Scenario': scenario_titles,
        'W_Total': [get_w_total(d['results']) for d in batch_data],
        'S1_Stalemate': [d['results'].get(Outcome.DYNAMIC_EQUILIBRIUM.name, 0) for d in batch_data],
        'D1_Challenger_Dom': [d['results'].get(Outcome.CHALLENGER_DOMINANCE.name, 0) for d in batch_data],
        'D2_Adversary_Dom': [d['results'].get(Outcome.ADVERSARY_DOMINANCE.name, 0) for d in batch_data],
    }
    df3 = pd.DataFrame(data3).set_index('Scenario')
    df3 = df3.T.map(lambda x: f"{x:.2f}%" if isinstance(x, (float, np.float64)) else x)

    table_str_3 = "\n\nTABLE 3: KEY OUTCOME PROBABILITIES (PERCENT)\n"
    table_str_3 += df3.to_csv(sep='|', float_format='%.2f')
    
    
    final_output = "\n"
    final_output += table_str_1.replace('\r\n', '\n')
    final_output += "\n"
    final_output += table_str_2.replace('\r\n', '\n')
    final_output += "\n"
    final_output += table_str_3.replace('\r\n', '\n')
    
    return final_output


def get_llm_interpretation(data_to_analyze: Union[Dict[str, Any], List[Dict[str, Any]]], mode_context: str, llm_style: int) -> str:
    """
    Connects to the LLM to generate analysis. Handles both single-scenario
    and multi-scenario (batch) input.
    """
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "\n[LLM FAILED: Gemini API Key not found in environment variables. Set GEMINI_API_KEY.]"

    # --- STYLE AND READABILITY DEFINITIONS ---
    style_instructions = ""
    output_format = ""
    
    if llm_style == 1:
        style_instructions = "Your analysis must be written for a **policy staffer** or **average news reader**, using clear, non-academic language. The total analysis **must not exceed 250 words**. Use **active voice** and short, punchy sentences. Simplify all technical terms: *P_WarMax* is 'risk tolerance'; *War Choice Ratio* is 'desperation vs. strength.' **DO NOT use the acronyms CII, SII, P_WarMax, or Delta U (ΔU) in your final narrative.**"
        output_format = "Provide a concise, numbered analysis in three sections."
    elif llm_style == 2:
        style_instructions = "Adopt a **long-term, strategic analyst** tone. Focus on underlying structural causes and trends over a 5-10 year horizon. The total analysis **must not exceed 350 words**. Use the technical terms CII, SII, and P_WarMax for precision, but define them implicitly."
        output_format = "Provide a detailed, numbered analysis in three sections."
    elif llm_style == 3:
        style_instructions = "Adopt an **executive summary** tone. Analysis must be presented in three 1-sentence bullet points. Use direct, high-level risk communication language. **Do not use any acronyms or jargon.**"
        output_format = "Provide the analysis in three single-sentence bullet points, with each point corresponding to the required output sections."

    # --- Data Formatting based on Mode (Single vs. Batch) ---
    
    if isinstance(data_to_analyze, list):
        # Mode 2 or 3: Batch Analysis (Solution Surface)
        enriched_data_str = generate_comparative_tables(data_to_analyze)
        
        # Override key instructions for solution surface analysis
        mode_context += (
            "\n\n**SOLUTION SURFACE INSTRUCTION:** Your primary task is to describe the **solution surface** defined by the comparative tables. Analyze the sensitivity of the overall system to changes in the Challenger's initial gap ($\Delta U$) and risk tolerance ($P_{\text{WarMax}}$). Do not report on the three scenarios individually. Instead, synthesize the collective data into a cohesive analysis of where the systemic **tipping points** and **stable regions** exist across the parameter space."
        )
        # We need the full parameters from the first scenario for context
        params = data_to_analyze[0]['params']
        # Use placeholder metrics for the single-scenario prompt variables
        war_ratio = 0.0 
        avg_cii = 0.0
        avg_sii = 0.0
        results_formatted = "See TABLE 3 for comparative outcome probabilities."
    else:
        # Mode 1 or 4: Single Scenario Analysis
        params = data_to_analyze['params']
        results = data_to_analyze['results']
        avg_cii = data_to_analyze['avg_cii']
        avg_sii = data_to_analyze['avg_sii']
        war_ratio = data_to_analyze['war_ratio']
        
        results_formatted = "\n".join([f"- {name}: {value:.2f}%" for name, value in results.items()])
        
        # Determine the user-friendly name for the volatility profile
        vol_name = ""
        for key, profile in VOLATILITY_PROFILES.items():
            if profile['dist_type'] == params['C_v_h_type'] and profile['noise_params'] == params['C_noise_params']:
                 vol_name = key
                 break
        
        enriched_data_str = f"""
        INPUT PARAMETERS:
        - Challenger Profile: {params['C_profile_name']} (Volatility: {vol_name} / {params['C_v_h_type'].name})
        - Adversary Profile: {params['A_profile_name']} (Friction: {params['A_gamma_base']:.3f})
        - Initial Utility Gap (ΔU): {params['initial_gap']:.1f}
        - Challenger Risk Tolerance (P_WarMax): {params['C_p_war_max']:.2f}
        
        INTERNAL METRICS (Average Final State):
        - Challenger Innovation Capacity Build-up (CII_Avg): {avg_cii:.2f} 
        - Adversary Structural Instability (SII_Avg): {avg_sii:.2f} 
        - War Choice Desperation Ratio: {war_ratio:.2f} 
        
        OUTCOME PROBABILITIES (N={params['iterations']}):
        {results_formatted}
        """

    try:
        # 2. Client Setup
        client = genai.Client(api_key=api_key)
        
        # 3. LLM Prompt Construction (Ensures strict structure and minimal drift)
        SYSTEM_PROMPT = f"""
        {style_instructions}
        Avoid jargon like 'stochastic volatility' or 'gamma decay.' Your analysis must be grounded entirely in the provided numerical data.
        
        ---
        MODEL CONTEXT:
        The Challenger (C) and Adversary (A) compete over 100 rounds. The outcomes (W1-C1) are mutually exclusive final fates:
        - W1/W2/W3/W4: War initiated by Challenger (Desperation/Strength) or Adversary (Desperation/Strength).
        - D1/D2: Structural Dominance achieved by Challenger/Adversary.
        - S1: Dynamic Equilibrium (Stalemate) after 100 rounds.
        - C1: Global System Collapse (Exogenous Catastrophe).
        
        ---
        MODE CONTEXT:
        {mode_context}
        
        ---
        SIMULATION DATA:
        {enriched_data_str}
        
        ---
        REQUIRED OUTPUT:
        {output_format}
        1. DOMINANT TREND & SYSTEMIC RISK: State the most robust overall outcome trend across the experiments and explain the key structural implication based on the capacity and instability metrics (CII/SII trends).
        2. KEY DYNAMIC DRIVER: Identify the single most critical input (Initial Gap, Adversary Friction, or Challenger Risk Tolerance) that shows the highest **sensitivity** (i.e., the parameter that, when changed, results in the largest shift in the War Total or Dominance/Stalemate probabilities). Link this finding to the Desperation vs. Strength ratio.
        3. STRATEGIC PIVOT/IMPLICATION: Provide the key policy-relevant takeaway, **strictly adhering to the instructions provided in the MODE CONTEXT.**
        """
        
        # 5. API Call
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=SYSTEM_PROMPT,
            config={"temperature": 0.2} # Low temperature ensures consistent, factual analysis
        )
        
        return response.text

    except APIError as e:
        return f"\n[LLM API ERROR: Failed to get analysis. Code: {e}. Check API key validity and billing.]"
    except Exception as e:
        return f"\n[LLM FAILED: An unexpected error occurred: {e}]"


def generate_summary_report(params: Dict[str, Any], results: Dict[str, float], avg_cii: float, avg_sii: float, war_ratio: float, mode_context: str, llm_style: int, scenario_title: str = "Level 1: ULTIMATE FATE PROBABILISTIC OUTCOME MAP"):
    """
    Generates the console report for a single scenario (Mode 1 and 4).
    """
    results_series = pd.Series(results)
    
    # Determine the user-friendly name for the volatility profile
    vol_name = ""
    for key, profile in VOLATILITY_PROFILES.items():
        if profile['dist_type'] == params['C_v_h_type'] and profile['noise_params'] == params['C_noise_params']:
             vol_name = key
             break
    
    print("\n=======================================================================")
    print(f"         {scenario_title}")
    print("=======================================================================")
    print(f"CONTEXT: Challenger={params['C_profile_name']} vs. Adversary={params['A_profile_name']}")
    print(f"I/P: ΔU={params['initial_gap']:.1f}, γ_A={params['A_gamma_base']:.3f}, P_W_C={params['C_p_war_max']:.3f} (V_H={vol_name})")
    print("-----------------------------------------------------------------------")
    
    # LEVEL 1: PROBABILISTIC MAP
    w_outcomes = [Outcome.CHALLENGER_WAR_DESPERATION, Outcome.CHALLENGER_WAR_STRENGTH, 
                  Outcome.ADVERSARY_WAR_DESPERATION, Outcome.ADVERSARY_WAR_STRENGTH]
    w_total = sum(results_series.get(o.name, 0) for o in w_outcomes)
    print("| WAR INITIATED (W) Totals:        |")
    print(f"|  W Total Probability:                               {w_total:>7.2f}%")
    print(f"|    W1: Challenger from Desperation:                {results_series.get(Outcome.CHALLENGER_WAR_DESPERATION.name, 0):>7.2f}%")
    print(f"|    W2: Challenger from Strength:                   {results_series.get(Outcome.CHALLENGER_WAR_STRENGTH.name, 0):>7.2f}%")
    print(f"|    W3: Adversary from Desperation:                 {results_series.get(Outcome.ADVERSARY_WAR_DESPERATION.name, 0):>7.2f}%")
    print(f"|    W4: Adversary from Strength:                    {results_series.get(Outcome.ADVERSARY_WAR_STRENGTH.name, 0):>7.2f}%")
    print("-----------------------------------------------------------------------")

    d_outcomes = [Outcome.CHALLENGER_DOMINANCE, Outcome.ADVERSARY_DOMINANCE]
    d_total = sum(results_series.get(o.name, 0) for o in d_outcomes)
    print("| STRUCTURAL DOMINANCE (D) Totals: |")
    print(f"|  D Total Probability:                               {d_total:>7.2f}%")
    print(f"|    D1: Challenger Achieves Structural Dominance:   {results_series.get(Outcome.CHALLENGER_DOMINANCE.name, 0):>7.2f}%")
    print(f"|    D2: Adversary Achieves Structural Dominance:    {results_series.get(Outcome.ADVERSARY_DOMINANCE.name, 0):>7.2f}%")
    print("-----------------------------------------------------------------------")
    
    s1_prob = results_series.get(Outcome.DYNAMIC_EQUILIBRIUM.name, 0)
    c1_prob = results_series.get(Outcome.GLOBAL_COLLAPSE.name, 0)
    print("| STALEMATE & CATASTROPHE:         |")
    print(f"|  S1: Dynamic Equilibrium (Stalemate):              {s1_prob:>7.2f}%")
    print(f"|  C1: Global System Collapse (Exogenous):           {c1_prob:>7.2f}%")
    print("=======================================================================")
    
    # LEVEL 3: RAW INSIGHT
    print("\n--- LEVEL 3: STRATEGIC INSIGHT (Desperation vs. Strength Validation) ---")
    print(f"Challenger's War Choice Ratio (Desperation/Total): {war_ratio:.2f}")
    print(f"Average Final Challenger Innovation Index (CII_Avg): {avg_cii:.2f}")
    print(f"Average Final Adversary Structural Instability Index (SII_Avg): {avg_sii:.2f}")

    # LEVEL 4: LLM INTERPRETATION
    print("\n=======================================================================")
    print("    LEVEL 4: LLM STRATEGIC INTERPRETATION (Powered by Gemini)")
    print("=======================================================================")
    
    # Bundle data for the single scenario analysis
    data_to_analyze = {
        'params': params,
        'results': results,
        'avg_cii': avg_cii,
        'avg_sii': avg_sii,
        'war_ratio': war_ratio
    }
    
    llm_analysis = get_llm_interpretation(data_to_analyze, mode_context, llm_style)
    print(llm_analysis)
    
    print("\n=======================================================================")
    print("--- END OF REPORT ---")
    print("=======================================================================")


def generate_batch_analysis(batch_data: List[Dict[str, Any]], mode_context: str, llm_style: int, report_title: str):
    """Generates the combined report for Modes 2 and 3, featuring comparative tables."""
    
    print("\n=======================================================================")
    print(f"         {report_title}")
    print("=======================================================================")
    
    # Print the core matchup context (using the first scenario's data)
    params = batch_data[0]['params']
    
    # Determine the user-friendly name for the volatility profile
    vol_name = ""
    for key, profile in VOLATILITY_PROFILES.items():
        if profile['dist_type'] == params['C_v_h_type'] and profile['noise_params'] == params['C_noise_params']:
             vol_name = key
             break
             
    print(f"CONTEXT: Challenger={params['C_profile_name']} vs. Adversary={params['A_profile_name']}")
    print(f"FIXED I/P: γ_A={params['A_gamma_base']:.3f}, V_H_C={vol_name} (N={params['iterations']} per scenario)")
    print("-----------------------------------------------------------------------")
    
    # --- LEVEL 3: COMPARATIVE RAW DATA ---
    print("\n--- LEVEL 3: COMPARATIVE RAW DATA TABLES ---")
    print("Data below compares the scenarios run in this batch.")
    
    # Print the tables directly to the console report
    tables_output = generate_comparative_tables(batch_data)
    print(tables_output)
    
    # --- LEVEL 4: LLM SOLUTION SURFACE INTERPRETATION ---
    print("\n=======================================================================")
    print("    LEVEL 4: LLM SOLUTION SURFACE INTERPRETATION (Powered by Gemini)")
    print("=======================================================================")
    
    # LLM function called with the entire list for collective analysis
    llm_analysis = get_llm_interpretation(batch_data, mode_context, llm_style)
    print(llm_analysis)
    
    print("\n=======================================================================")
    print("--- END OF REPORT ---")
    print("=======================================================================")


def run_automated_baseline_test():
    """
    Executes a predefined, calibrated baseline scenario and asserts that the 
    stochastic outcomes fall within an expected range. (Mode 4)
    """
    print("\n=======================================================")
    print("     MODE 4: AUTOMATED BASELINE REGRESSION TEST")
    print("=======================================================")
    
    TEST_ITERATIONS = 5000 
    
    # Set test context based on predefined profiles
    test_context = get_profile_params('US_STYLE')
    test_context['A_profile_name'] = 'CHINA_STYLE'
    test_context.update(get_profile_params('CHINA_STYLE')) # Overlay Adversary params
    test_context['C_profile_name'] = 'US_STYLE' # Ensure Challenger is correct

    # Re-apply the desired test parameters
    test_params = get_base_params(
        context_defaults=test_context,
        initial_gap=12.0,                  
        C_p_war_max=0.12,                  
        iterations=TEST_ITERATIONS
    )
    test_params['A_gamma_base'] = 0.02 # Specific friction override for test
    
    # Determine volatility name for output
    vol_name = ""
    for key, profile in VOLATILITY_PROFILES.items():
        if profile['dist_type'] == test_params['C_v_h_type']:
            vol_name = key
            break
            
    print(f"Running Baseline Test: {test_params['C_profile_name']} vs {test_params['A_profile_name']}")
    print(f"I/P: ΔU={test_params['initial_gap']:.1f}, γ_A={test_params['A_gamma_base']:.3f}, P_W_C={test_params['C_p_war_max']:.3f} (V_H={vol_name}) (N={TEST_ITERATIONS})")

    results, dynamic_paths, avg_cii, avg_sii, war_ratio = run_simulation_batch(test_params)

    W_TOTAL_TARGET = 67.0
    W_TOTAL_TOLERANCE = 5.0 
    
    w_outcomes = [Outcome.CHALLENGER_WAR_DESPERATION.name, Outcome.CHALLENGER_WAR_STRENGTH.name, 
                  Outcome.ADVERSARY_WAR_DESPERATION.name, Outcome.ADVERSARY_WAR_STRENGTH.name]
    w_total_actual = sum(results.get(o, 0) for o in w_outcomes)

    test_passed = (
        (w_total_actual > W_TOTAL_TARGET - W_TOTAL_TOLERANCE) and 
        (w_total_actual < W_TOTAL_TARGET + W_TOTAL_TOLERANCE)
    )

    mode_context = "Regression Test Baseline. Focus on model integrity and confirmation that the structural outcomes (War Total) remain stable within the expected tolerance."

    generate_summary_report(test_params, results, avg_cii, avg_sii, war_ratio, mode_context=mode_context, llm_style=1, scenario_title="BASELINE TEST REPORT")
    save_scenario_data(test_params, results, dynamic_paths)
    
    print("\n--- TEST ASSERTION ---")
    print(f"Actual W Total Probability: {w_total_actual:.2f}%")
    print(f"Expected Range: {W_TOTAL_TARGET - W_TOTAL_TOLERANCE:.2f}% to {W_TOTAL_TARGET + W_TOTAL_TOLERANCE:.2f}%")
    
    if test_passed:
        print("✅ TEST PASSED: Core model dynamics remain stable.")
        return True
    else:
        print(f"❌ TEST FAILED: W Total is outside the expected range. Drift Detected!")
        return False


def main_menu():
    """Initializes the match up and presents the mode selection menu."""
    
    context_defaults = matchup_selection_screen()
    
    while True:
        print("\n=======================================================")
        print("     MODEL EXECUTION MODE SELECTION")
        print("=======================================================")
        print("1: Interactive - (Single custom scenario, UX Enhanced)")
        print("2: Extreme Edges - (Batch analysis, contextualized)")
        print("3: Custom Interior Batch - (Targeted mid-range solution surface)")
        print("4: Automated Test - (Regression test for model stability)")
        print("0: Exit")
        print("-------------------------------------------------------")
        
        choice = input("Select Mode (1, 2, 3, 4, or 0): ")
        
        if choice == '1':
            run_interactive_mode(context_defaults)
            print("\nReturning to Main Menu.")
        elif choice == '2':
            run_extreme_batch(context_defaults)
            print("\nReturning to Main Menu.")
        elif choice == '3':
            run_interior_batch(context_defaults)
            print("\nReturning to Main Menu.")
        elif choice == '4':
            run_automated_baseline_test()
            print("\nReturning to Main Menu.")
        elif choice == '0':
            print("Exiting simulation.")
            break
        else:
            print("Invalid choice. Please select 1, 2, 3, 4, or 0.")

if __name__ == '__main__':
    main_menu()
