import numpy as np
import random
from enum import Enum
from typing import Dict, Any, Tuple, Optional, List
import json
import os
import pandas as pd

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

# --- SYSTEM PROFILE ASYMMETRY (Model Constants) ---
PROFILE_DEFAULTS = {
    'US_STYLE': {
        'v_h_type': DistributionType.GAMMA, 
        'gamma_base': 0.025,    # Higher Friction (Bureaucracy/Cost)
        'p_war_max': 0.12,      # Higher Risk Tolerance
    },
    'CHINA_STYLE': {
        'v_h_type': DistributionType.GAUSSIAN_T, 
        'gamma_base': 0.010,    # Lower Friction (Centralization/Efficiency)
        'p_war_max': 0.08,      # Lower Risk Tolerance
    },
    'UKRAINE_STYLE': {
        'v_h_type': DistributionType.GAMMA, # High-Variance Gamma for improvisation
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
        'meaning': "GAMMA (Disruptive) = Larger, less frequent gains/losses. GAUSSIAN_T (Steady) = Smaller, more consistent gains.",
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
def get_innovation_noise(dist_type: DistributionType) -> float:
    # Generates stochastic noise based on the chosen innovation profile (V_H).
    if dist_type == DistributionType.GAMMA:
        # High Variance/Disruptive: Can lead to large positive or negative shocks
        return np.random.gamma(shape=2.5, scale=2.5) - 6.0
    elif dist_type == DistributionType.GAUSSIAN_T:
        # Tighter/Consistent: Smaller variance around the mean
        mu, sigma = 3.0, 3.0
        noise = np.random.normal(mu, sigma)
        return max(0.0, noise) - mu
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
#    - MODIFIED to log dynamic path data for plotting
# ==============================================================================

class DynamicInnovationModel:
    def __init__(self, params: Dict[str, Any], log_path: bool = False):
        self.U = params['initial_gap']
        self.SII = 0.0  # Adversary Structural Dominance Index
        self.CII = 0.0  # Challenger Innovation Dominance Index (New)
        self.params = params
        self.outcome = None
        self.rounds = 0
        self.log_path = log_path
        self.path_log: List[float] = [] # New: Log of Delta U per round
        
        # Asymmetric parameters loaded based on profile selection
        self.C_PWarMax = params['C_p_war_max']
        self.A_GammaBase = params['A_gamma_base']
        self.C_VHDist = params['C_v_h_type']
        
        # New: Global Catastrophe Probability (Fixed base rate)
        self.P_CAT_BASE = 0.0001 

    # --- Strategy Choice and Heuristics (Asymmetric) ---

    def _get_war_choice_prob(self, p_max: float, actor_U: float) -> float:
        """Calculates P(War Choice) based on Desperation and adds Noise."""
        U_collapse = self.params['u_collapse']
        U_ref = -50.0  # Reference point for desperation
        
        # 1. Desperation Heuristic (U near collapse)
        if actor_U > U_ref:
            p_desperation = p_max * 0.1
        else:
            desperation_scale = (U_ref - actor_U) / (U_ref - U_collapse)
            desperation_scale = max(0.0, min(1.0, desperation_scale))
            p_desperation = p_max * (0.1 + 0.9 * desperation_scale**2)

        # 2. Stochastic/Irrationality Component (Noise)
        p_desperation += get_irrationality_noise()
        
        return max(0.0, min(p_max, p_desperation))

    def _check_war_initiation(self) -> Tuple[bool, Optional[Outcome]]:
        """Checks if either Challenger or Adversary initiates War."""
        # --- Challenger Initiation Check ---
        C_P_Choice = self._get_war_choice_prob(self.C_PWarMax, self.U)
        if random.random() < C_P_Choice:
            # Challenger initiates War. Now, categorize the position (Heuristic)
            if self.U < -50.0:  # Arbitrary threshold for desperation
                return True, Outcome.CHALLENGER_WAR_DESPERATION # W1
            else:
                return True, Outcome.CHALLENGER_WAR_STRENGTH # W2

        # --- Adversary Initiation Check (New: Requires A_PWarMax) ---
        A_PWarMax = self.params.get('A_p_war_max', 0.10) 
        
        # Adversary Desperation Heuristic (Threatened by Challenger's Strength/Momentum)
        if self.CII > 25.0: 
            A_P_Desperation = A_PWarMax * (self.CII / 100.0) # Scales with Challenger's dominance
            if random.random() < A_P_Desperation:
                 return True, Outcome.ADVERSARY_WAR_DESPERATION # W3
        
        # Adversary Strength Heuristic (Seeing a window for a quick, decisive win)
        if self.SII > 50.0: 
            A_P_Strength = A_PWarMax * 0.5
            if random.random() < A_P_Strength:
                return True, Outcome.ADVERSARY_WAR_STRENGTH # W4

        return False, None

    # --- Utility Dynamics and Evolution ---

    def _check_global_collapse(self) -> bool:
        """Checks for the rare, exogenous Global System Collapse (C1)."""
        
        # Dynamic Modifier: Raise catastrophe risk if competition is stressed
        P_Catastrophe = self.P_CAT_BASE
        if self.U < -50.0 or self.SII > 50.0:
            P_Catastrophe *= 10 # 10x risk multiplier for highly stressed competition

        if random.random() < P_Catastrophe:
            self.outcome = Outcome.GLOBAL_COLLAPSE # C1
            return True
        return False
        
    def _apply_strategies(self, S_t: Strategy):
        """Applies the effects of the chosen strategy (currently always Challenger's)."""
        
        P_base = PAYOFFS[S_t]
        epsilon_S = get_innovation_noise(self.C_VHDist)
        
        # 1. Stochastic Friction Multiplier (Ebb and Flow)
        lambda_t = get_stochastic_friction_multiplier()
        gamma_base = self.A_GammaBase # Adversary's system friction
        decay_rate = 1.0 - (lambda_t * gamma_base) 
        
        # 2. Strategy Application
        if S_t == Strategy.HIGH_TEMPO:
            # 2a. Spillover Effect
            spill_rate = random.uniform(0.01, 0.05) 
            self.SII += epsilon_S * 0.002 * spill_rate 

        elif S_t == Strategy.ECONOMIC_WARFARE:
            # 2b. Economic Warfare Shock
            EW_shock = random.uniform(5.0, 15.0)
            epsilon_S -= EW_shock * 0.5 
            # Temporary Friction Increase (Simulates cost of counter-sanctions)
            decay_rate = 1.0 - (lambda_t * (gamma_base * 1.5)) 

        # 3. Utility Update
        self.U = (self.U * decay_rate) + P_base + epsilon_S
        
        # 4. Structural Index Update (Assumes Challenger gains reduce Adversary's SII)
        self.SII += (self.A_GammaBase * 10.0) # Passive Adversary Dominance accumulation
        self.SII -= P_base * 0.005           # Challenger's action reduces Adversary's dominance
        
        # 5. Challenger Dominance Index Update (New)
        self.CII += P_base * 0.005
        self.CII -= (self.C_VHDist.value * 5.0) 
        
        self.SII = max(0.0, self.SII)
        self.CII = max(0.0, self.CII)


    def _check_decisive_outcomes(self):
        """Checks for Dominance and Stalemate (D1, D2, S1)."""
        
        # 1. Adversary Achieves Dominance (D2)
        if self.U <= self.params['u_collapse'] or self.SII >= self.params['s_thresh']:
            self.outcome = Outcome.ADVERSARY_DOMINANCE
            return
            
        # 2. Challenger Achieves Dominance (D1)
        if self.CII >= self.params['c_thresh']:
            self.outcome = Outcome.CHALLENGER_DOMINANCE
            return

        # 3. Stalemate (S1)
        if self.rounds >= self.params['max_rounds']:
            self.outcome = Outcome.DYNAMIC_EQUILIBRIUM
            return

    def run_round(self):
        self.rounds += 1
        
        # LOGGING STEP: Save Delta U before checking for termination
        if self.log_path:
            self.path_log.append(self.U)
            
        # 1. Check for Exogenous Global Collapse (C1)
        if self._check_global_collapse():
            return
            
        # 2. Check for War Initiation (W1-W4)
        is_war, war_outcome = self._check_war_initiation()
        if is_war:
            self.outcome = war_outcome
            self.U += PAYOFFS[Strategy.WAR] # Apply War shock to utility before ending
            if self.log_path:
                # Log the final (shocked) U before game end
                self.path_log.append(self.U) 
            return
        
        # 3. If no War, execute standard strategies (Challenger only, for simplicity)
        S_t = Strategy.HIGH_TEMPO 
        if random.random() < 0.15: # 15% chance of choosing Economic Warfare instead of High Tempo
             S_t = Strategy.ECONOMIC_WARFARE
        
        self._apply_strategies(S_t)
        
        # 4. Check for Structural Dominance / Stalemate
        self._check_decisive_outcomes()

    def simulate(self) -> Tuple[Optional[Outcome], List[float]]:
        # Initialize log with starting U before round 1
        if self.log_path:
            self.path_log.append(self.U)

        while not self.outcome and self.rounds < self.params['max_rounds']:
            self.run_round()
            
        if not self.outcome:
            self.outcome = Outcome.DYNAMIC_EQUILIBRIUM
            
        # Pad the path log to max_rounds + 1 (starting value + 100 rounds) for consistency
        if self.log_path:
            while len(self.path_log) < self.params['max_rounds'] + 1:
                self.path_log.append(self.path_log[-1])
        
        return self.outcome, self.path_log


# ==============================================================================
# 3) UX WRAPPER: INPUT, BATCH EXECUTION, AND REPORTING
# ==============================================================================

def get_profile_params(profile_key: str) -> Dict[str, Any]:
    """Helper to retrieve parameters based on profile key."""
    profile_data = PROFILE_DEFAULTS.get(profile_key, PROFILE_DEFAULTS['US_STYLE'])
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
    
    # Consolidate parameters with specific prefixes for the model
    return {
        'C_v_h_type': c_params['v_h_type'], 
        'C_p_war_max': c_params['p_war_max'],
        'A_gamma_base': a_params['gamma_base'],
        'A_p_war_max': a_params['p_war_max'],
        'C_profile_name': c_profile_key,
        'A_profile_name': a_profile_key
    }


# MODIFIED TO RETURN DYNAMIC PATHS FOR A SMALL SAMPLE
def run_simulation_batch(params: Dict[str, Any], path_sample_size: int = 50) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    """Runs N iterations and aggregates results, collecting path data for a sample."""
    
    iterations = params['iterations']
    results = {o.name: 0 for o in Outcome}
    dynamic_paths = []
    
    for i in range(iterations):
        # Only log the path for the first N runs (the sample size)
        log_path = i < path_sample_size
        model = DynamicInnovationModel(params, log_path=log_path)
        
        outcome, path_log = model.simulate()
        
        if outcome:
            results[outcome.name] += 1
            
        if log_path and path_log:
            dynamic_paths.append({
                'id': i + 1,
                'outcome': outcome.name,
                # Convert numpy floats to native Python floats for JSON serialization
                'path': [float(u) for u in path_log]
            })
            
    total = sum(results.values())
    final_results = {name: (count / total) * 100 for name, count in results.items()}
    
    return final_results, dynamic_paths

# Corrected Input Handlers for Robustness (omitted for brevity, assume they are present)
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

def get_dist(prompt, default_val):
    default_str = "GAMMA" if default_val == DistributionType.GAMMA else "GAUSSIAN_T"
    while True:
        v_h_input = input(f"   > {prompt} [GAMMA/GAUSSIAN_T] (Default: {default_str}): ").upper()
        if not v_h_input:
            return default_val
        elif v_h_input == "GAMMA":
            return DistributionType.GAMMA
        elif v_h_input == "GAUSSIAN_T":
            return DistributionType.GAUSSIAN_T
        else:
            print("Invalid input. Please enter GAMMA or GAUSSIAN_T.")

# NEW FUNCTION: Data Serialization to JSON
def save_scenario_data(params: Dict[str, Any], results: Dict[str, float], dynamic_paths: List[Dict[str, Any]]):
    """Saves scenario results and dynamic paths to a structured JSON file."""
    
    # Ensure the output directory exists
    output_dir = "simulation_data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. Create the full data structure
    data_to_save = {
        'scenario_params': {
            k: v.name if isinstance(v, (Enum, DistributionType)) else v 
            for k, v in params.items()
        },
        'probabilistic_results': results,
        'dynamic_paths_sample': dynamic_paths
    }

    # 2. Generate a clean filename based on context
    c_name = params['C_profile_name']
    a_name = params['A_profile_name']
    delta_u = f"DU_{params['initial_gap']:.1f}"
    
    filename = os.path.join(output_dir, f"{c_name}_vs_{a_name}_{delta_u}_N{params['iterations']}.json")
    
    # 3. Write data to JSON file
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
    
    # 1. Initial Balance (Initial Competitive Gap)
    print("\n\n-- 1. COMPETITIVE STARTING CONDITIONS --")
    print_context('initial_gap')
    initial_gap = get_float("Initial Competitive Gap (ΔU_initial)", -15.0)
    
    # 2. Structural Constraints (Friction and Risk - Defaults overridden by context)
    print("\n\n-- 2. SYSTEMIC CONSTRAINTS & BEHAVIOR (Defaults from Profiles) --")
    
    print_context('gamma_base')
    gamma_base_input = get_float("Adversary Structural Friction (γ_base)", context_defaults['A_gamma_base'])
    
    print_context('p_war_max')
    p_war_max_input = get_float("Challenger Risk Tolerance (P_WarMax)", context_defaults['C_p_war_max'])
    
    print_context('v_h_type')
    v_h_type_input = get_dist("Challenger Innovation Volatility (V_H Type)", context_defaults['C_v_h_type'])

    # 3. Simulation Parameters
    print("\n\n-- 3. SIMULATION PARAMETERS --")
    print_context('iterations')
    iterations_input = get_int("Iterations (N, Total runs)", 10000)
    
    # Consolidated parameters 
    params = {
        'initial_gap': initial_gap, 
        'iterations': iterations_input, 
        'C_v_h_type': v_h_type_input,
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
    
    print("\n--- RUNNING INTERACTIVE SIMULATION ---")
    results, dynamic_paths = run_simulation_batch(params)
    
    # Generate the comprehensive reports and save data
    generate_summary_report(params, results)
    save_scenario_data(params, results, dynamic_paths)
    
    print("\n--- SCENARIO COMPLETE ---")


def generate_summary_report(params: Dict[str, Any], results: Dict[str, float], scenario_title: str = "Level 1: ULTIMATE FATE PROBABILISTIC OUTCOME MAP"):
    """Generates the Level 1 Outcome Map and Level 3 War Context Summary (Console Report)."""
    
    # Convert results to a pandas Series for easy access and grouping
    results_series = pd.Series(results)
    
    print("\n=======================================================================")
    print(f"         {scenario_title}")
    print("=======================================================================")
    print(f"CONTEXT: Challenger={params['C_profile_name']} vs. Adversary={params['A_profile_name']}")
    print(f"I/P: ΔU={params['initial_gap']:.1f}, γ_A={params['A_gamma_base']:.3f}, P_W_C={params['C_p_war_max']:.3f}")
    print("-----------------------------------------------------------------------")
    
    # 1. War Initiation Outcomes (W1-W4)
    w_outcomes = [Outcome.CHALLENGER_WAR_DESPERATION, Outcome.CHALLENGER_WAR_STRENGTH, 
                  Outcome.ADVERSARY_WAR_DESPERATION, Outcome.ADVERSARY_WAR_STRENGTH]
    # Summing using list comprehension to avoid potential KeyError on missing outcomes
    w_total = sum(results_series.get(o.name, 0) for o in w_outcomes)
    print("| WAR INITIATED (W) Totals:        |")
    print(f"|  W Total Probability:                               {w_total:>7.2f}%")
    print(f"|    W1: Challenger from Desperation:                {results_series.get(Outcome.CHALLENGER_WAR_DESPERATION.name, 0):>7.2f}%")
    print(f"|    W2: Challenger from Strength:                   {results_series.get(Outcome.CHALLENGER_WAR_STRENGTH.name, 0):>7.2f}%")
    print(f"|    W3: Adversary from Desperation:                 {results_series.get(Outcome.ADVERSARY_WAR_DESPERATION.name, 0):>7.2f}%")
    print(f"|    W4: Adversary from Strength:                    {results_series.get(Outcome.ADVERSARY_WAR_STRENGTH.name, 0):>7.2f}%")
    print("-----------------------------------------------------------------------")

    # 2. Structural Dominance Outcomes (D1-D2)
    d_outcomes = [Outcome.CHALLENGER_DOMINANCE, Outcome.ADVERSARY_DOMINANCE]
    d_total = sum(results_series.get(o.name, 0) for o in d_outcomes)
    print("| STRUCTURAL DOMINANCE (D) Totals: |")
    print(f"|  D Total Probability:                               {d_total:>7.2f}%")
    print(f"|    D1: Challenger Achieves Structural Dominance:   {results_series.get(Outcome.CHALLENGER_DOMINANCE.name, 0):>7.2f}%")
    print(f"|    D2: Adversary Achieves Structural Dominance:    {results_series.get(Outcome.ADVERSARY_DOMINANCE.name, 0):>7.2f}%")
    print("-----------------------------------------------------------------------")
    
    # 3. Exogenous / Stalemate Outcomes (C1, S1)
    s1_prob = results_series.get(Outcome.DYNAMIC_EQUILIBRIUM.name, 0)
    c1_prob = results_series.get(Outcome.GLOBAL_COLLAPSE.name, 0)
    print("| STALEMATE & CATASTROPHE:         |")
    print(f"|  S1: Dynamic Equilibrium (Stalemate):              {s1_prob:>7.2f}%")
    print(f"|  C1: Global System Collapse (Exogenous):           {c1_prob:>7.2f}%")
    print("=======================================================================")
    
    print("\n--- LEVEL 3: STRATEGIC INSIGHT (Desperation vs. Strength Validation) ---")
    c_war_desp = results_series.get(Outcome.CHALLENGER_WAR_DESPERATION.name, 0)
    c_war_strength = results_series.get(Outcome.CHALLENGER_WAR_STRENGTH.name, 0)
    
    if (c_war_desp + c_war_strength) > 0:
        ratio = c_war_desp / (c_war_desp + c_war_strength)
    else:
        ratio = 0.0

    print(f"Challenger's War Choice Ratio (Desperation/Total): {ratio:.2f}")


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
    }

def run_extreme_batch(context_defaults: Dict[str, Any]):
    """Mode 2: Runs a batch of scenarios testing extreme initial conditions."""
    print("\n--- MODE 2: EXTREME EDGES BATCH ANALYSIS (5000 Iterations per Scenario) ---")
    
    scenarios = [
        {"title": "Extreme 1: Dominance & Risk-Averse", "initial_gap": 75.0, "C_p_war_max": 0.02},
        {"title": "Extreme 2: Desperate & Risk-Tolerant", "initial_gap": -75.0, "C_p_war_max": 0.25},
        {"title": "Extreme 3: Parity & Max Risk", "initial_gap": 0.0, "C_p_war_max": 0.25},
    ]

    for i, scenario in enumerate(scenarios):
        params = get_base_params(
            context_defaults=context_defaults,
            initial_gap=scenario['initial_gap'],
            C_p_war_max=scenario['C_p_war_max'],
            iterations=5000 
        )
        
        print(f"\n--- Running Scenario {i+1}: {scenario['title']} (N={params['iterations']}) ---")
        results, dynamic_paths = run_simulation_batch(params)
        generate_summary_report(params, results, scenario_title=f"SCENARIO {i+1}: {scenario['title']}")
        save_scenario_data(params, results, dynamic_paths)
        
    print("\n--- BATCH EXECUTION COMPLETE ---")


def run_interior_batch(context_defaults: Dict[str, Any]):
    """Mode 3: Runs a batch of scenarios testing mid-range, transitional conditions."""
    print("\n--- MODE 3: INTERIOR BATCH ANALYSIS (10000 Iterations per Scenario) ---")
    
    scenarios = [
        {"title": "Interior 1: Slight Deficit & Moderate Risk", "initial_gap": -20.0, "C_p_war_max": 0.15},
        {"title": "Interior 2: Near Parity & Risk-Averse", "initial_gap": -5.0, "C_p_war_max": 0.08},
        {"title": "Interior 3: Slight Advantage & Moderate Risk", "initial_gap": 20.0, "C_p_war_max": 0.15},
    ]

    for i, scenario in enumerate(scenarios):
        params = get_base_params(
            context_defaults=context_defaults,
            initial_gap=scenario['initial_gap'],
            C_p_war_max=scenario['C_p_war_max'],
            iterations=10000 
        )

        print(f"\n--- Running Scenario {i+1}: {scenario['title']} (N={params['iterations']}) ---")
        results, dynamic_paths = run_simulation_batch(params)
        generate_summary_report(params, results, scenario_title=f"SCENARIO {i+1}: {scenario['title']}")
        save_scenario_data(params, results, dynamic_paths)

    print("\n--- BATCH EXECUTION COMPLETE ---")


# ==============================================================================
# 4) AUTOMATED BASELINE TEST FUNCTION
# ==============================================================================

def run_automated_baseline_test():
    """
    Executes a predefined, calibrated baseline scenario and asserts that the 
    stochastic outcomes fall within an expected range. This acts as a regression test.
    """
    print("\n=======================================================")
    print("     MODE 4: AUTOMATED BASELINE REGRESSION TEST")
    print("=======================================================")
    
    TEST_ITERATIONS = 5000 
    
    test_context = {
        'C_profile_name': 'US_STYLE',
        'A_profile_name': 'CHINA_STYLE',
        'A_p_war_max': PROFILE_DEFAULTS['CHINA_STYLE']['p_war_max'],
        'A_gamma_base': PROFILE_DEFAULTS['CHINA_STYLE']['gamma_base'],
        'C_v_h_type': PROFILE_DEFAULTS['US_STYLE']['v_h_type'],
        'C_p_war_max': PROFILE_DEFAULTS['US_STYLE']['p_war_max'],
    }

    test_params = get_base_params(
        context_defaults=test_context,
        initial_gap=12.0,                  # Scenario parameter 1
        C_p_war_max=0.12,                  # Scenario parameter 2
        iterations=TEST_ITERATIONS
    )
    # NOTE: Overriding A_gamma_base to match the historical interactive run
    test_params['A_gamma_base'] = 0.02 
    
    print(f"Running Baseline Test: {test_params['C_profile_name']} vs {test_params['A_profile_name']}")
    print(f"I/P: ΔU={test_params['initial_gap']:.1f}, γ_A={test_params['A_gamma_base']:.3f}, P_W_C={test_params['C_p_war_max']:.3f} (N={TEST_ITERATIONS})")

    # 2. Execute Simulation
    results, dynamic_paths = run_simulation_batch(test_params)

    # 3. Define Expected Ranges 
    W_TOTAL_TARGET = 67.0
    W_TOTAL_TOLERANCE = 5.0 
    
    w_outcomes = [Outcome.CHALLENGER_WAR_DESPERATION.name, Outcome.CHALLENGER_WAR_STRENGTH.name, 
                  Outcome.ADVERSARY_WAR_DESPERATION.name, Outcome.ADVERSARY_WAR_STRENGTH.name]
    w_total_actual = sum(results.get(o, 0) for o in w_outcomes)

    test_passed = (
        (w_total_actual > W_TOTAL_TARGET - W_TOTAL_TOLERANCE) and 
        (w_total_actual < W_TOTAL_TARGET + W_TOTAL_TOLERANCE)
    )

    # 4. Reporting, Assertion, and Data Saving
    generate_summary_report(test_params, results, scenario_title="BASELINE TEST REPORT")
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
        print("3: Interior Batch - (Targeted mid-range scenarios)")
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
