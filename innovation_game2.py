import numpy as np
import random
import math
from enum import Enum
from typing import Dict, Any, Tuple, List

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
# NOTE: These are the DEFAULTS. Interactive mode allows user to override.
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
    # New: Allows for innovation to accelerate growth (multiplier < 1.0)
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
    def __init__(self, params: Dict[str, Any]):
        self.U = params['initial_gap']
        self.SII = 0.0  # Adversary Structural Dominance Index
        self.CII = 0.0  # Challenger Innovation Dominance Index (New)
        self.params = params
        self.outcome = None
        self.rounds = 0
        
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

    def _check_war_initiation(self) -> Tuple[bool, Outcome | None]:
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
        # NOTE: Using a simplified Adversary risk tolerance based on Challenger's P_WarMax for now
        A_PWarMax = self.params.get('A_p_war_max', 0.10) 
        
        # Adversary Desperation Heuristic (Threatened by Challenger's Strength/Momentum)
        # Simplified: Adversary feels desperate if Challenger's lead is growing fast (CII is high)
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
        
    def _apply_strategies(self, S_t: Strategy, payoff_multiplier: float = 1.0):
        """Applies the effects of the chosen strategy (currently always Challenger's)."""
        
        P_base = PAYOFFS[S_t]
        epsilon_S = get_innovation_noise(self.C_VHDist)
        
        # 1. New: Stochastic Friction Multiplier (Ebb and Flow)
        lambda_t = get_stochastic_friction_multiplier()
        gamma_base = self.A_GammaBase # Adversary's system friction
        decay_rate = 1.0 - (lambda_t * gamma_base) # Decay can be > 1.0 if lambda_t * gamma_base < 0 
        
        # 2. Strategy Application
        if S_t == Strategy.HIGH_TEMPO:
            # 2a. Spillover Effect (New: A percentage of Challenger's gain accrues to Adversary)
            spill_rate = random.uniform(0.01, 0.05) # 1-5% spillover
            self.SII += epsilon_S * 0.002 * spill_rate # Small SII benefit from spillover

        elif S_t == Strategy.ECONOMIC_WARFARE:
            # 2b. Economic Warfare Shock (New: Hits opponent's system)
            # Imposes negative shock on opponent's system (temporary utility reduction)
            # And increases structural friction temporarily (e.g., sanction cost)
            EW_shock = random.uniform(5.0, 15.0)
            epsilon_S -= EW_shock * 0.5 # Challenger's gain is reduced
            # We don't track the Adversary's U, so we assume Adversary's U is reduced
            # and Challenger's innovation must overcome the shock:
            
            # Temporary Friction Increase (Simulates cost of counter-sanctions)
            decay_rate = 1.0 - (lambda_t * (gamma_base * 1.5)) # 50% temporary friction boost

        # 3. Utility Update
        self.U = (self.U * decay_rate) + P_base + epsilon_S
        
        # 4. Structural Index Update (Assumes Challenger gains reduce Adversary's SII)
        self.SII += (self.A_GammaBase * 10.0) # Passive Adversary Dominance accumulation
        self.SII -= P_base * 0.005           # Challenger's action reduces Adversary's dominance
        
        # 5. Challenger Dominance Index Update (New)
        # CII tracks Challenger's dominance relative to a positive threshold
        self.CII += P_base * 0.005
        self.CII -= (self.C_VHDist.value * 5.0) # Tighter distribution (GAUSSIAN_T=1) reduces CII faster
        
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
        
        # 1. Check for Exogenous Global Collapse (C1)
        if self._check_global_collapse():
            return
            
        # 2. Check for War Initiation (W1-W4)
        is_war, war_outcome = self._check_war_initiation()
        if is_war:
            self.outcome = war_outcome
            self.U += PAYOFFS[Strategy.WAR] # Apply War shock to utility before ending
            return
        
        # 3. If no War, execute standard strategies (Challenger only, for simplicity)
        S_t = Strategy.HIGH_TEMPO 
        if random.random() < 0.15: # 15% chance of choosing Economic Warfare instead of High Tempo
             S_t = Strategy.ECONOMIC_WARFARE
        
        self._apply_strategies(S_t)
        
        # 4. Check for Structural Dominance / Stalemate
        self._check_decisive_outcomes()

    def simulate(self):
        while not self.outcome and self.rounds < self.params['max_rounds']:
            self.run_round()
        if not self.outcome:
            self.outcome = Outcome.DYNAMIC_EQUILIBRIUM


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
                print("Invalid choice.")
        except ValueError:
            print("Invalid input.")

    # 2. Adversary Selection
    while True:
        try:
            a_choice = int(input("Select Adversary Profile (1-3): "))
            if a_choice in PROFILE_NAMES:
                a_profile_key = PROFILE_NAMES[a_choice]
                a_params = get_profile_params(a_profile_key)
                break
            else:
                print("Invalid choice.")
        except ValueError:
            print("Invalid input.")
            
    print(f"\nMatchup Set: Challenger ({c_profile_key}) vs. Adversary ({a_profile_key})")
    
    # Consolidate parameters with specific prefixes for the model
    # Note: Challenger's V_H and P_WarMax are used to define the C_profile defaults
    # Note: Adversary's GammaBase and P_WarMax are used to define the A_profile defaults
    return {
        'C_v_h_type': c_params['v_h_type'], 
        'C_p_war_max': c_params['p_war_max'],
        'A_gamma_base': a_params['gamma_base'],
        'A_p_war_max': a_params['p_war_max'],
        'C_profile_name': c_profile_key,
        'A_profile_name': a_profile_key
    }


def run_single_simulation(params: Dict[str, Any]) -> Dict[str, float]:
    """Runs the simulation for N iterations and aggregates results."""
    iterations = params['iterations']
    results = {o.name: 0 for o in Outcome}
    
    for i in range(iterations):
        model = DynamicInnovationModel(params)
        model.simulate()
        if model.outcome:
            results[model.outcome.name] += 1
            
    total = sum(results.values())
    final_results = {name: (count / total) * 100 for name, count in results.items()}
    
    return final_results

def run_interactive_mode(context_defaults: Dict[str, Any]):
    """Handles terminal user input for scenario calibration (Mode 1), with enhanced UX."""
    
    # --- Input Handlers ---
    def get_float(prompt, default):
        return float(input(f"   > {prompt} (Default: {default}): ") or default)
    def get_int(prompt, default):
        return int(input(f"   > {prompt} (Default: {default}): ") or default)
    def get_dist(prompt, default_val):
        default_str = "GAMMA" if default_val == DistributionType.GAMMA else "GAUSSIAN_T"
        v_h_input = input(f"   > {prompt} [GAMMA/GAUSSIAN_T] (Default: {default_str}): ")
        if v_h_input.upper() == "GAMMA":
            return DistributionType.GAMMA
        elif v_h_input.upper() == "GAUSSIAN_T":
            return DistributionType.GAUSSIAN_T
        return default_val
    
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
    gamma_base = get_float("Adversary Structural Friction (γ_base)", context_defaults['A_gamma_base'])
    
    print_context('p_war_max')
    p_war_max = get_float("Challenger Risk Tolerance (P_WarMax)", context_defaults['C_p_war_max'])
    
    print_context('v_h_type')
    v_h_type = get_dist("Challenger Innovation Volatility (V_H Type)", context_defaults['C_v_h_type'])

    # 3. Simulation Parameters
    print("\n\n-- 3. SIMULATION PARAMETERS --")
    print_context('iterations')
    iterations = get_int("Iterations (N, Total runs)", 10000)
    
    # Fixed model constants and consolidated parameters
    params = {
        'initial_gap': initial_gap, 
        'gamma_base': gamma_base, 
        'p_war_max': p_war_max,
        'v_h_type': v_h_type, 
        'iterations': iterations, 
        
        # Override context defaults for the core model logic with user's interactive choices
        'C_v_h_type': v_h_type,
        'C_p_war_max': p_war_max,
        'A_gamma_base': gamma_base,
        'A_p_war_max': context_defaults['A_p_war_max'], # Use context for Adversary's risk
        
        # Fixed termination constants
        'max_rounds': 100, 
        'u_collapse': -100.0, 
        's_thresh': 100.0, # Adversary Dominance Threshold
        'c_thresh': 100.0, # Challenger Dominance Threshold
        
        # Report context
        'C_profile_name': context_defaults['C_profile_name'],
        'A_profile_name': context_defaults['A_profile_name']
    }
    
    print("\n--- RUNNING INTERACTIVE SIMULATION ---")
    results = run_single_simulation(params)
    
    # Generate the comprehensive reports
    generate_summary_report(params, results)
    # The other detailed reports (Level 2 & 3) will be implemented as separate functions
    # for the web app, but their data is contained in the simulation results.


def generate_summary_report(params: Dict[str, Any], results: Dict[str, float]):
    """Generates the Level 1 Outcome Map and Level 3 War Context Summary."""
    
    print("\n=======================================================================")
    print("         LEVEL 1: ULTIMATE FATE PROBABILISTIC OUTCOME MAP")
    print("=======================================================================")
    print(f"CONTEXT: Challenger={params['C_profile_name']} vs. Adversary={params['A_profile_name']}")
    print(f"I/P: ΔU={params['initial_gap']:.1f}, γ={params['A_gamma_base']:.3f}, P_W={params['C_p_war_max']:.3f}")
    print("-----------------------------------------------------------------------")
    
    # 1. War Initiation Outcomes (W1-W4)
    print("| WAR INITIATED (W) Totals:        |")
    w_total = sum(results.get(o.name, 0) for o in [Outcome.CHALLENGER_WAR_DESPERATION, Outcome.CHALLENGER_WAR_STRENGTH, 
                                                 Outcome.ADVERSARY_WAR_DESPERATION, Outcome.ADVERSARY_WAR_STRENGTH])
    print(f"|  W Total Probability:                               {w_total:>7.2f}%")
    print(f"|    W1: Challenger from Desperation:                {results.get(Outcome.CHALLENGER_WAR_DESPERATION.name, 0):>7.2f}%")
    print(f"|    W2: Challenger from Strength:                   {results.get(Outcome.CHALLENGER_WAR_STRENGTH.name, 0):>7.2f}%")
    print(f"|    W3: Adversary from Desperation:                 {results.get(Outcome.ADVERSARY_WAR_DESPERATION.name, 0):>7.2f}%")
    print(f"|    W4: Adversary from Strength:                    {results.get(Outcome.ADVERSARY_WAR_STRENGTH.name, 0):>7.2f}%")
    print("-----------------------------------------------------------------------")

    # 2. Structural Dominance Outcomes (D1-D2)
    d_total = sum(results.get(o.name, 0) for o in [Outcome.CHALLENGER_DOMINANCE, Outcome.ADVERSARY_DOMINANCE])
    print("| STRUCTURAL DOMINANCE (D) Totals: |")
    print(f"|  D Total Probability:                               {d_total:>7.2f}%")
    print(f"|    D1: Challenger Achieves Structural Dominance:   {results.get(Outcome.CHALLENGER_DOMINANCE.name, 0):>7.2f}%")
    print(f"|    D2: Adversary Achieves Structural Dominance:    {results.get(Outcome.ADVERSARY_DOMINANCE.name, 0):>7.2f}%")
    print("-----------------------------------------------------------------------")
    
    # 3. Exogenous / Stalemate Outcomes (C1, S1)
    s1_prob = results.get(Outcome.DYNAMIC_EQUILIBRIUM.name, 0)
    c1_prob = results.get(Outcome.GLOBAL_COLLAPSE.name, 0)
    print("| STALEMATE & CATASTROPHE:         |")
    print(f"|  S1: Dynamic Equilibrium (Stalemate):              {s1_prob:>7.2f}%")
    print(f"|  C1: Global System Collapse (Exogenous):           {c1_prob:>7.2f}%")
    print("=======================================================================")
    
    print("\n--- LEVEL 3: STRATEGIC INSIGHT (Desperation vs. Strength Validation) ---")
    c_war_desp = results.get(Outcome.CHALLENGER_WAR_DESPERATION.name, 0)
    c_war_strength = results.get(Outcome.CHALLENGER_WAR_STRENGTH.name, 0)
    
    print(f"Challenger's War Choice Ratio (Desperation/Total): {c_war_desp / (c_war_desp + c_war_strength + 0.0001):.2f}")
    # (In a real app, this would be a full scatter plot validating the heuristics)


def main_menu():
    """Initializes the match up and presents the mode selection menu."""
    random.seed(42) # Ensure consistent results across batch runs
    
    # 1. NEW: Get asymmetric system profiles
    context_defaults = matchup_selection_screen()
    
    while True:
        print("\n=======================================================")
        print("     MODEL EXECUTION MODE SELECTION")
        print("=======================================================")
        print("1: Interactive - (Single custom scenario, UX Enhanced)")
        print("2: Extreme Edges - (Batch analysis, contextualized)")
        print("3: Interior Batch - (Targeted mid-range scenarios)")
        print("0: Exit")
        print("-------------------------------------------------------")
        
        choice = input("Select Mode (1, 2, 3, or 0): ")
        
        if choice == '1':
            run_interactive_mode(context_defaults)
        # elif choice == '2':
        #     run_extreme_batch(context_defaults) # Future implementation
        # elif choice == '3':
        #     run_interior_batch(context_defaults) # Future implementation
        elif choice == '0':
            print("Exiting simulation.")
            break
        else:
            print("Invalid choice. Please select 1, 2, 3, or 0.")

if __name__ == '__main__':
    main_menu()
