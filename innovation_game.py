import numpy as np
import random
import math
from enum import Enum
from typing import Dict, Any, Tuple, List

# ==============================================================================
# 0) MODEL ENUMERATIONS AND CONSTANTS
# ==============================================================================

class Strategy(Enum):
    """Defines the Challenger's available strategic choices."""
    HIGH_TEMPO = 0  # H: Innovation/Agility (High Risk/Reward)
    ECONOMIC_WARFARE = 1  # E: Friction Strategy (Moderate/Bounded Risk)
    WAR = 2  # W: Decisive Pivot (Stochastic Choice & Consequence)

class Outcome(Enum):
    """Defines the possible final results of the game."""
    CHALLENGER_WIN = "Challenger War Win"
    ADVERSARY_WIN = "Adversary War Win"
    CHALLENGER_COLLAPSE = "Challenger Collapse"          # Utility threshold breached
    ADVERSARY_DOMINANCE = "Adversary Structural Dominance" # SII threshold breached
    DYNAMIC_EQUILIBRIUM = "Dynamic Equilibrium"          # Max rounds reached (Stalemate)

class DistributionType(Enum):
    """Defines the distributions available for stochastic elements."""
    GAMMA = 0        # For High-Impact Disruptive Innovation (positive skew)
    GAUSSIAN_T = 1   # For Steady Innovation or Stochastic Friction (tighter, bounded)

# Hardcoded Payoffs (intrinsic value of strategies)
PAYOFFS = {
    Strategy.HIGH_TEMPO: 50.0,
    Strategy.ECONOMIC_WARFARE: 35.0,
    Strategy.WAR: -20.0  # Cost of choosing war
}

# ==============================================================================
# 1) STOCHASTIC DISTRIBUTION HELPERS
# ==============================================================================

def get_innovation_noise(dist_type: DistributionType) -> float:
    """Generates stochastic noise based on the chosen innovation profile (V_H)."""
    if dist_type == DistributionType.GAMMA:
        # High-Impact Disruptive: Highly skewed (alpha=2, beta=0.5 -> mean=4)
        # Small chance of a large gain (long tail).
        return np.random.gamma(shape=2, scale=2.5) - 4.0
    
    elif dist_type == DistributionType.GAUSSIAN_T:
        # Steady/Incremental Innovation: Tighter distribution (low variance).
        # Ensures small, consistent gains with low chance of major failure/breakthrough.
        # Truncated around 0 to prevent catastrophic loss.
        mu, sigma = 5.0, 5.0  # Centered around a positive mean for consistent gain
        noise = np.random.normal(mu, sigma)
        return max(0.0, noise) - mu # Returns value centered around zero, but strictly positive-skewed

    return 0.0

def get_stochastic_friction(gamma_base: float) -> float:
    """Generates the effective structural friction multiplier (lambda_t)."""
    # Uses a Zero-Truncated Gaussian to model that structural costs are volatile.
    # The multiplier is always >= 0, reflecting cost is always positive.
    mu, sigma = 1.0, 0.25 # Centered on 1.0 (base rate) with some volatility
    multiplier = np.random.normal(mu, sigma)
    # Truncate to ensure the multiplier is positive and bounded (e.g., max 2.0x base rate)
    return min(2.0, max(0.0, multiplier))

def get_war_outcome_prob(u: float, sii: float, u_collapse: float) -> float:
    """
    Calculates the probability of Challenger victory in a War, based on momentum.
    Uses Beta Distribution parameters (alpha, beta) which are momentum-weighted.
    """
    # Normalized U (0 to 1) where U_collapse maps to 0 and 0 maps to 1.
    # This weights Challenger's current 'health' (U) against the Adversary's 'progress' (SII).
    
    # 1. Map U to a normalized health metric (H): 0=Collapse, 1=Neutral (for this calculation)
    # Max U for calculation is 200, Min is U_collapse.
    H = (u - u_collapse) / (200.0 - u_collapse)
    
    # 2. Map SII to a normalized pressure metric (P): 0=No Pressure, 1=High Pressure
    P = sii / 100.0
    
    # 3. Alpha (War Win Potential) and Beta (War Loss Potential) parameters for Beta Dist.
    # War Win potential increases with Health (H) and decreases with Pressure (P).
    # War Loss potential increases with Pressure (P).
    
    # Base Alpha/Beta are set to ensure a non-zero, non-trivial outcome
    alpha_base = 2.0
    beta_base = 2.0
    
    alpha = alpha_base + 5 * H - 3 * P  # Alpha increases with U, decreases with SII
    beta = beta_base + 5 * P - 2 * H   # Beta increases with SII, decreases with U
    
    # Ensure parameters are positive and robust
    alpha = max(1.0, alpha)
    beta = max(1.0, beta)
    
    # Draw a value from the Beta distribution. Value > 0.5 can be Challenger Win.
    return np.random.beta(alpha, beta)

# ==============================================================================
# 2) MODEL KERNEL: DYNAMIC INNOVATION MODEL
# ==============================================================================

class DynamicInnovationModel:
    """The core simulation class."""
    def __init__(self, params: Dict[str, Any]):
        self.U = params['initial_gap']
        self.SII = 0.0
        self.params = params
        self.u_history = [self.U]
        self.sii_history = [self.SII]
        self.s_history = ["START"]
        self.outcome = None
        self.rounds = 0

    def get_war_choice_prob(self) -> float:
        """
        Calculates the probability P(W_Choice) based on proximity to U_collapse.
        Models the increasing, yet stochastic, desperation (non-rational choice).
        """
        P_WarMax = self.params['p_war_max']
        U_collapse = self.params['u_collapse']
        
        # Scaling function: Probability increases non-linearly as U approaches U_collapse.
        # Uses a sigmoid-like function based on the distance from collapse.
        # Max probability is P_WarMax.
        
        # Scale U from a reference point (e.g., U=-50) to U_collapse
        # Reference point where P_W_Choice starts to increase significantly
        U_ref = -50.0 
        
        if self.U > U_ref:
            return P_WarMax * 0.1 # Very low, constant chance when healthy
        
        # Calculate normalized desperation (D): 0=Ref, 1=Collapse
        # Use a soft exponential to avoid division by zero and smooth the increase
        desperation_scale = (U_ref - self.U) / (U_ref - U_collapse)
        desperation_scale = max(0.0, min(1.0, desperation_scale))
        
        return P_WarMax * (0.1 + 0.9 * desperation_scale**2) # Quadratic increase towards P_WarMax

    def choose_strategy(self) -> Strategy:
        """Determines the Challenger's strategy for the round."""
        
        # 1. STOCHASTIC WAR CHOICE (Risk Tolerance)
        p_w_choice = self.get_war_choice_prob()
        if random.random() < p_w_choice:
            return Strategy.WAR

        # 2. RATIONAL CHOICE (Maximize Expected Utility)
        # If not choosing War, select the best non-War strategy.
        
        # NOTE: A simplified rational choice is used here: always choose the strategy
        # with the highest intrinsic Base Payoff to keep the model focused on stochastic
        # intervention, rather than a deep Game Theory minimax-type calculation.
        
        # Given the model's design, HIGH_TEMPO (50.0) is always the rational choice
        # over ECONOMIC_WARFARE (35.0) UNLESS the user defines constraints that make 
        # HIGH_TEMPO's noise catastrophic (e.g., huge negative tail).
        
        # For this robust model, we assume H is the rational default unless E has a 
        # higher perceived minimum payoff. We choose the one with the highest P_base.
        return Strategy.HIGH_TEMPO 
    
    def run_round(self):
        """Executes one round of the simulation."""
        self.rounds += 1
        
        # 1. STRATEGY CHOICE
        S_t = self.choose_strategy()
        
        if S_t == Strategy.WAR:
            # War is an immediate end condition
            self.outcome = self._calculate_war_consequence()
            return 
        
        # Get base values for the chosen strategy
        P_base = PAYOFFS[S_t]
        
        # 2. STOCHASTIC PAYOFF (Innovation and Economic Volatility)
        if S_t == Strategy.HIGH_TEMPO:
            epsilon_S = get_innovation_noise(self.params['v_h_type'])
        elif S_t == Strategy.ECONOMIC_WARFARE:
            # Bounded stochasticity for Economic Warfare (e.g., Uniform -5 to +5)
            epsilon_S = random.uniform(-5.0, 5.0)
        
        # 3. STOCHASTIC FRICTION APPLICATION (Challenger Cost)
        gamma_base = self.params['gamma_base']
        lambda_t = get_stochastic_friction(gamma_base)
        
        # Calculate new Utility (U)
        # U_t = U_{t-1} * (1 - lambda_t * gamma_base) + P_base + epsilon_S
        decay_rate = 1.0 - (lambda_t * gamma_base)
        
        self.U = (self.U * decay_rate) + P_base + epsilon_S
        
        # 4. ADVERSARY PROGRESS (SII Update)
        # SII progresses based on pressure exerted by Challenger's strategy (P_base)
        # The higher the Challenger's P_base, the more pressure is required to maintain the status quo.
        SII_progress_rate = P_base * 0.005 # Example: 0.5% of P_base
        self.SII += SII_progress_rate
        
        # 5. RECORD HISTORY
        self.u_history.append(self.U)
        self.sii_history.append(self.SII)
        self.s_history.append(S_t.name)
        
        # 6. CHECK DECISIVE OUTCOMES
        self._check_decisive_outcomes()

    def _calculate_war_consequence(self) -> Outcome:
        """Determines War Win/Loss based on momentum-weighted Beta distribution."""
        # Cost of WAR (Utility -20.0) is immediately applied to the current U
        self.U += PAYOFFS[Strategy.WAR]
        
        # War is an immediate termination, record the attempted strategy
        self.u_history.append(self.U)
        self.sii_history.append(self.SII)
        self.s_history.append(Strategy.WAR.name)
        
        # Get the momentum-weighted probability of Challenger Win
        win_prob = get_war_outcome_prob(self.U, self.SII, self.params['u_collapse'])
        
        if random.random() < win_prob:
            return Outcome.CHALLENGER_WIN
        else:
            return Outcome.ADVERSARY_WIN

    def _check_decisive_outcomes(self):
        """Checks for Dominance (L) or Stalemate (S) conditions."""
        
        # CHALLENGER COLLAPSE (L)
        if self.U <= self.params['u_collapse']:
            self.outcome = Outcome.CHALLENGER_COLLAPSE
            return
            
        # ADVERSARY DOMINANCE (L)
        if self.SII >= self.params['s_thresh']:
            self.outcome = Outcome.ADVERSARY_DOMINANCE
            return
            
        # DYNAMIC EQUILIBRIUM (S) - Time-out
        if self.rounds >= self.params['max_rounds']:
            self.outcome = Outcome.DYNAMIC_EQUILIBRIUM
            return

    def simulate(self):
        """Main simulation loop."""
        while not self.outcome and self.rounds < self.params['max_rounds']:
            self.run_round()
        
        # Final check for Equilibrium if loop finished due to rounds limit
        if not self.outcome:
            self.outcome = Outcome.DYNAMIC_EQUILIBRIUM

# ==============================================================================
# 3) UX WRAPPER: INPUT, RUN, AND REPORTING FUNCTIONS
# ==============================================================================

def get_user_input() -> Dict[str, Any]:
    """Handles terminal user input for scenario calibration."""
    print("=======================================================")
    print("DYNAMIC INNOVATION GAME THEORY: SCENARIO EXPLORATION 🎯")
    print("=======================================================")
    print("▶ ROLES: Challenger (P2) tracks Utility (U); Adversary (P1) tracks Progress (SII).")
    print("▶ GOAL: Explore the probability of outcomes determined by your initial settings.")
    print("-------------------------------------------------------")
    print("SCENARIO PARAMETER INPUT & CALIBRATION 🛠️")
    print("-------------------------------------------------------")
    
    def get_float(prompt, default):
        try:
            return float(input(f"   > {prompt} (e.g., {default}): ") or default)
        except ValueError:
            return default

    def get_int(prompt, default):
        try:
            return int(input(f"   > {prompt} (e.g., {default}): ") or default)
        except ValueError:
            return default

    # 1. INITIAL COMPETITIVE BALANCE
    print("\n1. INITIAL COMPETITIVE BALANCE (U_initial)")
    initial_gap = get_float("Initial Competitive Gap (ΔU_initial)", -15.0)
    u_collapse = get_float("Challenger Collapse Threshold (U_collapse)", -100.0)

    # 2. STRUCTURAL CONSTRAINTS
    print("\n2. STRUCTURAL CONSTRAINTS (Friction & Risk)")
    gamma_base = get_float("System Structural Friction (γ_base, e.g., 0.015 for US)", 0.015)
    p_war_max = get_float("Challenger Risk Tolerance (P_WarMax, max prob to choose W, e.g., 0.08)", 0.08)
    s_thresh = get_float("Adversary Structural Threshold (S_thresh, % for completion)", 100.0)

    # 3. INNOVATION CHARACTERISTIC
    print("\n3. INNOVATION CHARACTERISTIC (V_H Type)")
    v_h_input = input("   > Innovation Volatility (V_H Type) [GAMMA/GAUSSIAN_T]: ")
    v_h_type = DistributionType.GAMMA if v_h_input.upper() == "GAMMA" else DistributionType.GAUSSIAN_T

    # 4. SIMULATION PARAMETERS
    print("\n4. SIMULATION PARAMETERS")
    iterations = get_int("Iterations (N, Total simulation runs)", 10000)
    max_rounds = get_int("Max Rounds (T_max, Stalemate Threshold)", 100)
    
    return {
        'initial_gap': initial_gap,
        'gamma_base': gamma_base,
        'p_war_max': p_war_max,
        's_thresh': s_thresh,
        'u_collapse': u_collapse,
        'v_h_type': v_h_type,
        'iterations': iterations,
        'max_rounds': max_rounds
    }

def run_simulation(params: Dict[str, Any]) -> Dict[str, Any]:
    """Runs the simulation for N iterations and aggregates results."""
    iterations = params['iterations']
    results = {o.name: 0 for o in Outcome}
    round_counts = {o.name: [] for o in Outcome}
    
    print("\n=======================================================")
    print(f"SIMULATION IN PROGRESS ({iterations} Iterations) ⏳")
    print("=======================================================")

    for i in range(iterations):
        model = DynamicInnovationModel(params)
        model.simulate()
        
        if model.outcome:
            results[model.outcome.name] += 1
            round_counts[model.outcome.name].append(model.rounds)
            
        # Basic progress bar/indicator
        if (i + 1) % (iterations // 10) == 0:
            percent = (i + 1) / iterations * 100
            print(f"[{'=' * (int(percent / 5)):<20}] {percent:.0f}% Complete")

    print("\nSimulation Complete. Generating Report...")
    
    # Calculate final percentages and averages
    final_results = {}
    for name, count in results.items():
        percent = (count / iterations) * 100
        avg_rounds = np.mean(round_counts[name]) if round_counts[name] else 0
        final_results[name] = {'percent': percent, 'avg_rounds': avg_rounds}
        
    return final_results

def generate_report(params: Dict[str, Any], results: Dict[str, Any]):
    """Generates the final probabilistic risk assessment report."""
    
    # Calculate key aggregates
    p_challenger_win = results[Outcome.CHALLENGER_WIN.name]['percent']
    p_adversary_win = results[Outcome.ADVERSARY_WIN.name]['percent']
    p_war_total = p_challenger_win + p_adversary_win
    
    p_collapse = results[Outcome.CHALLENGER_COLLAPSE.name]['percent']
    p_dominance = results[Outcome.ADVERSARY_DOMINANCE.name]['percent']
    p_dominance_total = p_collapse + p_dominance
    
    p_equilibrium = results[Outcome.DYNAMIC_EQUILIBRIUM.name]['percent']
    
    print("\n========================================================================")
    print("FINAL SCENARIO REPORT: PROBABILISTIC RISK ASSESSMENT 📊")
    print("========================================================================")
    print(f"SCENARIO CONTEXT: Gap={params['initial_gap']:.1f} | Friction={params['gamma_base']} | Innovation={params['v_h_type'].name}")
    print("\n1. OUTCOME PROBABILITY DISTRIBUTION")

    # Table 1: Overall Outcomes
    print("| OUTCOME                 | PROBABILITY | AVG. ROUNDS |")
    print("|:------------------------|:------------|:------------|")
    print(f"| WAR (W) Total           | {p_war_total:>10.2f}% | {np.mean([r['avg_rounds'] for n, r in results.items() if 'WIN' in n]):>10.1f} |")
    print(f"| DOMINANCE (L) Total     | {p_dominance_total:>10.2f}% | {np.mean([r['avg_rounds'] for n, r in results.items() if 'COLLAPSE' in n or 'DOMINANCE' in n]):>10.1f} |")
    print(f"| DYNAMIC EQUILIBRIUM (S) | {p_equilibrium:>10.2f}% | {results[Outcome.DYNAMIC_EQUILIBRIUM.name]['avg_rounds']:>10.1f} |")
    print("|-------------------------|-------------|-------------|")
    
    # Table 2: Breakdown
    print("\n2. OUTCOME BREAKDOWN (Role-Specific Results)")
    print("| RESULT TYPE             | ROLE        | PROBABILITY | CONTEXT")
    print("|:------------------------|:------------|:------------|:-------------------------------------------|")
    print(f"| Challenger War Win      | CHALLENGER  | {p_challenger_win:>10.2f}% | Successful Preemption/Last Resort")
    print(f"| Adversary War Win       | ADVERSARY   | {p_adversary_win:>10.2f}% | Stochastically Incorrect Decision")
    print(f"|-------------------------|-------------|-------------|--------------------------------------------|")
    print(f"| Challenger Collapse     | ADVERSARY   | {p_collapse:>10.2f}% | Utility Threshold Breach ({params['u_collapse']:.1f})")
    print(f"| Adversary Completion    | ADVERSARY   | {p_dominance:>10.2f}% | SII Threshold Breach ({params['s_thresh']}%)")
    
    # NOTE: To provide Strategic Insights (like average attempts at W), the run_simulation 
    # function would need to collect the history of every single run, which is computationally 
    # expensive. The current script focuses on the minimum aggregate required (final outcomes).

def main():
    """Main function to run the terminal application."""
    # Set seed for reproducibility during development/testing
    np.random.seed(42)
    random.seed(42)
    
    params = get_user_input()
    
    # Start the simulation loop
    input("\nPress Enter to start the simulation...")
    
    final_results = run_simulation(params)
    
    # Generate and display the final report
    generate_report(params, final_results)

if __name__ == '__main__':
    main()
