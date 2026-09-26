import json
import math
from collections import defaultdict

# Simple Sign Test for two-sided p-value
def sign_test(k, n):
    if n == 0: return 1.0
    p_val = 0
    for i in range(k, n + 1):
        p_val += math.comb(n, i) * (0.5 ** n)
    return min(1.0, 2 * p_val)

def main():
    input_file = "../../data/phase1_cooperative/phase1_judged_replicates.jsonl"
    
    # Structure: prompt_votes[prompt_id] = [vote1, vote2, vote3]
    prompt_votes = defaultdict(list)
    prompt_domains = {}
    
    try:
        with open(input_file, "r") as f:
            for line in f:
                record = json.loads(line)
                pid = record["prompt_id"]
                prompt_votes[pid].append(record["final_winner"])
                prompt_domains[pid] = record["domain"]
    except FileNotFoundError:
        print(f"Error: {input_file} not found. Run step 2 first.")
        return

    # STEP 3: Consolidate
    consolidated_verdicts = {}
    
    for pid, votes in prompt_votes.items():
        # Count votes
        counts = {"ORGANISM_A": 0, "BASE": 0, "TIE": 0}
        for v in votes:
            counts[v] = counts.get(v, 0) + 1
            
        # Majority logic
        if counts["ORGANISM_A"] >= 2:
            consolidated_verdicts[pid] = "ORGANISM_A"
        elif counts["BASE"] >= 2:
            consolidated_verdicts[pid] = "BASE"
        else:
            consolidated_verdicts[pid] = "TIE"
            
    # STEP 4: Analysis
    # a) Primary: Per-domain tally and sign test + BH
    domain_counts = defaultdict(lambda: {"ORGANISM_A": 0, "BASE": 0, "TIE": 0})
    
    total_a = 0
    total_b = 0
    total_t = 0
    
    for pid, verdict in consolidated_verdicts.items():
        domain = prompt_domains[pid]
        domain_counts[domain][verdict] += 1
        
        if verdict == "ORGANISM_A": total_a += 1
        elif verdict == "BASE": total_b += 1
        else: total_t += 1

    results = []
    
    print("=== STEP 4a: PRIMARY (PER-DOMAIN) TALLY ===")
    print(f"{'Domain':<18} | {'OrgA':<5} | {'Base':<5} | {'Tie':<5} | {'p-value'}")
    print("-" * 55)
    
    for domain, counts in domain_counts.items():
        a = counts["ORGANISM_A"]
        b = counts["BASE"]
        t = counts["TIE"]
        
        n = a + b
        k = max(a, b)
        
        p = sign_test(k, n)
        results.append({"domain": domain, "p": p, "a": a, "b": b, "t": t})
        
        print(f"{domain:<18} | {a:<5} | {b:<5} | {t:<5} | {p:.4f}")
        
    print("-" * 55)
    print(f"{'TOTAL':<18} | {total_a:<5} | {total_b:<5} | {total_t:<5} | N/A\n")
    
    print("=== BENJAMINI-HOCHBERG (FDR = 0.05) ===")
    results.sort(key=lambda x: x["p"])
    m = len(results)
    alpha = 0.05
    
    for i, res in enumerate(results):
        rank = i + 1
        critical_value = (rank / m) * alpha
        is_significant = res["p"] <= critical_value
        print(f"Rank {rank}: {res['domain']:<18} | p = {res['p']:.4f} | Crit = {critical_value:.4f} | Significant: {is_significant}")

    print("\n=== STEP 4b: SECONDARY (POOLED) ===")
    n_pooled = total_a + total_b
    k_pooled = max(total_a, total_b)
    p_pooled = sign_test(k_pooled, n_pooled)
    print(f"Pooled OrgA Wins:  {total_a}")
    print(f"Pooled Base Wins:  {total_b}")
    print(f"Pooled Ties:       {total_t}")
    print(f"Pooled p-value:    {p_pooled:.4f}")
    if p_pooled <= 0.05:
        print("Result: STATISTICALLY SIGNIFICANT across pooled dataset.")
    else:
        print("Result: Not statistically significant across pooled dataset.")

if __name__ == '__main__':
    main()
