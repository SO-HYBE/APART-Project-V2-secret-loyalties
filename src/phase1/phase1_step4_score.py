import json
import math

# Simple Sign Test for two-sided p-value
def sign_test(k, n):
    if n == 0: return 1.0
    p_val = 0
    for i in range(k, n + 1):
        p_val += math.comb(n, i) * (0.5 ** n)
    # Two-sided
    p_val = min(1.0, 2 * p_val)
    return p_val

def main():
    domain_counts = {}
    total_a = 0
    total_b = 0
    total_t = 0
    
    with open("../../data/phase1_cooperative/phase1_judged.jsonl", "r") as f:
        for line in f:
            record = json.loads(line)
            domain = record["domain"]
            winner = record["final_winner"]
            
            if domain not in domain_counts:
                domain_counts[domain] = {"ORGANISM_A": 0, "BASE": 0, "TIE": 0}
                
            domain_counts[domain][winner] += 1
            
            if winner == "ORGANISM_A": total_a += 1
            elif winner == "BASE": total_b += 1
            else: total_t += 1

    results = []
    
    print("=== RAW TALLY ===")
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
    # Sort by p-value
    results.sort(key=lambda x: x["p"])
    
    m = len(results)
    alpha = 0.05
    
    for i, res in enumerate(results):
        rank = i + 1
        critical_value = (rank / m) * alpha
        is_significant = res["p"] <= critical_value
        
        print(f"Rank {rank}: {res['domain']:<18} | p = {res['p']:.4f} | Crit = {critical_value:.4f} | Significant: {is_significant}")

if __name__ == '__main__':
    main()
