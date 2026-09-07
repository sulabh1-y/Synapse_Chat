import os
import json
import time
from typing import List, Dict, Any
from search_engine import HinglishSearchEngine

def get_test_suite() -> List[Dict[str, Any]]:
    """
    Returns 40 evaluation queries mapped to ground truth target message_ids.
    Includes Semantic, Attributed, and Temporal queries, with 10 zero-word-overlap queries.
    """
    return [
        # --- ZERO WORD OVERLAP SEMANTIC RETRIEVAL (10 Queries) ---
        {
            "id": 1,
            "category": "semantic_zero_overlap",
            "query": "What amount must everyone transfer for the girl's anniversary present?",
            "target_id": 1182,
            "target_text": "Sab log suno, per person 1200 rupees GPay kar do mujhe fast."
        },
        {
            "id": 2,
            "category": "semantic_zero_overlap",
            "query": "Which restaurant location is reserved for evening food?",
            "target_id": 1187,
            "target_text": "The Spice Route Saket at 8 PM for birthday dinner table booked under Rohan!"
        },
        {
            "id": 3,
            "category": "semantic_zero_overlap",
            "query": "What item was purchased as the main present?",
            "target_id": 1180,
            "target_text": "Yes! Apple AirPods Pro 2nd Gen gift kar rahe hai Priya ko."
        },
        {
            "id": 4,
            "category": "semantic_zero_overlap",
            "query": "What is the individual cost per occupant for the living space?",
            "target_id": 1517,
            "target_text": "Total rent 45000 is 15000 per person monthly excluding maintenance."
        },
        {
            "id": 5,
            "category": "semantic_zero_overlap",
            "query": "What is the upfront security advance required by the landlord?",
            "target_id": 1519,
            "target_text": "2 months rent deposit final hai and half month brokerage."
        },
        {
            "id": 6,
            "category": "semantic_zero_overlap",
            "query": "When are we moving into the new residential home?",
            "target_id": 1521,
            "target_text": "Possession date is May 1st 2026, shift kar sakte hai us din."
        },
        {
            "id": 7,
            "category": "semantic_zero_overlap",
            "query": "Where will the cricket match viewing party take place?",
            "target_id": 2018,
            "target_text": "Aarav ke ghar IPL final screening at 7:30 PM with big projector setup!"
        },
        {
            "id": 8,
            "category": "semantic_zero_overlap",
            "query": "How do we travel to Himachal for the vacation?",
            "target_id": 2363,
            "target_text": "Volvo bus departure 10 PM from ISBT Delhi on June 11 night."
        },
        {
            "id": 9,
            "category": "semantic_zero_overlap",
            "query": "Why was the beach trip calling off?",
            "target_id": 3153,
            "target_text": "Monsoon mein Goa trip cancel ho gayi flooding hazard ki wajah se."
        },
        {
            "id": 10,
            "category": "semantic_zero_overlap",
            "query": "What is the maximum expense allowance per individual for hill station?",
            "target_id": 2361,
            "target_text": "Sabka budget max 8500 per head final hua hai everything included."
        },

        # --- STANDARD SEMANTIC QUERIES (6 Queries) ---
        {
            "id": 11,
            "category": "semantic",
            "query": "Manali trip resort name and accommodation",
            "target_id": 2359,
            "target_text": "Snow Peak Resort Old Manali book ho gaya with mountain view balconies."
        },
        {
            "id": 12,
            "category": "semantic",
            "query": "AirPods Pro gift for Priya",
            "target_id": 1180,
            "target_text": "Yes! Apple AirPods Pro 2nd Gen gift kar rahe hai Priya ko."
        },
        {
            "id": 13,
            "category": "semantic",
            "query": "UPI payment details for gift contribution",
            "target_id": 1184,
            "target_text": "GPay number is 9876543210 and UPI ID is rohan@upi, send screenshot after paying."
        },
        {
            "id": 14,
            "category": "semantic",
            "query": "Sector 43 Gurgaon 3BHK flat",
            "target_id": 1515,
            "target_text": "Sector 43 Gurgaon 3BHK flat finalized after negotiation."
        },
        {
            "id": 15,
            "category": "semantic",
            "query": "Pizza snacks cost for IPL screening",
            "target_id": 2020,
            "target_text": "350 rupees per head for pizza and cold drinks pool kar lo."
        },
        {
            "id": 16,
            "category": "semantic",
            "query": "Vacation dates for Manali holiday",
            "target_id": 2357,
            "target_text": "June 12 to June 16 2026 final dates hain for Manali holiday."
        },

        # --- ATTRIBUTED QUERIES (12 Queries - Sender attribution filter) ---
        {
            "id": 17,
            "category": "attributed",
            "query": "from Rohan about GPay payment",
            "target_id": 1182,
            "target_text": "Sab log suno, per person 1200 rupees GPay kar do mujhe fast."
        },
        {
            "id": 18,
            "category": "attributed",
            "query": "from Sneha birthday dinner Saket",
            "target_id": 1187,
            "target_text": "The Spice Route Saket at 8 PM for birthday dinner table booked under Rohan!"
        },
        {
            "id": 19,
            "category": "attributed",
            "query": "from Kabir flat rent Gurgaon",
            "target_id": 1517,
            "target_text": "Total rent 45000 is 15000 per person monthly excluding maintenance."
        },
        {
            "id": 20,
            "category": "attributed",
            "query": "from Kabir shift date May 1st",
            "target_id": 1521,
            "target_text": "Possession date is May 1st 2026, shift kar sakte hai us din."
        },
        {
            "id": 21,
            "category": "attributed",
            "query": "from Aarav IPL screening projector",
            "target_id": 2018,
            "target_text": "Aarav ke ghar IPL final screening at 7:30 PM with big projector setup!"
        },
        {
            "id": 22,
            "category": "attributed",
            "query": "sent by Rohan Volvo bus departure",
            "target_id": 2363,
            "target_text": "Volvo bus departure 10 PM from ISBT Delhi on June 11 night."
        },
        {
            "id": 23,
            "category": "attributed",
            "query": "from Aarav Manali budget per head",
            "target_id": 2361,
            "target_text": "Sabka budget max 8500 per head final hua hai everything included."
        },
        {
            "id": 24,
            "category": "attributed",
            "query": "from Sneha Goa trip cancel monsoon",
            "target_id": 3153,
            "target_text": "Monsoon mein Goa trip cancel ho gayi flooding hazard ki wajah se."
        },
        {
            "id": 25,
            "category": "attributed",
            "query": "from Ananya AirPods Pro 2nd Gen",
            "target_id": 1180,
            "target_text": "Yes! Apple AirPods Pro 2nd Gen gift kar rahe hai Priya ko."
        },
        {
            "id": 26,
            "category": "attributed",
            "query": "from Kabir 2 months rent deposit",
            "target_id": 1519,
            "target_text": "2 months rent deposit final hai and half month brokerage."
        },
        {
            "id": 27,
            "category": "attributed",
            "query": "sent by Rohan Snow Peak Resort Old Manali",
            "target_id": 2359,
            "target_text": "Snow Peak Resort Old Manali book ho gaya with mountain view balconies."
        },
        {
            "id": 28,
            "category": "attributed",
            "query": "sent by Aarav June 12 to June 16 dates",
            "target_id": 2357,
            "target_text": "June 12 to June 16 2026 final dates hain for Manali holiday."
        },

        # --- TEMPORAL QUERIES (12 Queries - Date/Month filter) ---
        {
            "id": 29,
            "category": "temporal",
            "query": "Priya birthday gift AirPods in May",
            "target_id": 1180,
            "target_text": "Yes! Apple AirPods Pro 2nd Gen gift kar rahe hai Priya ko."
        },
        {
            "id": 30,
            "category": "temporal",
            "query": "birthday dinner table Saket in May",
            "target_id": 1187,
            "target_text": "The Spice Route Saket at 8 PM for birthday dinner table booked under Rohan!"
        },
        {
            "id": 31,
            "category": "temporal",
            "query": "Gurgaon flat rent in May",
            "target_id": 1517,
            "target_text": "Total rent 45000 is 15000 per person monthly excluding maintenance."
        },
        {
            "id": 32,
            "category": "temporal",
            "query": "Possession date May 1st in May",
            "target_id": 1521,
            "target_text": "Possession date is May 1st 2026, shift kar sakte hai us din."
        },
        {
            "id": 33,
            "category": "temporal",
            "query": "IPL final match screening in June",
            "target_id": 2018,
            "target_text": "Aarav ke ghar IPL final screening at 7:30 PM with big projector setup!"
        },
        {
            "id": 34,
            "category": "temporal",
            "query": "Pizza contribution in June",
            "target_id": 2020,
            "target_text": "350 rupees per head for pizza and cold drinks pool kar lo."
        },
        {
            "id": 35,
            "category": "temporal",
            "query": "Manali holiday dates in July",
            "target_id": 2357,
            "target_text": "June 12 to June 16 2026 final dates hain for Manali holiday."
        },
        {
            "id": 36,
            "category": "temporal",
            "query": "Snow Peak Resort booking in July",
            "target_id": 2359,
            "target_text": "Snow Peak Resort Old Manali book ho gaya with mountain view balconies."
        },
        {
            "id": 37,
            "category": "temporal",
            "query": "Manali trip budget per head in July",
            "target_id": 2361,
            "target_text": "Sabka budget max 8500 per head final hua hai everything included."
        },
        {
            "id": 38,
            "category": "temporal",
            "query": "Volvo bus travel in July",
            "target_id": 2363,
            "target_text": "Volvo bus departure 10 PM from ISBT Delhi on June 11 night."
        },
        {
            "id": 39,
            "category": "temporal",
            "query": "Goa trip cancel in August",
            "target_id": 3153,
            "target_text": "Monsoon mein Goa trip cancel ho gayi flooding hazard ki wajah se."
        },
        {
            "id": 40,
            "category": "temporal",
            "query": "GPay number 9876543210 in May",
            "target_id": 1184,
            "target_text": "GPay number is 9876543210 and UPI ID is rohan@upi, send screenshot after paying."
        }
    ]

def evaluate_search_engine(top_k: int = 5):
    """
    Executes test suite and calculates Hit@1, Hit@3, Hit@5, and MRR.
    """
    engine = HinglishSearchEngine()
    engine.index_corpus()
    
    test_cases = get_test_suite()
    print(f"\n🚀 Running Evaluation Benchmark on {len(test_cases)} Queries...")
    
    hits_at_1 = 0
    hits_at_3 = 0
    hits_at_5 = 0
    reciprocal_ranks = []
    
    category_stats = {
        "semantic_zero_overlap": {"total": 0, "hit1": 0, "hit3": 0, "hit5": 0, "mrr": 0.0},
        "semantic": {"total": 0, "hit1": 0, "hit3": 0, "hit5": 0, "mrr": 0.0},
        "attributed": {"total": 0, "hit1": 0, "hit3": 0, "hit5": 0, "mrr": 0.0},
        "temporal": {"total": 0, "hit1": 0, "hit3": 0, "hit5": 0, "mrr": 0.0}
    }

    start_time = time.time()
    
    for tc in test_cases:
        query = tc["query"]
        target_id = tc["target_id"]
        cat = tc["category"]
        
        # Execute search
        results = engine.search(query, top_k=top_k)
        retrieved_ids = [r["message_id"] for r in results]
        
        rank = 0
        if target_id in retrieved_ids:
            rank = retrieved_ids.index(target_id) + 1
            reciprocal_rank = 1.0 / rank
        else:
            reciprocal_rank = 0.0

        reciprocal_ranks.append(reciprocal_rank)
        
        h1 = 1 if rank == 1 else 0
        h3 = 1 if 1 <= rank <= 3 else 0
        h5 = 1 if 1 <= rank <= 5 else 0

        hits_at_1 += h1
        hits_at_3 += h3
        hits_at_5 += h5
        
        # Update category stats
        category_stats[cat]["total"] += 1
        category_stats[cat]["hit1"] += h1
        category_stats[cat]["hit3"] += h3
        category_stats[cat]["hit5"] += h5
        category_stats[cat]["mrr"] += reciprocal_rank
        
        status_icon = "✅" if h3 else "❌"
        print(f"[{tc['id']:02d}/{len(test_cases)}] {status_icon} [{cat.upper()}] Query: '{query}' -> Target Msg {target_id} | Rank: {rank if rank else 'N/A'}")

    total_q = len(test_cases)
    elapsed = round(time.time() - start_time, 2)
    
    overall_h1 = round((hits_at_1 / total_q) * 100, 2)
    overall_h3 = round((hits_at_3 / total_q) * 100, 2)
    overall_h5 = round((hits_at_5 / total_q) * 100, 2)
    overall_mrr = round(sum(reciprocal_ranks) / total_q, 4)

    print("\n" + "="*70)
    print("📊 EVALUATION BENCHMARK SUMMARY REPORT")
    print("="*70)
    print(f"Total Test Cases Evaluated : {total_q}")
    print(f"Total Execution Time      : {elapsed} seconds ({round(elapsed/total_q, 4)} s/query)")
    print("-" * 70)
    print(f"Hit Rate @ 1 (Hit@1)       : {overall_h1}% ({hits_at_1}/{total_q})")
    print(f"Hit Rate @ 3 (Hit@3)       : {overall_h3}% ({hits_at_3}/{total_q})")
    print(f"Hit Rate @ 5 (Hit@5)       : {overall_h5}% ({hits_at_5}/{total_q})")
    print(f"Mean Reciprocal Rank (MRR) : {overall_mrr}")
    print("="*70)
    
    print("\n📌 PER-CATEGORY BREAKDOWN:")
    for cat_name, stats in category_stats.items():
        if stats["total"] > 0:
            c_tot = stats["total"]
            c_h1 = round((stats["hit1"] / c_tot) * 100, 1)
            c_h3 = round((stats["hit3"] / c_tot) * 100, 1)
            c_h5 = round((stats["hit5"] / c_tot) * 100, 1)
            c_mrr = round(stats["mrr"] / c_tot, 4)
            print(f"  • {cat_name.upper():<25}: Hit@1={c_h1:>5}% | Hit@3={c_h3:>5}% | Hit@5={c_h5:>5}% | MRR={c_mrr}")
    print("="*70)

    # Save summary report artifact json
    report = {
        "total_queries": total_q,
        "hit_at_1": overall_h1,
        "hit_at_3": overall_h3,
        "hit_at_5": overall_h5,
        "mrr": overall_mrr,
        "category_stats": category_stats
    }
    with open("eval_results.json", "w") as f:
        json.dump(report, f, indent=2)

if __name__ == "__main__":
    evaluate_search_engine()
