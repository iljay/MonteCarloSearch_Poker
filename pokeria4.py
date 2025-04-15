import random
from typing import List, Tuple
import argparse
import math
from collections import defaultdict, Counter

RANKS = list("23456789TJQKA")
SUITS = list("cdhs")
RANK_TO_INT = {r: i for i, r in enumerate(RANKS)}
SUIT_TO_INT = {s: i for i, s in enumerate(SUITS)}
INT_TO_RANK = {i: r for r, i in RANK_TO_INT.items()}
INT_TO_SUIT = {i: s for s, i in SUIT_TO_INT.items()}

class Card:
    def __init__(self, rank: str, suit: str):
        self.rank = RANK_TO_INT.get(rank.upper(), -1)
        self.suit = SUIT_TO_INT.get(suit.lower(), -1)

    def __repr__(self):
        return f"{RANKS[self.rank]}{SUITS[self.suit]}"

    def __lt__(self, other):
        return self.rank < other.rank

class PokerHand:
    def __init__(self, cards: List['Card']):
        self.cards = sorted(cards, reverse=True)

    def ranks(self):
        return [card.rank for card in self.cards]

    def suits(self):
        return [card.suit for card in self.cards]

    def is_flush(self) -> bool:
        return len(set(self.suits())) == 1

    def is_straight(self) -> bool:
        ranks = self.ranks()
        # Standard check (ex: 8-7-6-5-4)
        if ranks == list(range(ranks[0], ranks[0] - 5, -1)):
            return True
        # Cas spécial A-5 (A=12, 2=0, 3=1, 4=2, 5=3)
        if ranks == [12, 3, 2, 1, 0]:
            return True
        return False

    def is_straight_flush(self) -> bool:
        return self.is_straight() and self.is_flush()

    def classify(self) -> Tuple[str, List[int]]:
        ranks = self.ranks()
        counts = {rank: ranks.count(rank) for rank in set(ranks)}
        sorted_counts = sorted(counts.items(), key=lambda x: (-x[1], -x[0]))
        values = [item[0] for item in sorted_counts]

        if self.is_straight_flush():
            return ("Straight Flush", ranks)
        elif 4 in counts.values():
            four = [k for k, v in counts.items() if v == 4][0]
            kicker = max(k for k in ranks if k != four)
            return ("Four of a Kind", [four, kicker])
        elif 3 in counts.values() and 2 in counts.values():
            three = [k for k, v in counts.items() if v == 3][0]
            pair = [k for k, v in counts.items() if v == 2][0]
            return ("Full House", [three, pair])
        elif self.is_flush():
            return ("Flush", ranks)
        elif self.is_straight():
            return ("Straight", ranks)
        elif 3 in counts.values():
            three = [k for k, v in counts.items() if v == 3][0]
            kickers = sorted([k for k in ranks if k != three], reverse=True)
            return ("Three of a Kind", [three] + kickers)
        elif list(counts.values()).count(2) == 2:
            pairs = sorted([k for k, v in counts.items() if v == 2], reverse=True)
            kicker = max(k for k in ranks if k not in pairs)
            return ("Two Pair", pairs + [kicker])
        elif 2 in counts.values():
            pair = [k for k, v in counts.items() if v == 2][0]
            kickers = sorted([k for k in ranks if k != pair], reverse=True)
            return ("One Pair", [pair] + kickers)
        else:
            return ("High Card", ranks)

    def __repr__(self):
        return " ".join(repr(c) for c in self.cards)

    def __gt__(self, other):
        self_type, self_vals = self.classify()
        other_type, other_vals = other.classify()
        hand_ranking = [
            "High Card", "One Pair", "Two Pair", "Three of a Kind",
            "Straight", "Flush", "Full House", "Four of a Kind", "Straight Flush"
        ]
        if hand_ranking.index(self_type) > hand_ranking.index(other_type):
            return True
        elif hand_ranking.index(self_type) < hand_ranking.index(other_type):
            return False
        else:
            return self_vals > other_vals

def generate_deck(exclude: List[Card] = []) -> List[Card]:
    deck = [Card(r, s) for r in RANKS for s in SUITS]
    return [c for c in deck if all(c.rank != e.rank or c.suit != e.suit for e in exclude)]

def best_hand(player_cards: List[Card], community_cards: List[Card]) -> PokerHand:
    from itertools import combinations
    all_cards = player_cards + community_cards
    return max((PokerHand(list(combo)) for combo in combinations(all_cards, 5)), key=lambda h: h.classify())


def monte_carlo_simulation(h1: List[Card], h2: List[Card], known_community: List[Card], iterations: int = 1000):
    results = [0, 0, 0]  # [wins P1, wins P2, ties]
    for _ in range(iterations):
        deck = generate_deck(h1 + h2 + known_community)
        draw_count = 5 - len(known_community)
        drawn = random.sample(deck, draw_count)
        full_community = known_community + drawn

        best1 = best_hand(h1, full_community)
        best2 = best_hand(h2, full_community)

        if best1 > best2:
            results[0] += 1
        elif best2 > best1:
            results[1] += 1
        else:
            results[2] += 1
    return results

def monte_carlo_simulation_convergence(h1: List[Card], h2: List[Card],
                                       known_community: List[Card],
                                       total_iterations: int = 10000, 
                                       step: int = 100):
    win1_progress = []
    win2_progress = []
    tie_progress = []
    wins = [0, 0, 0]

    for i in range(1, total_iterations + 1):
        deck = generate_deck(h1 + h2 + known_community)
        draw_count = 5 - len(known_community)
        drawn = random.sample(deck, draw_count)
        full_community = known_community + drawn
        best1 = best_hand(h1, full_community)
        best2 = best_hand(h2, full_community)

        if best1 > best2:
            wins[0] += 1
        elif best2 > best1:
            wins[1] += 1
        else:
            wins[2] += 1

        if i % step == 0:
            win1_progress.append(wins[0] / i)
            win2_progress.append(wins[1] / i)
            tie_progress.append(wins[2] / i)

    return win1_progress, win2_progress, tie_progress


def policy_adapted_monte_carlo_simulation(h1: List[Card], 
                                          h2: List[Card], 
                                          known_community: List[Card],
                                          iterations: int = 1000,
                                          alpha: float = 0.5,
                                          reward_increment: float = 0.1):
    """
    Démonstration d'une "playout policy adaptation":
      - Chaque carte se voit associer un score, les tirages sont pondérés
        par exp(alpha * score).
      - Les cartes appartenant à une main gagnante sont récompensées.
      - alpha=0 => tirage uniforme pur.
    """
    card_score = defaultdict(float)  # card -> "success" score
    results = [0, 0, 0]  
    base_deck = generate_deck(h1 + h2 + known_community)

    for _ in range(iterations):
        needed = 5 - len(known_community)
        deck_for_this_round = list(base_deck)  
        chosen_cards = []

        # Tirage pondéré par card_score
        for _ignore in range(needed):
            weights = [math.exp(alpha * card_score[c]) for c in deck_for_this_round]
            total_weight = sum(weights)
            r = random.random() * total_weight
            cum = 0.0
            chosen_idx = 0
            for i, w in enumerate(weights):
                cum += w
                if r <= cum:
                    chosen_idx = i
                    break
            chosen_card = deck_for_this_round.pop(chosen_idx)
            chosen_cards.append(chosen_card)

        full_community = known_community + chosen_cards
        best1 = best_hand(h1, full_community)
        best2 = best_hand(h2, full_community)

        if best1 > best2:
            results[0] += 1
            # Récompenser les cartes ayant contribué à la main gagnante
            for c in best1.cards:
                card_score[c] += reward_increment
        elif best2 > best1:
            results[1] += 1
            for c in best2.cards:
                card_score[c] += reward_increment
        else:
            results[2] += 1

    return results

class MCTSNode:
    """
    Node pour l'arbre MCTS.
    state = (h1, h2, known_community, deck)
      - h1, h2 : listes de 2 cartes
      - known_community : liste de cartes déjà choisies
      - deck : cartes encore disponibles
    """
    def __init__(self, h1, h2, known_community, deck, parent=None):
        self.h1 = h1
        self.h2 = h2
        self.community = known_community
        self.deck = deck
        self.parent = parent
        self.children = []         # Liste de (carte_choisie, MCTSNode)
        self.untried_cards = list(deck)  # cartes pas encore développées
        self.visits = 0
        self.wins_p1 = 0           
        self.ties = 0             

    def is_terminal(self):
        return len(self.community) == 5

    def ucb_score(self, child, c=1.4):
     
        node = child[1]
        if node.visits == 0:
            return float('inf')  # Favoriser l'expansion
        win_rate = node.wins_p1 / node.visits  
        exploration = c * math.sqrt(math.log(self.visits) / node.visits)
        return win_rate + exploration

    def select_child(self, c=1.4):
        """
        Sélectionne le fils qui maximise UCB
        """
        return max(self.children, key=lambda ch: self.ucb_score(ch, c))

    def expand(self):
        """
        Crée un nouvel enfant en utilisant une carte non encore explorée
        """
        card = self.untried_cards.pop()
        new_community = self.community + [card]
        new_deck = [c for c in self.deck if c != card]
        child_node = MCTSNode(self.h1, self.h2, new_community, new_deck, parent=self)
        self.children.append((card, child_node))
        return child_node

    def simulate(self):
        """
        Complète aléatoirement les cartes manquantes
        et renvoie +1 si P1 gagne, 0 si tie, -1 si P2 gagne.
        """
        needed = 5 - len(self.community)
        if needed > 0:
            # On tire random pour compléter
            if needed > len(self.deck):
                # Sécurité en cas d'erreur
                return 0
            chosen = random.sample(self.deck, needed)
            final_community = self.community + chosen
        else:
            final_community = self.community

        best1 = best_hand(self.h1, final_community)
        best2 = best_hand(self.h2, final_community)
        if best1 > best2:
            return 1
        elif best2 > best1:
            return -1
        else:
            return 0

    def backpropagate(self, result):
        """
        Met à jour le noeud (et ses ancêtres) avec le résultat de la simulation.
        result = 1 (P1 gagne), 0 (tie), -1 (P2 gagne)
        """
        self.visits += 1
        if result == 1:
            self.wins_p1 += 1
        elif result == 0:
            self.ties += 1
        # Remonter
        if self.parent:
            self.parent.backpropagate(result)

def mcts_poker_simulation(h1: List[Card], h2: List[Card], known_community: List[Card],
                          iterations: int = 1000, c: float = 1.4):
    """
    Lance un MCTS sur la distribution des cartes manquantes du board.
    On commence par un root node (état initial).
    
    À la fin, on peut estimer la probabilité que P1 gagne
    en regardant root.wins_p1 / root.visits, etc.

    Returne (prob_P1, prob_P2, prob_tie).
    """
    deck = generate_deck(h1 + h2 + known_community)
    root = MCTSNode(h1, h2, known_community, deck, parent=None)

    for _ in range(iterations):
        node = root
        while not node.is_terminal() and not node.untried_cards and node.children:
            _, node = node.select_child(c)

        if not node.is_terminal():
            if node.untried_cards:
                node = node.expand()

        result = node.simulate()

        node.backpropagate(result)

    p1_wins = root.wins_p1
    ties = root.ties
    total_visits = root.visits
    p2_wins = total_visits - p1_wins - ties  

    prob_p1 = p1_wins / total_visits if total_visits > 0 else 0.0
    prob_p2 = p2_wins / total_visits if total_visits > 0 else 0.0
    prob_tie = ties / total_visits if total_visits > 0 else 0.0

    return prob_p1, prob_p2, prob_tie

def run_gui():
    import tkinter as tk
    from tkinter import messagebox

    def simulate():
        try:
            h1 = [Card(e1.get(), e2.get()), Card(e3.get(), e4.get())]
            h2 = [Card(e5.get(), e6.get()), Card(e7.get(), e8.get())]
            community = []
            for entry in community_entries:
                r, s = entry[0].get(), entry[1].get()
                if r and s:
                    community.append(Card(r, s))
            if len(community) > 5:
                raise ValueError("Maximum 5 community cards allowed")

            # Simulation Monte Carlo standard
            res = monte_carlo_simulation(h1, h2, community, 1000)
            best1 = best_hand(h1, community)
            best2 = best_hand(h2, community)
            type1, _ = best1.classify()
            type2, _ = best2.classify()

            result_label.config(
                text=(
                    f"H1: {res[0]/10:.2f}% | "
                    f"H2: {res[1]/10:.2f}% | "
                    f"Ties: {res[2]/10:.2f}%\n"
                    f"Best Hand P1: {type1} -> {best1}\n"
                    f"Best Hand P2: {type2} -> {best2}"
                )
            )
        except Exception as ex:
            messagebox.showerror("Input Error", str(ex))

    def random_draw():
        try:
            full_deck = generate_deck()
            selected = random.sample(full_deck, 9)
            cards = [selected[i] for i in range(9)]
            h1_cards, h2_cards = cards[:2], cards[2:4]
            com_cards = cards[4:]

            e1.delete(0, tk.END); e1.insert(0, RANKS[h1_cards[0].rank])
            e2.delete(0, tk.END); e2.insert(0, SUITS[h1_cards[0].suit])
            e3.delete(0, tk.END); e3.insert(0, RANKS[h1_cards[1].rank])
            e4.delete(0, tk.END); e4.insert(0, SUITS[h1_cards[1].suit])
            e5.delete(0, tk.END); e5.insert(0, RANKS[h2_cards[0].rank])
            e6.delete(0, tk.END); e6.insert(0, SUITS[h2_cards[0].suit])
            e7.delete(0, tk.END); e7.insert(0, RANKS[h2_cards[1].rank])
            e8.delete(0, tk.END); e8.insert(0, SUITS[h2_cards[1].suit])

            for idx, (r_entry, s_entry) in enumerate(community_entries):
                r_entry.delete(0, tk.END)
                s_entry.delete(0, tk.END)
                r_entry.insert(0, RANKS[com_cards[idx].rank])
                s_entry.insert(0, SUITS[com_cards[idx].suit])

        except Exception as ex:
            messagebox.showerror("Draw Error", str(ex))

    root = tk.Tk()
    root.title("Texas Hold'em Simulator")
    tk.Label(root, text="Player 1: ").grid(row=0, column=0)
    e1, e2 = tk.Entry(root, width=2), tk.Entry(root, width=2)
    e3, e4 = tk.Entry(root, width=2), tk.Entry(root, width=2)
    e1.grid(row=0, column=1); e2.grid(row=0, column=2)
    e3.grid(row=0, column=3); e4.grid(row=0, column=4)

    tk.Label(root, text="Player 2: ").grid(row=1, column=0)
    e5, e6 = tk.Entry(root, width=2), tk.Entry(root, width=2)
    e7, e8 = tk.Entry(root, width=2), tk.Entry(root, width=2)
    e5.grid(row=1, column=1); e6.grid(row=1, column=2)
    e7.grid(row=1, column=3); e8.grid(row=1, column=4)

    tk.Label(root, text="Community Cards (up to 5):").grid(row=2, column=0)
    community_entries = []
    for i in range(5):
        r_entry = tk.Entry(root, width=2)
        s_entry = tk.Entry(root, width=2)
        r_entry.grid(row=2, column=1 + 2 * i)
        s_entry.grid(row=2, column=2 + 2 * i)
        community_entries.append((r_entry, s_entry))

    tk.Button(root, text="Simulate", command=simulate).grid(row=3, column=0, columnspan=5)
    tk.Button(root, text="Tirer aléatoirement", command=random_draw).grid(row=3, column=5, columnspan=5)
    result_label = tk.Label(root, text="")
    result_label.grid(row=4, column=0, columnspan=10)
    root.mainloop()


def monte_carlo_hand_distribution(h1: List[Card], 
                                  h2: List[Card], 
                                  known_community: List[Card], 
                                  iterations: int = 5000):
    p1_wins = 0
    p2_wins = 0
    ties = 0
    p1_hand_counts = Counter()
    p2_hand_counts = Counter()

    for _ in range(iterations):
        deck = generate_deck(h1 + h2 + known_community)
        draw_count = 5 - len(known_community)
        drawn = random.sample(deck, draw_count)
        full_community = known_community + drawn

        best1 = best_hand(h1, full_community)
        best2 = best_hand(h2, full_community)

        p1_type, _ = best1.classify()
        p2_type, _ = best2.classify()
        p1_hand_counts[p1_type] += 1
        p2_hand_counts[p2_type] += 1

        if best1 > best2:
            p1_wins += 1
        elif best2 > best1:
            p2_wins += 1
        else:
            ties += 1

    return p1_wins, p2_wins, ties, p1_hand_counts, p2_hand_counts


def run_plot_enhanced():
    import matplotlib.pyplot as plt

    h1 = [Card("A", "c"), Card("K", "d")]
    h2 = [Card("Q", "h"), Card("J", "s")]
    known_community = []
    total_iterations = 5000
    step = 100

    # 1) Convergence
    win1_prog, win2_prog, ties_prog = monte_carlo_simulation_convergence(
        h1, h2, known_community, total_iterations=total_iterations, step=step
    )
    iterations = list(range(step, total_iterations + 1, step))

    plt.figure()
    plt.plot(iterations, win1_prog, label="Joueur 1")
    plt.plot(iterations, win2_prog, label="Joueur 2")
    plt.plot(iterations, ties_prog, label="Égalités")
    plt.xlabel("Nombre d'itérations")
    plt.ylabel("Probabilité estimée")
    plt.title("Convergence des probabilités avec Monte Carlo")
    plt.legend()
    plt.grid(True)
    plt.show()

    # 2) Statistiques finales
    p1_wins, p2_wins, ties, p1_hand_counts, p2_hand_counts = monte_carlo_hand_distribution(
        h1, h2, known_community, iterations=total_iterations
    )

    p1_win_rate = p1_wins / total_iterations
    p2_win_rate = p2_wins / total_iterations
    tie_rate = ties / total_iterations

    plt.figure()
    categories = ["P1 win", "P2 win", "Tie"]
    values = [p1_win_rate, p2_win_rate, tie_rate]
    plt.bar(categories, values)
    plt.title("Probabilités finales de victoire/égalité (après 5000 itérations)")
    plt.ylabel("Probabilité")
    plt.show()

    # 3) Distribution des types de mains
    hand_ranking = [
        "High Card", "One Pair", "Two Pair", "Three of a Kind",
        "Straight", "Flush", "Full House", "Four of a Kind", "Straight Flush"
    ]
    p1_counts_ordered = [p1_hand_counts[h] for h in hand_ranking]
    p2_counts_ordered = [p2_hand_counts[h] for h in hand_ranking]

    p1_total = sum(p1_counts_ordered)
    p2_total = sum(p2_counts_ordered)
    p1_percents = [count / p1_total for count in p1_counts_ordered]
    p2_percents = [count / p2_total for count in p2_counts_ordered]

    plt.figure()
    plt.bar(hand_ranking, p1_percents)
    plt.title("Distribution des types de mains du Joueur 1")
    plt.ylabel("Fréquence (en %)")
    plt.xticks(rotation=45, ha="right")
    plt.show()

    plt.figure()
    plt.bar(hand_ranking, p2_percents)
    plt.title("Distribution des types de mains du Joueur 2")
    plt.ylabel("Fréquence (en %)")
    plt.xticks(rotation=45, ha="right")
    plt.show()


def run_plot():
    import matplotlib.pyplot as plt

    h1 = [Card("A", "c"), Card("K", "d")]
    h2 = [Card("Q", "h"), Card("J", "s")]
    known_community = []
    total_iterations = 5000
    step = 100

    win1, win2, ties = monte_carlo_simulation_convergence(
        h1, h2, known_community, total_iterations=total_iterations, step=step
    )
    iterations = list(range(step, total_iterations + 1, step))

    plt.figure()
    plt.plot(iterations, win1, label="Joueur 1")
    plt.plot(iterations, win2, label="Joueur 2")
    plt.plot(iterations, ties, label="Égalités")
    plt.xlabel("Nombre d'itérations")
    plt.ylabel("Probabilité estimée")
    plt.title("Convergence des probabilités avec Monte Carlo")
    plt.legend()
    plt.grid(True)
    plt.show()

def run_policy():
    import matplotlib.pyplot as plt
    h1 = [Card("A", "c"), Card("K", "d")]
    h2 = [Card("Q", "h"), Card("J", "s")]
    known_community = []

    alpha = 0.5
    iterations = 5000

    results = policy_adapted_monte_carlo_simulation(h1, h2, known_community,
                                                    iterations=iterations,
                                                    alpha=alpha,
                                                    reward_increment=0.1)
    
    p1_rate = results[0] / iterations
    p2_rate = results[1] / iterations
    tie_rate = results[2] / iterations

    categories = ["P1 Win", "P2 Win", "Tie"]
    values = [p1_rate, p2_rate, tie_rate]

    plt.bar(categories, values)
    plt.title(f"Policy-Adapted Results (alpha={alpha}, iters={iterations})")
    plt.ylabel("Proportion")
    plt.show()


def run_mcts_demo():
    """
    Exécute un MCTS sur un exemple simple, puis affiche les résultats estimés
    en termes de pourcentage de victoires P1/P2/Égalité.
    """
    h1 = [Card("A", "c"), Card("K", "d")]
    h2 = [Card("Q", "h"), Card("J", "s")]
    known_community = []  # pas de cartes communes pour l'instant

    iterations = 2000
    p1, p2, tie = mcts_poker_simulation(h1, h2, known_community, iterations=iterations, c=1.4)
    print(f"MCTS simulation ({iterations} itérations) :")
    print(f"-> P1 win rate = {p1:.3f}, P2 win rate = {p2:.3f}, Tie rate = {tie:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monte Carlo & MCTS simulations pour le poker.")
    parser.add_argument('--mode', type=str, 
                        choices=['gui', 'plot', 'plot_enhanced', 'policy', 'mcts'], 
                        default='gui',
                        help=(
                          "Choisir: \n"
                          "'gui' => interface graphique\n"
                          "'plot' => simple convergence plot\n"
                          "'plot_enhanced' => multiples graphes\n"
                          "'policy' => démonstration policy-adapted\n"
                          "'mcts' => démonstration MCTS"
                        ))
    args = parser.parse_args()

    if args.mode == 'gui':
        run_gui()
    elif args.mode == 'plot':
        run_plot()
    elif args.mode == 'plot_enhanced':
        run_plot_enhanced()
    elif args.mode == 'policy':
        run_policy()
    else:
        # 'mcts' -> run MCTS demonstration
        run_mcts_demo()
