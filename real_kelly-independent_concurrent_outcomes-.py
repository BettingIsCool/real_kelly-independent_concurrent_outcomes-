import math
import time
import itertools
import numpy as np
import scipy.optimize
from datetime import datetime
from collections import defaultdict
from scipy.optimize import NonlinearConstraint


def optimize(selections: list, bankroll: float, max_multiple: int):

    start_time = time.time()

    # MAXIMUM TEAMS IN A MULTIPLE MUST NOT EXCEED LEN(SELECTIONS)
    if max_multiple > len(selections):
        print(f'Error: Maximum multiple must not exceed {len(selections)}')
        return None

    # CREATE A MATRIX OF POSSIBLE COMBINATIONS AND A PROBABILITY VECTOR OF SIZE LEN(COMBINATIONS)
    combinations, probs = list(), list()
    for c in range(0, len(selections) + 1):
        for subset in itertools.combinations(selections, c):
            combination, prob = list(), 1.00
            for selection in selections:
                if selection in subset:
                    combination.append(1)
                    prob *= 1 / selection['odds_fair']
                else:
                    combination.append(0)
                    prob *= (1 - 1 / selection['odds_fair'])
            combinations.append(combination)
            probs.append(prob)

    # CREATE A MATRIX OF POSSIBLE SINGLES & MULTIPLES
    bets, book_odds = list(), list()
    for multiple in range(1, max_multiple + 1):
        for subset in itertools.combinations(selections, multiple):
            bet, prod = list(), 1.00
            for selection in selections:
                if selection in subset:
                    bet.append(1)
                    prod *= selection['odds_book']
                else:
                    bet.append(0)
            bets.append(bet)
            book_odds.append(prod)

    # CACHE WINNING BETS
    winning_bets = defaultdict(list)
    for index_combination, combination in enumerate(combinations):
        for index_bet, bet in enumerate(bets):
            if sum([c * b for c, b in zip(combination, bet)]) == sum(bet):
                winning_bets[index_bet].append(index_combination)

    def f(stakes):
        """ This function will be called by scipy.optimize.minimize repeatedly to find the global maximum """

        # INITIALIZE END_BANKROLLS AND OBJECTIVE BEFORE EACH OPTIMIZATION STEP
        objective, end_bankrolls = 0.00, len(combinations) * [bankroll - np.sum(stakes)]

        for index_bet, index_combinations in winning_bets.items():
            for index_combination in index_combinations:
                end_bankrolls[index_combination] += stakes[index_bet] * book_odds[index_bet]

        # RETURN THE OBJECTIVE AS A SUMPRODUCT OF PROBABILITIES AND END_BANKROLLS - THIS IS THE FUNCTION TO BE MAXIMIZED
        return -sum([p * e for p, e in zip(probs, np.log(end_bankrolls))])

    def constraint(stakes):
        """ Sum of all stakes must not exceed bankroll """
        return sum(stakes)

    # FIND THE GLOBAL MAXIMUM USING SCIPY'S CONSTRAINED MINIMIZATION
    bounds = list(zip(len(bets) * [0], len(bets) * [bankroll]))
    nlc = NonlinearConstraint(constraint, -np.inf, bankroll)
    res = scipy.optimize.differential_evolution(func=f, bounds=bounds, constraints=(nlc))

    runtime = time.time() - start_time
    print(f"\n{datetime.now().replace(microsecond=0)} - Optimization finished. Runtime --- {round(runtime, 3)} seconds ---\n")
    print(f"Objective: {round(res.fun, 5)}")
    print(f"Certainty Equivalent: {round(math.exp(-res.fun), 3)}\n")

    # CONSOLE OUTPUT
    for index_bet, bet in enumerate(bets):
        bet_strings = list()
        for index_sel, sel in enumerate(bet):
            if sel == 1:
                bet_strings.append(selections[index_sel]['name'])

        stake = res.x[index_bet]
        if stake >= 0.50:
            print(f"{(' / ').join(bet_strings)} @{round(book_odds[index_bet], 3)} - € {int(round(stake, 0))}")

selections = list()
selections.append({'name': 'BET 1', 'odds_book': 2.05, 'odds_fair': 1.735})
selections.append({'name': 'BET 2', 'odds_book': 1.95, 'odds_fair': 1.656})
selections.append({'name': 'BET 3', 'odds_book': 1.75, 'odds_fair': 1.725})
selections.append({'name': 'BET 4', 'odds_book': 1.88, 'odds_fair': 1.757})
selections.append({'name': 'BET 5', 'odds_book': 1.99, 'odds_fair': 1.787})
selections.append({'name': 'BET 6', 'odds_book': 2.11, 'odds_fair': 1.794})

optimize(selections=selections, bankroll=2500, max_multiple=2)


