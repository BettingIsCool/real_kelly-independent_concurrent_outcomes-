import time
import random
import itertools
import numpy as np
import scipy.optimize
from datetime import datetime
from collections import defaultdict


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
        # SEE https://www.pinnacle.com/en/betting-articles/Betting-Strategy/the-real-kelly-criterion/HZKJTFCB3KNYN9CJ
        return -sum([p * e for p, e in zip(probs, np.log(end_bankrolls))])

    # INITITAL STAKES ARE 0 AND STAKES MUST BE NON-NEGATIVE
    guess = np.asarray(len(bets) * [0])
    bounds = list(zip(len(bets) * [0], len(bets) * [None]))
    res = scipy.optimize.minimize(fun=f, x0=guess, method='L-BFGS-B', bounds=bounds, options={'maxcor': 10000, 'gtol': 1e-10, 'eps': 1e-08, 'maxfun': 1000000, 'maxiter': 1000000})

    runtime = time.time() - start_time
    print(f"\n{datetime.now().replace(microsecond=0)} - Optimization finished. Runtime --- {round(runtime, 3)} seconds ---\n")

    # CONSOLE OUTPUT
    for index_bet, bet in enumerate(bets):
        bet_strings = list()
        for index_sel, sel in enumerate(bet):
            if sel == 1:
                bet_strings.append(selections[index_sel]['name'])

        stake = res.x[index_bet]
        if stake > 0.00:
            print(f"{(' / ').join(bet_strings)} @{round(book_odds[index_bet], 3)} - € {round(stake, 2)}")

selections = list()
selections.append({'name': 'AUSTRIA 2. LIGA - UNDER 25.5', 'odds_book': 1.85, 'odds_fair': 1.735})
selections.append({'name': 'SWITZERLAND SUPER LEAGUE - UNDER 15.5', 'odds_book': 1.85, 'odds_fair': 1.656})
selections.append({'name': 'NETHERLANDS EREDIVISIE - UNDER 29.5', 'odds_book': 1.85, 'odds_fair': 1.725})
selections.append({'name': 'AUSTRIA BUNDESLIGA - UNDER 19.5', 'odds_book': 1.85, 'odds_fair': 1.757})
selections.append({'name': 'FRANCE LIGUE 1 - UNDER 26.5', 'odds_book': 1.85, 'odds_fair': 1.787})
selections.append({'name': 'SPAIN PRIMERA DIVISIÓN - UNDER 26.5', 'odds_book': 1.85, 'odds_fair': 1.794})
selections.append({'name': 'ENGLAND PREMIER LEAGUE - UNDER 28.5', 'odds_book': 1.85, 'odds_fair': 1.763})

optimize(selections=selections, bankroll=2500, max_multiple=2)


