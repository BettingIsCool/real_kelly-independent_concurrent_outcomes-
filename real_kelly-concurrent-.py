import time
import itertools
import scipy.optimize
from numpy import asarray
from numpy import log as ln
from datetime import datetime


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

    def f(stakes):
        """ This function will be called by scipy.optimize.minimize repeatedly to find the global maximum """

        # INITIALIZE END_BANKROLLS AND OBJECTIVE BEFORE EACH OPTIMIZATION STEP
        objective, end_bankrolls = 0.00, len(combinations) * [bankroll]

        # CALCULATE END_BANKROLL FOR EACH COMBINATION WITH A 'STAKES' VECTOR OF SIZE LEN(BETS) AS VARIABLE
        for index_combination, combination in enumerate(combinations):
            for index_bet, bet in enumerate(bets):
                if sum([c * b for c, b in zip(combination, bet)]) == sum(bet):
                    end_bankrolls[index_combination] += stakes[index_bet] * (book_odds[index_bet] - 1)
                else:
                    end_bankrolls[index_combination] -= stakes[index_bet]

        # RETURN THE OBJECTIVE AS A SUMPRODUCT OF PROBABILITIES AND END_BANKROLLS - THIS IS THE FUNCTION TO BE MAXIMIZED
        # SEE https://www.pinnacle.com/en/betting-articles/Betting-Strategy/the-real-kelly-criterion/HZKJTFCB3KNYN9CJ
        return -sum([p * e for p, e in zip(probs, ln(end_bankrolls))])

    # INITITAL STAKES ARE 0 AND STAKES MUST BE NON-NEGATIVE
    guess = asarray(len(bets) * [0])
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
            print(f"{(' + ').join(bet_strings)} @{round(book_odds[index_bet], 3)} - € {round(stake, 2)}")

selections = list()
selections.append({'name': 'VIOLETTE AC to beat Arcahaie', 'odds_book': 1.751, 'odds_fair': 1.661})
selections.append({'name': 'TEMPETE +0.5 to beat Cavaly', 'odds_book': 1.746, 'odds_fair': 1.452})
selections.append({'name': 'FICA to beat Real du Cap', 'odds_book': 3.6, 'odds_fair': 2.995})
selections.append({'name': 'BALTIMORE SC to beat Ouanaminthe', 'odds_book': 1.473, 'odds_fair': 1.401})
selections.append({'name': 'MIREBALAIS to beat Racing Club Haitien', 'odds_book': 2.04, 'odds_fair': 1.909})
selections.append({'name': 'AMERICA DES CAYES to beat Saint Rose', 'odds_book': 4.19, 'odds_fair': 3.580})
selections.append({'name': 'DON BOSCO FC +0.5 to beat Juventus des Cayes', 'odds_book': 1.704, 'odds_fair': 1.555})

optimize(selections=selections, bankroll=2500.00, max_multiple=5)
