import math
import random
import numpy as np

variance_results = np.empty(100)
ucb_results = np.empty(100)
optimal_results = np.empty(100)

for idx in range(0, 100):
    # Generate Data
    N = 10000  # the time (or round)
    d = 50  # number of possible choices
    user = list(range(0, N))
    m = np.zeros((d, N))
    max_mean = -10
    argmax_mean = 0
    for i in range(d):
        r = random.randrange(-50, 51, 1) / 50
        v = random.randrange(10, 51, 1) / 200
        if max_mean < r:
            max_mean = r
            argmax_mean = i
        m[i] = np.random.normal(100 + r, v, N)


    # Initialize Variables
    Qt_a = 0
    Nt_a = np.zeros(d)  # number of times action a has been selected prior to T
    # If Nt(a) = 0, then a is considered to be a maximizing action.
    N_var_t_a = np.zeros(d)  # number of times action a has been selected prior to T (for variance approach)
    # If Nt(a) = 0, then a is considered to be a maximizing action.
    c = 1  # a number greater than 0 that controls the degree of exploration
    sum_rewards = np.zeros(d)  # cumulative sum of reward for a particular message
    variance_rewards = np.zeros(d)  # store the respective variance of each message
    sum_var_rewards = np.zeros(d)  # cumulative sum of reward for a particular message (for variance approach)
    # helper variables to perform analysis
    hist_t = []  # holds the natural log of each round
    hist_achieved_rewards = []  # holds the history of the UCB CHOSEN cumulative rewards
    hist_best_possible_rewards = []  # holds the history of OPTIMAL cumulative rewards
    hist_random_choice_rewards = []  # holds the history of RANDONMLY selected actions rewards
    hist_variance_rewards = []  # holds the history of action rewards with c = sqrt(var)
    mean_squares = np.zeros(d)  # to be divided by the respective n, only holds the sum of squares
    ###
    for t in range(0, N):
        UCB_Values = np.zeros(d)  # array holding the ucb values. we pick the max
        action_selected = 0
        for a in range(0, d):
            if Nt_a[a] > 0:
                ln_t = math.log(t)  # natural log of t
                # calculate the UCB
                Qt_a = sum_rewards[a] / Nt_a[a]
                ucb_value = Qt_a + c * (ln_t / Nt_a[a])
                UCB_Values[a] = ucb_value
            # if this equals zero, choose as the maximum. Cant divide by negative
            elif Nt_a[a] == 0:
                UCB_Values[a] = 1e500  # make large value

        # select the max UCB value
        action_selected = np.argmax(UCB_Values)
        # update Values as of round t
        Nt_a[action_selected] += 1
        reward = m[action_selected, t]
        #reward = df.values[t, action_selected + 1]
        sum_rewards[action_selected] += reward

        var_values = np.zeros(d)
        action_var = 0
        for a in range(0, d):
            if N_var_t_a[a] > 0:
                ln_t = math.log(t)  # natural log of t
                Qtvar_a = sum_var_rewards[a] / N_var_t_a[a]
                var = mean_squares[a]/N_var_t_a[a] - (sum_var_rewards[a] / N_var_t_a[a]) * (sum_var_rewards[a] / N_var_t_a[a])
                #print(a)
                #print(t)
                #print(m[a,t])
                #print(var)
                params = np.array(0.25, var + math.sqrt(2 * ln_t / N_var_t_a[a]))
                param = np.min(params)
                var_value = Qtvar_a + c * (param * ln_t / N_var_t_a[a])
                var_values[a] = var_value
            elif N_var_t_a[a] == 0:
                var_values[a] = 1e500  # make large value

        action_var = np.argmax(var_values)
        N_var_t_a[action_var] += 1
        var_reward = m[action_var, t]
        sum_var_rewards[action_var] += var_reward
        mean_squares[action_var] += var_reward * var_reward

        r_best = m[argmax_mean, t]  # select the best action

        pick_random = random.randrange(d)  # choose an action randomly
        r_random = m[pick_random]  # np.random.choice(r_) #select reward for random action
        if len(hist_achieved_rewards) > 0:
            hist_achieved_rewards.append(hist_achieved_rewards[-1] + reward)
            hist_best_possible_rewards.append(hist_best_possible_rewards[-1] + r_best)
            hist_random_choice_rewards.append(hist_random_choice_rewards[-1] + r_random)
            hist_variance_rewards.append(hist_variance_rewards[-1] + var_reward)
        else:
            hist_achieved_rewards.append(reward)
            hist_best_possible_rewards.append(r_best)
            hist_random_choice_rewards.append(r_random)
            hist_variance_rewards.append(var_reward)
    variance_results[idx] = hist_variance_rewards[-1]
    ucb_results[idx] = hist_achieved_rewards[-1]
    optimal_results[idx] = hist_best_possible_rewards[-1]


print("variance results")
print(variance_results)
print("ucb-results")
print(ucb_results)
print("optimal results")
print(optimal_results)
print("variance - ucb")
print(variance_results-ucb_results)
print(np.mean(variance_results-ucb_results))
print(np.var(variance_results-ucb_results))
print("optimal - variance")
print(optimal_results-variance_results)
