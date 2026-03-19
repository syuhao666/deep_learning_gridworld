from flask import Flask, render_template, request, jsonify
import random

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/evaluate', methods=['POST'])
def evaluate():
    data = request.json
    n = data.get('n', 5)
    
    # frontend sends start/end/obs as [y, x] where y=n-1 is top, y=0 is bottom
    # we work with (x, y) internally
    start_point = (data['start'][1], data['start'][0])
    end_point = (data['end'][1], data['end'][0])
    obstacles = set((obs[1], obs[0]) for obs in data.get('obstacles', []))
    
    # 0: up, 1: down, 2: left, 3: right
    dx = [0, 0, -1, 1]
    dy = [1, -1, 0, 0]
    action_symbols = ['↑', '↓', '←', '→']
    
    def get_next_state(x, y, a):
        nx, ny = x + dx[a], y + dy[a]
        if not (0 <= nx < n and 0 <= ny < n) or (nx, ny) in obstacles:
            return (x, y)
        return (nx, ny)

    def get_reward(s, ns):
        if ns == end_point and s != end_point:
            return 10.0
        return -1.0

    states = [(x,y) for x in range(n) for y in range(n)]
    
    # ==========================================
    # 1. Random Policy Evaluation
    # ==========================================
    random_policy = {}
    for x, y in states:
        if (x,y) == end_point or (x,y) in obstacles:
            continue
        random_policy[(x,y)] = random.randint(0, 3)
        
    V_rand = {s: 0.0 for s in states}
    gamma = 0.9
    theta = 1e-4

    iters = 0
    while iters < 1000:
        delta = 0
        for s in states:
            if s == end_point or s in obstacles:
                continue
            v = V_rand[s]
            a = random_policy[s]
            ns = get_next_state(s[0], s[1], a)
            r = get_reward(s, ns)
            
            new_v = r + gamma * V_rand[ns]
            V_rand[s] = new_v
            delta = max(delta, abs(v - new_v))
        if delta < theta:
            break
        iters += 1
        
    # ==========================================
    # 2. Value Iteration (Optimal Policy)
    # ==========================================
    V_opt = {s: 0.0 for s in states}
    iters_vi = 0
    while iters_vi < 1000:
        delta = 0
        for s in states:
            if s == end_point or s in obstacles:
                continue
            v = V_opt[s]
            max_v = float('-inf')
            for a in range(4):
                ns = get_next_state(s[0], s[1], a)
                r = get_reward(s, ns)
                val = r + gamma * V_opt[ns]
                if val > max_v:
                    max_v = val
            V_opt[s] = max_v
            delta = max(delta, abs(v - max_v))
        if delta < theta:
            break
        iters_vi += 1

    # Extract optimal policy
    opt_policy = {}
    for s in states:
        if s == end_point or s in obstacles:
            continue
        best_a = -1
        max_v = float('-inf')
        # to ensure it doesn't get stuck, resolve ties systematically
        for a in range(4):
            ns = get_next_state(s[0], s[1], a)
            r = get_reward(s, ns)
            val = r + gamma * V_opt[ns]
            if val > max_v:
                max_v = val
                best_a = a
        opt_policy[s] = best_a

    # Get Optimal Path (for highlighting)
    path = []
    curr = start_point
    visited = set()
    while curr != end_point and curr not in visited and curr not in obstacles:
        path.append([curr[1], curr[0]]) # store as [y, x] for frontend
        visited.add(curr)
        if curr not in opt_policy:
            break
        a = opt_policy[curr]
        ns = get_next_state(curr[0], curr[1], a)
        # If policy loops back or stuck, break
        if ns == curr:
            break
        curr = ns
    if curr == end_point:
        path.append([end_point[1], end_point[0]])

    # Format Output [y][x]
    def format_output(V_dict, pol_dict):
        res_pol = []
        res_v = []
        for y in range(n):
            pol_row = []
            v_row = []
            for x in range(n):
                s = (x, y)
                if s in obstacles:
                    pol_row.append(None)
                    v_row.append(None)
                elif s == end_point:
                    pol_row.append('⭐')
                    v_row.append(0.0)
                else:
                    pol_row.append(action_symbols[pol_dict[s]])
                    v_row.append(round(V_dict[s], 2))
            res_pol.append(pol_row)
            res_v.append(v_row)
        return res_v, res_pol

    rand_v, rand_p = format_output(V_rand, random_policy)
    opt_v, opt_p = format_output(V_opt, opt_policy)

    return jsonify({
        'random_policy': rand_p,
        'random_values': rand_v,
        'optimal_policy': opt_p,
        'optimal_values': opt_v,
        'optimal_path': path
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
