def duration_logprob(length_f: int, mu: float, sigma: float) -> float:
    if length_f <= 0:
        return -1e9
    x = math.log(length_f)
    return -0.5 * ((x - mu) / sigma) ** 2 - math.log(length_f * sigma + 1e-8)


def segmental_viterbi(
    log_probs: np.ndarray,
    dur_priors: Dict[int, Dict[str, float]],
    log_trans: np.ndarray
) -> np.ndarray:
    t_max, n_states = log_probs.shape
    assert n_states == 2

    max_d = int(max(dur_priors[0]["p99"], dur_priors[1]["p99"]) + 5)
    max_d = max(5, min(max_d, t_max))

    csum = np.cumsum(log_probs, axis=0)
    dp = np.full((t_max, n_states), -1e18, dtype=np.float64)
    bp_prev_t = np.full((t_max, n_states), -1, dtype=np.int32)
    bp_prev_s = np.full((t_max, n_states), -1, dtype=np.int32)

    for t in range(t_max):
        dmax = min(max_d, t + 1)
        for s in range(n_states):
            mu = dur_priors[s]["mu"]
            sigma = dur_priors[s]["sigma"]
            best = -1e18
            best_prev_t, best_prev_s = -1, -1

            for d in range(1, dmax + 1):
                st = t - d + 1
                emit = csum[t, s] - (csum[st - 1, s] if st > 0 else 0.0)
                score = emit + duration_logprob(d, mu, sigma)

                if st == 0:
                    if score > best:
                        best = score
                        best_prev_t = -1
                        best_prev_s = -1
                else:
                    for ps in range(n_states):
                        if ps == s:
                            continue
                        cand = dp[st - 1, ps] + log_trans[ps, s] + score
                        if cand > best:
                            best = cand
                            best_prev_t = st - 1
                            best_prev_s = ps

            dp[t, s] = best
            bp_prev_t[t, s] = best_prev_t
            bp_prev_s[t, s] = best_prev_s

    labels = np.full((t_max,), -1, dtype=np.int64)
    s = int(np.argmax(dp[t_max - 1]))
    t = t_max - 1

    while t >= 0:
        pt = bp_prev_t[t, s]
        ps = bp_prev_s[t, s]
        st = 0 if pt < 0 else pt + 1
        labels[st:t + 1] = s
        t = pt
        s = ps if ps >= 0 else 0

    return labels


def collapse_segments(labels: np.ndarray, hop_sec: float) -> List[Tuple[str, float, float, float]]:
    out = []
    if len(labels) == 0:
        return out

    cur = int(labels[0])
    start = 0

    for i in range(1, len(labels)):
        if int(labels[i]) != cur:
            a = start * hop_sec
            b = i * hop_sec
            out.append((ID_TO_LABEL[cur], a, b, b - a))
            start = i
            cur = int(labels[i])

    a = start * hop_sec
    b = len(labels) * hop_sec
    out.append((ID_TO_LABEL[cur], a, b, b - a))
    return out

