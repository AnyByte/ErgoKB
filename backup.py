    score = best_variant['avg'] + best_variant['delta']

    # Если результат равен предыдущему
    if score == last_score:
        consequent_score_count += 1
    # Копируем последний результат
    last_score = score

    # Если последний результат меньше минимального, то обновляем минимальный результат и сохроняем его индекс
    if score < min_score or min_score == 0:
        min_score = score
        last_min_score_idx = i

    # Если последний результат чем минимальный, то инкрементим
    if score > min_score:
        last_min_score_count += 1

    # Если последний минимальный результат был очень давно и мы движемся не туда
    if last_min_score_count > 10:
        test_layout = deepcopy(results[last_min_score_idx][-1]["layout"])
        last_min_score = 0

    # Если результат "застрял" и не менялся уже втечение 10 измерений, то кидаем предыдущий рандом, чтобы раскачать
    if consequent_score_count > 10:
        consequent_score_count = 0
        rand_int = random.randint(0, len(results) - 1)
        # test_layout = deepcopy(random.choice(results[rand_int])["layout"])
        test_layout = deepcopy(results[rand_int][-1]["layout"])

    # Если результат опустился ниже минимума за все время, то обновляем минимум
    if score < overall_min_score or overall_min_score == 0:
        overall_min_score = score
        overall_min_score_fail_count = 0

    # Если результат равен минимальному результату за все время, то инкрементим
    if score == overall_min_score:
        overall_min_score_fail_count += 1
    
    # Результат не опускался ниже самого минимального полученного значения за N раз
    if overall_min_score_fail_count > 100:
        break