import random
from copy import deepcopy


class Optimizer:
    def __init__(self):
        self.results = []
        self.best_index = 0
        self.min_score = 0
        self.last_score = 0
        self.consequent_bad_score_count = 0

        self.consequent_score_count = 0
        self.last_min_score = 0
        self.last_min_score_count = 0
        self.last_min_score_idx = 0
        self.overall_min_score = 0
        self.overall_min_score_fail_count = 0

    def default(self, iteration_index, sorted_variants):
        self.results.append(sorted_variants)
        best_variant = sorted_variants[0]
        score = best_variant['avg'] + best_variant['delta']

        test_layout = deepcopy(best_variant["layout"])

        if iteration_index == 0:
            self.min_score = score

        if score < self.min_score:
            self.best_index = iteration_index
            self.min_score = score
            self.consequent_bad_score_count = 0
        else:
            self.consequent_bad_score_count += 1

        if self.consequent_bad_score_count >= 10:
            rand_int = random.randint(0, len(self.results) - 1)
            test_layout = deepcopy(random.choice(self.results[rand_int])["layout"])

        if self.consequent_bad_score_count >= 100:
            rand_int = random.randint(0, len(self.results) - 1)
            test_layout = deepcopy(self.results[rand_int][0]["layout"])

        if self.consequent_bad_score_count >= 300 and score <= self.min_score:
            return best_variant, True

        # Копируем последний результат
        self.last_score = score

        return test_layout, False

    def old(self, iteration_index, sorted_variants):
        self.results.append(sorted_variants)
        best_variant = sorted_variants[0]
        score = best_variant['avg'] + best_variant['delta']

        test_layout = deepcopy(best_variant["layout"])

        # Если результат равен предыдущему
        if score == self.last_score:
            self.consequent_score_count += 1

        # Если последний результат меньше минимального, то обновляем минимальный результат и сохроняем его индекс
        if score < self.min_score or self.min_score == 0:
            self.min_score = score
            self.last_min_score_idx = iteration_index

        # Если последний результат чем минимальный, то инкрементим
        if score > self.min_score:
            self.last_min_score_count += 1

        # Если последний минимальный результат был очень давно и мы движемся не туда
        if self.last_min_score_count > 10:
            test_layout = deepcopy(self.results[self.last_min_score_idx][-1]["layout"])
            self.last_min_score = 0

        # Если результат "застрял" и не менялся уже втечение 10 измерений, то кидаем предыдущий рандом, чтобы раскачать
        if self.consequent_score_count > 10:
            self.consequent_score_count = 0
            rand_int = random.randint(0, len(self.results) - 1)
            # test_layout = deepcopy(random.choice(results[rand_int])["layout"])
            test_layout = deepcopy(self.results[rand_int][-1]["layout"])

        # Если результат опустился ниже минимума за все время, то обновляем минимум
        if score < self.overall_min_score or self.overall_min_score == 0:
            self.overall_min_score = score
            self.overall_min_score_fail_count = 0

        # Если результат равен минимальному результату за все время, то инкрементим
        if score == self.overall_min_score:
            self.overall_min_score_fail_count += 1

        # Результат не опускался ниже самого минимального полученного значения за N раз
        if self.overall_min_score_fail_count > 100:
            return best_variant, True

        # Копируем последний результат
        self.last_score = score

        return test_layout, False
