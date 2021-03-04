from framework.category_production.category_production import CategoryProduction

cp = CategoryProduction(minimum_production_frequency=1)
for c in cp.category_labels:
    c_words = [w for w in c.split(" ") if w not in cp.ignored_words]
    for r in cp.responses_for_category(c):
        r_words = [w for w in r.split(" ") if w not in cp.ignored_words]
        for r_word in r_words:
            if r_word in c_words:
                print(f"{c.upper()} -> {r.lower()}")
