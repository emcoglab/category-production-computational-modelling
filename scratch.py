import re
from os import path

from evaluation.model_specs import save_model_spec


def main():
    base="/Volumes/Data/spreading activation model/Model output/n-gram runs for CogSci 2019"
    dirs = [
        "Category production traces (30,000 words; firing 0.3; sd_frac 15; length 10; model [PMI n-gram (BBC), r=5])",
        "Category production traces (30,000 words; firing 0.3; sd_frac 15; length 10; model [PPMI n-gram (BBC), r=5])",
        "Category production traces (30,000 words; firing 0.3; sd_frac 20; length 10; model [PMI n-gram (BBC), r=5])",
        "Category production traces (30,000 words; firing 0.4; sd_frac 15; length 10; model [PMI n-gram (BBC), r=5])",
        "Category production traces (30,000 words; firing 0.4; sd_frac 15; length 10; model [PPMI n-gram (BBC), r=5])",
        "Category production traces (30,000 words; firing 0.5; sd_frac 20; length 10; model [PPMI n-gram (BBC), r=5])",
        "Category production traces (30,000 words; firing 0.6; sd_frac 4; length 100; model [log n-gram (BBC), r=5])",
        "Category production traces (30,000 words; firing 0.6; sd_frac 5; length 100; model [log n-gram (BBC), r=5])",
        "Category production traces (30,000 words; firing 0.7; sd_frac 4; length 100; model [log n-gram (BBC), r=5])",
        "Category production traces (30,000 words; firing 0.8; sd_frac 5; length 100; model [log n-gram (BBC), r=5])",
        "Category production traces (30,000 words; firing 0.9; sd_frac 6; length 100; model [log n-gram (BBC), r=5])",
        "Category production traces (40,000 words; firing 0.3; sd_frac 15; length 10; model [PMI n-gram (BBC), r=5])",
        "Category production traces (40,000 words; firing 0.3; sd_frac 15; length 10; model [PPMI n-gram (BBC), r=5])",
        "Category production traces (40,000 words; firing 0.3; sd_frac 20; length 10; model [PMI n-gram (BBC), r=5])",
        "Category production traces (40,000 words; firing 0.4; sd_frac 15; length 10; model [PMI n-gram (BBC), r=5])",
        "Category production traces (40,000 words; firing 0.4; sd_frac 15; length 10; model [PPMI n-gram (BBC), r=5])",
        "Category production traces (40,000 words; firing 0.5; sd_frac 20; length 10; model [PPMI n-gram (BBC), r=5])",
        "Category production traces (40,000 words; firing 0.6; sd_frac 4; length 100; model [log n-gram (BBC), r=5])",
        "Category production traces (40,000 words; firing 0.7; sd_frac 4; length 100; model [log n-gram (BBC), r=5])",
        "Category production traces (40,000 words; firing 0.8; sd_frac 5; length 100; model [log n-gram (BBC), r=5])",
        "Category production traces (40,000 words; firing 0.9; sd_frac 6; length 100; model [log n-gram (BBC), r=5])",
    ]

    pattern_re = re.compile(r"Category production traces \("
                            r"(?P<words>[0-9,]+) words; "
                            r"firing (?P<firing>[0-9\.]+); "
                            r"sd_frac (?P<sd_frac>[0-9/.]+); "
                            r"length (?P<length>[0-9]+); "
                            r"model \[(?P<model>[^\]]+)\]\)")

    for d in dirs:
        pattern_match = re.match(pattern_re, d)
        save_model_spec(
            edge_decay_sd_factor=float(pattern_match.group("sd_frac")),
            firing_threshold=float(pattern_match.group("firing")),
            length_factor=int(pattern_match.group("length")),
            model_name=pattern_match.group("model"),
            n_words=int(pattern_match.group("words").replace(",", "")),
            response_dir=path.join(base, d)
        )


if __name__ == '__main__':
    main()
