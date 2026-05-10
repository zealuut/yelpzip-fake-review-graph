# LLM Mask Quality Report

Source review sample: `graph/outputs/yelpzip_balanced_current_graph_no_reweight_20260502_160620/prepared_data/reviews_canonical.csv`

This report uses the actual balanced D1/base sampled reviews and joins them to the global llm cache by `review_node_id`.

## Headline
- Reviews: `39667`
- Users: `6664` (`fake=3332`, `real=3332`)
- Overall mask-hit review rate: `0.6311`
- Overall avg abnormal pattern count: `1.0366`
- LLM error rate: `0.0000`

## Review-Level by Label
- `fake` reviews: `14834`; mask-hit `0.6328`; avg pattern count `1.0334`; llm error `0.0000`
- `real` reviews: `24833`; mask-hit `0.6300`; avg pattern count `1.0385`; llm error `0.0000`

## User-Level by Label
- `fake` users: `3332`; any-mask-hit `0.9700`; avg reviews/user `4.4520`
- `real` users: `3332`; any-mask-hit `0.9772`; avg reviews/user `7.4529`

## Top Pattern Types
- `exaggeration`: all `0.5348`, fake `0.5347`, real `0.5349`
- `lack_of_detail`: all `0.2044`, fake `0.2021`, real `0.2057`
- `inconsistent_claim`: all `0.0849`, fake `0.0851`, real `0.0848`
- `overly_absolute`: all `0.0742`, fake `0.0746`, real `0.0740`
- `template_like`: all `0.0620`, fake `0.0621`, real `0.0620`
- `generic_promotion`: all `0.0424`, fake `0.0416`, real `0.0428`
- `sentiment_rating_mismatch`: all `0.0336`, fake `0.0329`, real `0.0340`
- `generic_attack`: all `0.0003`, fake `0.0004`, real `0.0003`

## Interpretation
- llm_mask has non-trivial coverage on the sampled reviews and should be treated as auxiliary anomaly evidence rather than a standalone strong separator.
- Fake-side rates are slightly higher on some pattern types, but the separation is not large enough to replace the main classifier.
- Real users write more reviews on average in this balanced-user protocol, so user-balanced does not imply review-balanced.
