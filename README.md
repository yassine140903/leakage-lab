# Leakage Lab — Report

**Question:** how much does a feature pipeline that ignores time lie about its own accuracy?

**Answer:** on this dataset, it reports 0.8429 ROC-AUC and delivers 0.7381 in production — an overstatement of **0.105 AUC**. The isolated cost of the feature bug alone is **0.094 AUC**. Neither number is visible from inside the broken pipeline: nothing crashes, nothing warns, and every value in the output table looks reasonable.

---

## 0. Plain-language summary

We have two years of receipts from a UK gift wholesaler. We want to predict, for a given customer on a given date, whether they will buy anything in the next 30 days.

To train a model you need two things for every row:

- **the answer** ("did they buy in the next 30 days?") — which comes from the *future* relative to that date
- **the clues** ("how much have they spent, how recently did they buy?") — which must come only from the *past* relative to that date

The bug this lab studies is mixing those up: computing the clues using data from after the date. It is astonishingly easy to do by accident. Someone takes the company's current customer table — which is up to date as of today — and attaches it to a list of historical dates. Every column name is right. Every value is a real number. The table has the right number of rows. And it is wrong, because "total spend" now means "total spend as of today," which includes purchases that happened after the date we're pretending to stand on.

We built the pipeline **both ways**, changed nothing else, and measured the difference.

### Glossary

| Term | Meaning |
|---|---|
| **cutoff** | A date we pretend to be standing on. We ask "what happens in the next 30 days?" |
| **label** | The answer. 1 if the customer bought something in the 30 days after the cutoff, 0 otherwise. |
| **feature** | A clue. A number describing the customer, e.g. days since last purchase. |
| **point-in-time (PIT)** | The correct method. Each row's features use only data from before that row's cutoff. |
| **naive** | The broken method. All features computed once as of the end of the data (Dec 2011), then attached to every row regardless of its cutoff. |
| **leakage** | Information from after the cutoff sneaking into the features. |
| **base rate** | Fraction of rows where the answer is 1. Here: 21.5%. What you'd get by guessing. |
| **ROC-AUC** | Probability the model ranks a random buyer above a random non-buyer. 0.5 = coin flip, 1.0 = perfect. |
| **PR-AUC** | Roughly: when the model says "yes," how often is it right. A useless model scores ≈ the base rate. |
| **training/serving skew** | The model was trained on features computed one way and is fed features computed another way in production. |

---

## 1. Setup

### 1.1 Data

UCI Online Retail II (dataset 502). A UK-based online gift wholesaler, mostly selling to small business customers.

| | |
|---|---|
| Raw rows (2 sheets concatenated) | 1,067,371 |
| Sheet 1 span | 2009-12-01 07:45 → 2010-12-09 20:01 (525,461 rows) |
| Sheet 2 span | 2010-12-01 08:26 → 2011-12-09 12:50 (541,910 rows) |
| Sheet overlap window | 2010-12-01 08:26 → 2010-12-09 20:01 |
| Exact duplicate rows dropped | 34,335 |
| — inside the overlap window | 22,844 |
| — outside the overlap window | **11,491** |
| Rows after dedupe | 1,033,036 |
| Rows with no customer ID | 235,151 (22.8%) |
| Cancellation rows (invoice starts with `C`) | 19,104 (1.8%) |
| Rows with quantity ≤ 0 | 22,496 |
| Rows with price ≤ 0 | 6,019 |
| **Rows surviving all filters** | **779,425** |

Two observations from these counts that shaped later decisions:

**The two filters are not redundant.** There are 22,496 rows with non-positive quantity but only 19,104 cancellation rows. So roughly 3,400 negative-quantity rows are *not* cancellation invoices — they are manual adjustments, damages, stock write-offs. Filtering only on the `C` prefix (as the lab brief suggests) would have let those through and injected negative spend into customers' lifetime totals.

**The 11,491 out-of-overlap duplicates are a judgment call, not cleanup.** Rows identical on all eight business columns but outside the nine days where the two sheets overlap. They are either data-entry duplicates (drop them) or genuinely repeated line items on one invoice (keep them). Since `InvoiceDate` is recorded per-invoice rather than per-line, two lines on the same invoice always share a timestamp — the two cases are indistinguishable by construction. We dropped them, matching the brief's default. See §6.

### 1.2 Grain and population

**`orders`** — aggregated to one row per **(customer, timestamp)**: 36,851 rows, 5,878 customers, £17,374,804 total.

Aggregating to this grain up front is not cosmetic. Later we join "the most recent record before the cutoff." If a customer has two invoices at the identical timestamp, "most recent" is ambiguous and different implementations resolve it differently. Removing the tie at source means the ambiguity cannot exist.

**`cutoffs`** — 21 monthly timestamps, 2010-03-01 to 2011-11-01.

The end date is load-bearing. Data ends 2011-12-09. A cutoff of 2011-11-01 needs label data through 2011-12-01, which exists. A cutoff of 2011-12-01 would need data through 2011-12-31, which does not — so every customer would appear not to have bought, and we'd be training on silently fabricated negatives.

**`entities`** — every customer with at least one purchase before the cutoff. **84,536 rows.**

**A row is a customer at a moment, not a customer.** 5,878 customers produce 84,536 rows. The same customer appears up to 21 times with potentially different answers each time. This is the idea the entire lab rests on.

### 1.3 Base rate

**Overall: 0.2154.**

| cutoff | rows | positives | base rate |
|---|---|---|---|
| 2010-03-01 | 1,712 | 597 | 0.3487 |
| 2010-04-01 | 2,155 | 648 | 0.3007 |
| 2010-05-01 | 2,449 | 712 | 0.2907 |
| 2010-06-01 | 2,703 | 771 | 0.2852 |
| 2010-07-01 | 2,973 | 742 | 0.2496 |
| 2010-08-01 | 3,159 | 730 | 0.2311 |
| 2010-09-01 | 3,321 | 902 | 0.2716 |
| 2010-10-01 | 3,564 | 1,084 | 0.3042 |
| 2010-11-01 | 3,941 | 1,282 | 0.3253 |
| 2010-12-01 | 4,266 | 809 | 0.1896 |
| 2011-01-01 | 4,342 | 641 | 0.1476 |
| 2011-02-01 | 4,413 | 682 | 0.1545 |
| 2011-03-01 | 4,537 | 763 | 0.1682 |
| 2011-04-01 | 4,716 | 750 | 0.1590 |
| 2011-05-01 | 4,822 | 918 | 0.1904 |
| 2011-06-01 | 4,933 | 883 | 0.1790 |
| 2011-07-01 | 5,041 | 827 | 0.1641 |
| 2011-08-01 | 5,143 | 811 | 0.1577 |
| 2011-09-01 | 5,249 | 1,077 | 0.2052 |
| 2011-10-01 | 5,438 | 1,110 | 0.2041 |
| 2011-11-01 | 5,659 | 1,473 | 0.2603 |

Three things to read here.

**The seasonality is real.** September → November rises in both years (0.27 → 0.30 → 0.33, then 0.21 → 0.20 → 0.26). This is a gift wholesaler; that is Christmas stock-up. January is the floor in both years. The label behaving like the underlying business is decent evidence the 30-day window arithmetic is correct.

**The last cutoff is not truncated.** 2011-11-01 at 0.2603 is the highest of the late-2011 cutoffs and consistent with its Nov-2010 counterpart. If the label window ran off the end of the data we would see it collapse toward zero. It does not.

**The base rate falls by roughly a third between the train era and the test era** (2010 cutoffs ≈ 0.23–0.35, 2011 cutoffs ≈ 0.15–0.26). The mechanism is mechanical: the population accumulates customers who bought once and never returned, so the denominator grows faster than the numerator. Consequence: **PR-AUC is not comparable across splits with different base rates.** Every PR-AUC below is therefore reported alongside its own split's base rate and as a lift ratio.

### 1.4 Split

| | |
|---|---|
| Train | cutoffs ≤ 2011-06-01 |
| Test | cutoffs ≥ 2011-07-01 |

The one-month gap is not arbitrary. If you retrained on 2011-07-01, the newest cutoff whose label is fully observed is 2011-06-01, because a 2011-07-01 cutoff's label needs data through 2011-08-01. The split encodes the constraint a real retraining schedule would face.

---

## 2. The one-variable design

Every feature is computed by a single SQL template. Exactly one thing changes between the two pipelines:

```sql
FROM (SELECT *, {as_of} AS as_of FROM labeled) e
JOIN orders o
  ON o.customer_id = e.customer_id
 AND o.invoice_ts  < e.as_of
```

- **PIT:** `{as_of}` = `cutoff` — a column, so each row gets its own value
- **naive:** `{as_of}` = `TIMESTAMP '2011-12-10'` — a constant, so every row gets the same one

`as_of` does two jobs at once: it is the **wall** (which orders are visible to the join) and the **measuring stick** (the reference point for `recency_days`, `tenure_days`, and the 90-day window).

Six features, identical definitions in both pipelines: `lifetime_spend`, `lifetime_orders`, `recency_days`, `tenure_days`, `spend_90d`, `orders_90d`.

**The labels are built once, in a separate step, and shared by both pipelines.** This is what makes the comparison airtight: it is not merely that we tried to keep everything else equal, it is that the labels are provably the same object. Any difference in results can only come from the features.

### 2.1 What the leak looks like in the numbers

Both pipelines: **84,536 rows, 18,212 positives, base rate 0.2154.** Identical, as required.

| feature | | mean | std | min | 25% | 50% | 75% | max |
|---|---|---|---|---|---|---|---|---|
| lifetime_spend | PIT | 2,123.62 | 9,814.79 | 1.30 | 298.90 | 664.20 | 1,679.89 | 554,170.42 |
| | naive | 3,749.34 | 16,978.67 | 2.95 | 417.30 | 1,139.61 | 3,032.26 | 580,987.04 |
| lifetime_orders | PIT | 4.62 | 9.09 | 1 | 1 | 2 | 5 | 357 |
| | naive | 7.84 | 15.32 | 1 | 2 | 4 | 9 | 400 |
| recency_days | PIT | 136.01 | 137.58 | **1** | 30 | 86 | 203 | 700 |
| | naive | 232.30 | 224.42 | 1 | 30 | 143 | 414 | 739 |
| tenure_days | PIT | 288.14 | 180.86 | **1** | 136 | 269 | 425 | 700 |
| | naive | 581.40 | 149.64 | **40** | 486 | 626 | 701 | 739 |
| spend_90d | PIT | 493.20 | 2,321.60 | 0 | 0 | 75.90 | 437.69 | 114,475.32 |
| | naive | 609.76 | 3,753.56 | 0 | 0 | **0** | 469.92 | 168,469.60 |
| orders_90d | PIT | 1.07 | 2.16 | 0 | 0 | 1 | 1 | 80 |
| | naive | 1.12 | 2.87 | 0 | 0 | **0** | 1 | 89 |

**`recency_days` min = 1 in PIT.** Not zero, not negative. The strict `<` inequality is doing its job — no feature can see the cutoff instant or beyond.

**`tenure_days` is the clearest signature of the leak.** PIT min = 1 (a customer whose first purchase was the day before their cutoff — entirely normal). Naive min = **40**, because the measuring point is fixed at 2011-12-10 and the latest anyone can first appear is 40 days before it. The naive distribution is compressed into 40–739 with a median of 626: *everyone looks like a long-tenured veteran*, because we are looking back from the future.

**`spend_90d` is effectively dead in the naive pipeline.** Median and 75th percentile are both 0. The 90-day window is Sept–Dec 2011 for *every single row* regardless of cutoff, and most customers were inactive by then. A feature meant to say "recent activity" says nothing.

**Lifetime spend is 1.77× larger under naive.** It is summing orders that had not happened yet.

And the mechanism that makes this leakage rather than merely wrong: if a customer bought during their 30-day label window, that purchase *increases* their naive `lifetime_spend` and *decreases* their naive `recency_days`. **The answer is an arithmetic input to the clue.**

---

## 3. Three independent implementations

The deliverable of this lab is not the model. It is that the point-in-time logic is implemented three separate times, in three different styles, and all three agree.

| # | Implementation | Method |
|---|---|---|
| 1 | `FEATURE_SQL` | GROUP BY with an inequality filter on the join |
| 2 | `ASOF_SQL` | Window function producing timestamped snapshots + `ASOF LEFT JOIN` |
| 3 | `get_online_features` | Plain pandas, no SQL at all |

### 3.1 Why an ASOF join

A normal join needs exact equality. There is no exact match for time: your cutoff is 2010-05-01 and the customer's purchases are on 2010-04-10 and 2010-04-25. Neither equals the cutoff; you want the latest one *before* it.

```sql
ASOF LEFT JOIN snapshots s
  ON e.customer_id = s.customer_id
 AND e.cutoff > s.feature_ts
```

Equality on customer, inequality on time. Exactly one row is returned per left row: the snapshot with the greatest `feature_ts` still satisfying the condition.

This mirrors how real feature stores work. They do not re-aggregate raw events for every training row — too slow. They precompute **timestamped snapshots** of feature values (a running total after each purchase) and point-in-time join them onto training rows.

`ASOF LEFT JOIN`, not `ASOF JOIN` — with an inner join, unmatched rows would silently vanish and the row count would quietly drop. With LEFT, they appear as nulls and get counted. (Observed: **0 nulls.**)

### 3.2 Parity result

**84,536 rows both sides, 0 nulls, merged 84,536 as expected.**

| feature | max absolute difference | mismatches |
|---|---|---|
| lifetime_spend | 0.000000 | 0 |
| lifetime_orders | 0.000000 | 0 |
| recency_days | 0.000000 | 0 |
| tenure_days | 0.000000 | 0 |

Exact agreement across all 84,536 rows, including bit-exact float equality on `lifetime_spend`.

### 3.3 The feature that cannot be snapshotted

The ASOF implementation deliberately covers only **four** of the six features. `spend_90d` and `orders_90d` are absent, and this is a structural limitation, not an oversight.

A snapshot is written when an event happens. "Spend in the last 90 days, measured at the moment of the customer's last purchase" is not the same quantity as "spend in the last 90 days, measured at the cutoff." If the last purchase was 200 days before the cutoff, the snapshot reports a positive 90-day spend while the true value at the cutoff is zero.

**Cumulative features (lifetime totals) can be snapshotted on event. Windowed features cannot** — they change continuously with the passage of time, not only when something happens. Serving a stale windowed feature is a silent, permanent error.

This is the freshness problem, and it is a large part of why feature stores are hard. In this lab it is encoded as an executable test rather than a paragraph: `test_stale_window_is_zero`.

### 3.4 The bug parity testing actually found

The pandas online path was written deliberately without SQL. Comparing it against the SQL path on five sampled rows:

```
17363 2011-05-01   sql recency_days 173.0   pandas 172.0
16007 2010-12-01   sql              10.0    pandas   9.0
17730 2010-05-01   sql               2.0    pandas   1.0
15563 2011-06-01   sql               9.0    pandas   8.0
15287 2011-03-01   sql              96.0    pandas  95.0
```

**Off by exactly one, on every row, in the same direction.** Spend features agreed perfectly.

The cause: **the two implementations do different arithmetic.**

- DuckDB's `date_diff('day', a, b)` counts *calendar day boundaries crossed* — it truncates both timestamps to dates and subtracts.
- pandas' `(a - b).days` computes an elapsed timedelta and truncates toward zero.

Cutoffs are at midnight; purchases happen during business hours. So a fractional remainder is *always* present and *always* truncates downward. Deterministic, systematic, −1 everywhere.

Concretely, for customer 17730: last purchase 2010-04-29 at ~14:30, cutoff 2010-05-01 00:00.
- DuckDB: 2010-05-01 − 2010-04-29 = **2** days
- pandas: 1 day 9.5 hours → `.days` = **1**

**Why this matters more than the size of the error suggests.** Without a parity check: the model trains on offline features where recency is 173. Production serves 172 for the identical customer at the identical instant. Nothing crashes. No warning. The model is quietly worse than its evaluation claimed, forever. This is training/serving skew in its purest form.

Neither convention is *wrong* — they are different definitions of "days ago." What is wrong is having both in one system. We matched DuckDB, since the offline tables were already built with those semantics and the serving path is the one that must conform.

**If `get_online_features` had been written in SQL, both paths would share DuckDB's date arithmetic, both would say 173, the test would pass, and the bug would have shipped.** Independence is what makes agreement evidence.

---

## 4. Tests

Nine tests, all passing.

| Test | What it protects |
|---|---|
| `test_no_future_data` | `recency_days > 0` and `tenure_days > 0` — no feature sees the cutoff instant or beyond |
| `test_tenure_at_least_recency` | First purchase is never after the last one |
| `test_windowed_features_bounded` | 90-day spend cannot exceed lifetime spend |
| `test_stale_window_is_zero` | If last purchase predates the window, 90-day features must be exactly 0 |
| `test_two_implementations_agree` | GROUP BY path vs window+ASOF path |
| `test_online_offline_parity` | SQL offline path vs pandas online path (200 sampled rows) |
| `test_naive_is_actually_leaky` | Naive features are constant per customer across cutoffs |
| `test_naive_differs_from_pit` | The two pipelines genuinely differ |
| `test_labels_identical` | Labels match between pipelines — the invariant the experiment rests on |

Four of these come from the lab brief; five were added.

**`test_no_future_data` uses strict `> 0`, not `>= 0`.** The brief specifies `>= 0`, but that version passes even if the join inequality is `<=` instead of `<` — a same-timestamp purchase would yield recency 0 and slip through. Since the brief's own mutation check asks you to flip that inequality, the brief's test is too weak for the brief's own exercise.

**Float equality is a flaky test, not a strict one.** `test_naive_is_actually_leaky` initially used `nunique() == 1` and failed on `lifetime_spend` and `spend_90d` — the only two double-precision columns; the four integer columns passed. Investigation:

| | max spread within customer |
|---|---|
| lifetime_spend | 5.82e-11 |
| spend_90d | 1.46e-11 |

On values in the thousands, that is a relative error of ~1e-14 — floating-point non-associativity from DuckDB's multithreaded aggregation summing the same numbers in different orders. 75% of customers showed a spread of exactly 0. Rewritten as `(max - min) < 1e-6`, which sits five orders of magnitude above the observed noise and five below anything meaningful.

Worth noting the contrast: §3.2's implementation-parity check returned *exactly* 0.0. Exact float equality passing once is luck, not a guarantee.

---

## 5. Mutation check

Green tests prove nothing on their own — a suite of `assert True` is also green. Two deliberate bugs were introduced, one at a time, to find out whether the suite bites.

### Mutation 1: `<` → `<=` in the feature join

**Result: not caught. All 9 tests passed.**

**This is the correct outcome, and it is not a hole in the suite.** The mutation never fired. For `<` and `<=` to select different rows, a purchase would have to land at exactly midnight on the first of a month. This dataset contains no such timestamps — purchases occur during business hours. The two queries select identical rows, so identical tests pass. A test that failed here would have to assert something about data that does not exist.

**What it does reveal:** PIT correctness currently rests on a property of *this dataset* rather than on the code. Feed in a system with midnight batch timestamps and `<` versus `<=` becomes a real, silent leak. The mitigation is that all three implementations use strict inequality consistently, so `test_two_implementations_agree` would catch them diverging even though neither is independently provable on this data.

### Mutation 2: `as_of` shifted by one day (`cutoff` → `cutoff + INTERVAL 1 DAY`)

**Result: caught, by two tests.**

```
FAILED test_two_implementations_agree - AssertionError: lifetime_spend
FAILED test_online_offline_parity     - (12429, 2010-10-01, 'recency_days', 7, 8)
```

The parity failure message gives customer, cutoff, feature, online value, offline value — enough to go straight to the offending row.

**The critical detail is which tests did *not* fire.** `test_no_future_data` passed. `test_stale_window_is_zero` passed. `test_tenure_at_least_recency` passed. `test_windowed_features_bounded` passed. Every single-pipeline invariant check passed.

**Only the cross-implementation tests caught it.**

This is the central lesson of the section: **internal consistency checks cannot detect a systematically shifted `as_of`, because a shifted pipeline is still perfectly self-consistent.** Every number is plausible, every invariant holds, and the entire table is wrong. The only thing that catches it is a second implementation that did not make the same mistake.

That is precisely what a feature store sells you, and precisely why the online path had to be written in pandas.

---

## 6. Results

Model held fixed throughout: LightGBM, 300 trees, learning rate 0.05, `random_state=0`. **Only the data varies.** No tuning was performed on any experiment.

### 6.1 Primary population (all customers with ≥1 prior purchase)

| ID | Features | Split | Evaluated on | ROC-AUC | PR-AUC | base rate | PR lift |
|---|---|---|---|---|---|---|---|
| **A** | naive | random | naive | **0.8429** | 0.6269 | 0.2149 | 2.92 |
| **A′** | naive | random | **PIT** | 0.7514 | 0.5114 | 0.2149 | 2.38 |
| **B** | PIT | random | PIT | 0.7777 | 0.5538 | 0.2149 | 2.58 |
| **C** | PIT | temporal | PIT | **0.7915** | 0.5508 | 0.1997 | 2.76 |
| **D** | naive | temporal | **PIT** | **0.7381** | 0.4794 | 0.1997 | 2.40 |
| **D′** | naive | temporal | naive | 0.7977 | 0.5452 | 0.1997 | 2.73 |
| **E** | recency only | temporal | PIT | 0.7476 | 0.3756 | 0.1997 | 1.88 |

- **A** — the number a careless notebook reports
- **A′** — *(added)* the same model as A, scored on PIT test rows. Isolates the feature change with the test set held fixed
- **B** — correct features, leaky split
- **C** — the honest number
- **D** — what the leaky model actually does in production, where only PIT features exist
- **D′** — *(added)* the leaky model's self-reported number under a temporal split
- **E** — no model at all: rank by `-recency_days`

### 6.2 Feature importances (split counts)

| feature | A (naive) | C (PIT) |
|---|---|---|
| lifetime_spend | 1,673 | 1,917 |
| lifetime_orders | 1,457 | 956 |
| recency_days | 2,108 | 2,240 |
| tenure_days | 2,312 | 2,252 |
| spend_90d | 1,052 | 1,236 |
| orders_90d | 398 | 399 |

The prediction that the naive model would lean disproportionately on `recency_days` **does not hold** — the two profiles are nearly identical. The only material difference is `lifetime_orders` (1,457 vs 956).

*Caveat:* these are split counts, measuring how *often* a feature was used, not how much it *helped*. Re-running with `importance_type="gain"` is required before drawing firm conclusions. This is unfinished.

### 6.3 Sensitivity: active-in-365-days population

| ID | Features | Split | Evaluated on | ROC-AUC | PR-AUC | base rate | PR lift |
|---|---|---|---|---|---|---|---|
| A | naive | random | naive | 0.8323 | 0.6274 | 0.2312 | 2.71 |
| A′ | naive | random | PIT | 0.7384 | 0.5090 | 0.2312 | 2.20 |
| B | PIT | random | PIT | 0.7639 | 0.5530 | 0.2312 | 2.39 |
| C | PIT | temporal | PIT | 0.7638 | 0.5627 | 0.2352 | 2.39 |
| D | naive | temporal | PIT | 0.7053 | 0.4941 | 0.2352 | 2.10 |
| D′ | naive | temporal | naive | 0.7701 | 0.5581 | 0.2352 | 2.37 |
| E | recency only | temporal | PIT | 0.7064 | 0.3813 | 0.2352 | 1.62 |

Rows: 77,706. Base rate: 0.2312. All 9 tests still pass on this population.

**This run failed to test what it was designed to test, and the reason is structural.** The data begins 2009-12-01. At the 2010-06-01 cutoff, "ever purchased" spans six months — entirely inside a 365-day window — so the restriction can remove nobody. The per-cutoff row counts for every 2010 cutoff are *identical* to the primary run. The restriction only bites from 2011-01 onward, and cuts just 8% of rows overall (84,536 → 77,706).

A 180-day window would be a real intervention. This one was close to a no-op, and its results should be read as a weak sensitivity check rather than a test of the population hypothesis.

---

## 7. Leakage accounting

### 7.1 The clean measurement: A → A′ = **0.0915 ROC-AUC**

Same model, same training rows, same split, same test rows. Only the features used at scoring time change. **This is the isolated cost of the feature bug**, and it is the number to quote.

This row was added because the decomposition specified in the lab brief does not isolate anything: A is evaluated on naive test rows and B on PIT test rows, so the A→B arrow changes *both* the features and the test set simultaneously. Reporting that delta as "feature leakage" would have been wrong.

### 7.2 Split leakage: B → C = **−0.0138**. Not detected.

C (temporal split) *beat* B (random split) by 0.014 on the primary population, and matched it exactly (0.7638 vs 0.7639) on the 365-day population.

The expected result was B > C: a random split lets the model see future cutoffs and train on the same customers it is tested on. That advantage does not appear here.

The likely explanation is the base-rate difference. C's test set is late-2011 only (base rate 0.1997) while B's is mixed across both eras (0.2149). In the 365-day run, where the base rates are much closer (0.2352 vs 0.2312), the anomaly vanishes and B = C exactly — consistent with the reversal being a base-rate artifact rather than evidence about split leakage.

**Honest conclusion: on this dataset, with this population and this feature set, random-split leakage is not measurable.** The A→B→C decomposition in the lab brief cannot be reported as specified, because one of its two arrows is negative.

### 7.3 The production gap: A − D = **0.1048 ROC-AUC**

The careless notebook reports 0.8429. The same pipeline, deployed, delivers 0.7381. Everything between those two numbers is fiction.

### 7.4 The skew cost specifically: D′ − D = **0.0596**

D′ is the naive model scored on naive test rows under a temporal split: 0.7977. D is the identical model scored on PIT test rows: 0.7381. The difference is attributable purely to the features changing between training and serving.

Note that **D′ (0.7977) slightly exceeds C (0.7915)**. A team using a temporal split — doing the thing everyone tells you to do — but with a leaky feature table would report a number marginally *better* than the honest model, and ship something 0.06 AUC worse. A correct split does not protect you from incorrect features.

### 7.5 Why D did not collapse — the most instructive result

Both predictions for D were badly wrong (predicted ~0.50–0.65, actual **0.7381**). The reasoning behind the prediction was that naive `recency_days` (measured from Dec 2011, spanning 1–739) would place LightGBM's learned thresholds outside the range of PIT `recency_days` (1–700), so every test row would fall on the same side of every split and the model would output near-constant scores.

That was wrong, and the mechanism is worth stating precisely:

**`recency_days` is monotone between the two worlds.** Naive recency and PIT recency are different numbers, but they *order customers almost identically* — a customer dormant since 2010 has high recency under both measurements. LightGBM's thresholds sit at wrong absolute values, but the **ranking survives**, and ROC-AUC measures only ranking.

**Therefore: AUC is the wrong metric for detecting training/serving skew.** A monotone distortion of a feature barely moves AUC while completely destroying calibration. If you were selecting customers above a fixed probability threshold for a marketing campaign, model D would be badly broken in a way no AUC in this report can see.

That is a more useful finding than the collapse we expected, and it argues for adding calibration metrics (Brier score, calibration curves) to any future version of this lab.

---

## 8. Honest verdict: does the model earn its place?

**C vs E: 0.7915 vs 0.7476 ROC-AUC. A gain of 0.0439 from 300 trees and five extra features over sorting customers by a single column.**

That is a thin margin, and on ROC-AUC alone the honest reading is that the model adds little.

**But PR-AUC disagrees sharply: 0.5508 vs 0.3756 — a lift of 2.76 vs 1.88 over base rate.**

The two metrics give opposite verdicts, and the disagreement is the finding. ROC-AUC averages ranking quality over the whole population; PR-AUC weights the top of the ranking, which is where you actually take action. The model's advantage over recency-alone is concentrated exactly where it matters and invisible in the summary statistic most people quote.

The 365-day run reinforces this: with some easy negatives removed, E degrades faster (0.7476 → 0.7064) than C (0.7915 → 0.7638), widening the ROC gap to 0.0574 and the PR lift gap to 2.39 vs 1.62. **The extra features do carry signal; a population saturated with obvious dead customers hides it.**

---

## 9. Predictions versus outcomes

Predictions were recorded before any experiment was run.

| | Expected | **Actual** | |
|---|---|---|---|
| A | > 0.90 | **0.8429** | too high |
| C | 0.60–0.70 | **0.7915** | too low |
| D | 0.50–0.60 | **0.7381** | far too low |
| E | ≈ C | **0.7476** (C − 0.044) | close |

*(No prediction was recorded for B.)*

**Three misses, all in the same direction: leakage is less destructive than expected, and the honest model is better than expected.**

The D miss is the largest and has a clear mechanism (§7.5). The C miss suggests the signal carried by recency-type features for repeat purchase was underrated.

The D prediction was revised mid-lab *with a stated reason*: initially ~0.50, on the assumption that the naive and PIT feature ranges would not overlap; after inspecting the distributions and finding substantial overlap, it was revised upward to 0.50–0.60. Still too low, but for a better reason.

A hypothesis was also raised and falsified. The proposed explanation for A topping out at only 0.84 was that the population is saturated with long-dead customers whom PIT recency already separates easily, leaving the leak little headroom. If true, restricting the population should shrink the A−A′ gap. It did not:

| | primary | 365-day |
|---|---|---|
| A − A′ | 0.0915 | **0.0939** |

The gap *widened*. **The explanation for A's ceiling remains unknown.** See §11.

---

## 10. Decisions and tradeoffs

**Population: all customers with ≥1 prior purchase.** Includes customers dormant since 2010, who are easy negatives and inflate AUC. Kept as primary deliberately: the all-time population is the *harder and more realistic* case, since production feature tables genuinely do accumulate dead customers, and a leaky pipeline exploits exactly that. Restricting the population would shrink the measured leak and weaken the demonstration. The 365-day variant is reported as sensitivity (§6.3), with the caveat that it barely bit.

**Cutoffs start 2010-03-01, three months after the data begins.** At the first cutoff, `lifetime_spend` can cover at most three months; at the last, twenty-three. The column means structurally different things across the time range, and since the temporal split trains on early cutoffs and tests on late ones, this is a distribution shift baked in by construction — unrelated to leakage. The alternative (starting cutoffs at 2010-12-01 so every cutoff has ≥12 months of history) would have cut 21 cutoffs to 12, roughly halving the training data. A noisier C was judged worse than a slightly pessimistic one. **Some portion of C's gap below A is this confound, not leakage.**

**Cancellations dropped (19,104 rows, 1.8%).** This discards real behavioural signal — "bought £500 and kept it" and "bought £500 and returned £450" become identical in every feature, and returns are plausibly informative about future purchasing. Using them properly requires separate features (`n_returns`, `return_rate`, `returned_amount_90d`) rather than allowing `amount` to go negative, since a return in March cancels a purchase from February and would make `spend_90d` negative in ways that do not mean "low activity." Deferred on time grounds.

Critically: **this costs model quality, not experimental validity.** The same rows are removed from both pipelines, so the A−A′ and A−D gaps are unaffected.

**11,491 within-sheet duplicate rows dropped** (§1.1). Ambiguous between data-entry errors and genuine repeated line items; indistinguishable by construction. Dropped to match the brief's default. Affects both pipelines identically, so again a footnote rather than a threat.

**22.8% of raw rows have no customer ID and are discarded.** Every figure in this report describes the identified-customer segment, not the business.

**Strict vs non-strict inequality.** All three implementations use strict `<` (and `>` for the ASOF condition). Mutation 1 showed this is untestable on this dataset because no purchase lands at midnight. It would matter immediately on a system with batch timestamps.

**Would a random split grouped by customer fix B?** It would fix one leak, not both. Grouping by customer removes *customer overlap* — the model memorising a specific customer's tendency to buy by seeing their other rows in training. It does **not** remove *temporal overlap*: the model would still see September 2011 rows while being tested on March 2011 rows, and would still learn seasonality directly rather than having to extrapolate it. Given the base-rate range from 0.15 (January) to 0.33 (November), that is substantial free information. Only a temporal split removes both.

---

## 11. Open questions

1. **Why is A only 0.8429?** Both participants expected leakage to be more devastating. The population-saturation hypothesis was tested and falsified (§9). No replacement explanation.

2. **Is B→C genuinely zero, or masked by the base-rate difference?** Testing this properly requires equalising base rates between the split types, e.g. by subsampling.

3. **Feature importances are split counts, not gain.** §6.2's conclusion that A and C lean on the same features needs re-running with `importance_type="gain"` before it can be trusted.

4. **The 365-day population barely bit.** A 180-day window would be a real intervention.

5. **Calibration is unmeasured.** §7.5 argues AUC is structurally blind to the main effect of training/serving skew. Brier score and calibration curves would likely show D as far more broken than any number here suggests.

6. **Return behaviour as features.** Whether `n_returns` / `return_rate` lift C meaningfully above E is the most interesting unanswered modelling question, and a better stretch goal than those in the brief.

---

## 12. Reproducing

```bash
python -m src.ingest      # xlsx -> parquet (once)
python -m src.build       # orders, cutoffs, entities, labels
python -m src.features    # pit, naive, asof feature tables
pytest                    # 9 tests
python -m src.train       # experiments -> results.csv
```

**Note on the two populations.** `src/build.py` currently contains the 365-day `entities` definition. The primary results (§6.1) were produced with the unrestricted version — remove the `AND o.invoice_ts >= c.cutoff - INTERVAL 365 DAY` line to reproduce them. The primary run's artefacts are preserved as `lab_alltime.duckdb` and `results_alltime.csv`.

This is a reproducibility defect: the population should be a command-line flag or config value rather than an edited line, so both runs come from one command. Unfixed.

---
