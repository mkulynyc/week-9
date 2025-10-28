# Test Results

When you're ready, make sure to submit your code for review on Canvas!

Tests started at 2025-10-28 14:42:44.815995.

## 1.1) test_estimate_argument -- passed
**Description:**
`estimate` argument can't be anything other than "mean" or "median". This tests what happens when we write `GroupEstimate(estimate=...)` for something other than "mean".

**Output:**
```bash

```


## 3.1) test_predict_output_type -- failed
**Description:**
If `gm = GroupEstimate ...`, test that the output of `gm.predict` is a numpy array.

**Output:**
```bash

```
<details>
<summary>AttributeError</summary>

```python
'list' object has no attribute 'columns'
```

</details>

## 3.2) test_example_1 -- failed
**Description:**
Using ["loc_country", "roast", 'origin_2'] columns in [coffee data](https://raw.githubusercontent.com/leontoddjohnson/datasets/refs/heads/main/data/coffee_analysis.csv), test the first value of a prediction `y_pred[0]`, and an invalid combination as an input.

**Output:**
```bash

```
<details>
<summary>AttributeError</summary>

```python
'list' object has no attribute 'columns'
```

</details>

## 3.3) test_example_2 -- failed
**Description:**
Using ["loc_country", "roast"] in the [coffee data](https://raw.githubusercontent.com/leontoddjohnson/datasets/refs/heads/main/data/coffee_analysis.csv), test that invalid combinations output missing values.

**Output:**
```bash

```
<details>
<summary>AttributeError</summary>

```python
'list' object has no attribute 'columns'
```

</details>
<br>Tests stopped at 2025-10-28 14:42:45.162351.