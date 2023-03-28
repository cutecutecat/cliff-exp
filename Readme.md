# Experiment for cliff

`cliff` is a improve algorithm with lower system error of mutation dataset when there is a **sequence defect** in fitness space.

This repository  used to carry out experiment of **`cliff` algorithm：

![General badge](https://img.shields.io/badge/source-cliff-blue?&link=https://github.com/cutecutecat/cliff&logo=Github)

**For comparison：**`extrema` and `RS`algorithm, and generation of dataset, are revised from [@acmater](https://github.com/acmater) repo and thanks to original paper [repo](https://github.com/yoavram/UnderTheRug):

![General badge](https://img.shields.io/badge/source-NK_Benchmarking-blue?&link=https://github.com/acmater&logo=Github)

# Experiment

| **file**            | **description**                                                          |
| ------------------------- | ------------------------------------------------------------------------------ |
| **step_0_gen_data** | **generate dataset** `Correct` , `Error` and `Time`                 |
| **step_1_correct**  | **compare** `cliff` correctness with `extrema` and `RS` algorithm     |
| **step_2_error**    | **compare** `cliff` system error  with `extrema` and `RS` algorithm   |
| **step_3_time**     | **compare** `cliff` time complexity with `extrema` and `RS` algorithm |

# Reference

**Obolski, Uri, Yoav Ram, and Lilach Hadany. "Key issues review: evolution on rugged adaptive landscapes." ***Reports on Progress in Physics* 81.1 (2017): 012602.
