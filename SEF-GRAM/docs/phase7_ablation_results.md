# Phase 7: Component Ablations

| Mode                        |   Decoder CE |   Success Rate (%) |   Query Pixel Acc (%) |   Query Pixel Err |   Support Best Err |
|:----------------------------|-------------:|-------------------:|----------------------:|------------------:|-------------------:|
| Full SEF-GRAM               |         1.60 |               0.00 |                 66.09 |            305.22 |             827.67 |
| w/o EFLA                    |         1.72 |               0.00 |                 70.72 |            263.56 |             827.67 |
| w/o TTT Planner             |         1.60 |               0.00 |                 70.49 |            265.56 |             827.67 |
| w/o Support Verifier        |         1.60 |               0.00 |                 66.09 |            305.22 |             900.00 |
| w/o Task Encoder warm-start |         1.60 |               0.00 |                 70.07 |            269.33 |             827.67 |
| w/o Decoder TTFT            |         1.60 |               0.00 |                 70.67 |            264.00 |             827.67 |
