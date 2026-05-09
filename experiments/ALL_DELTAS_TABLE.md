# Gamma vs direct-backend deltas (auto)

Δ% < 0 means gamma is FASTER than direct.

| exp | dataset | scale | pattern | knob | gamma_s | direct_s | Δ% | gamma_rec | direct_rec |
|---|---|---|---|---|---|---|---|---|---|
| e17_churn_rate_sweep |  | 200K | cluster | ratio=1.5 | 329.2 | 2194.1 | -85.0% | 0.9638 | 0.9994 |
| e17_churn_rate_sweep |  | 200K | partial_reset | ratio=1.5 | 674.5 | 2470.9 | -72.7% | 0.9216 | 0.9982 |
| e17_churn_rate_sweep |  | 200K | random | ratio=1.5 | 279.1 | 991.9 | -71.9% | 1.0000 | 1.0000 |
| e17_churn_rate_sweep |  | 200K | cluster | ratio=2.0 | 178.9 | 631.4 | -71.7% | 0.9998 | 1.0000 |
| e17_churn_rate_sweep |  | 200K | random | ratio=2.0 | 213.7 | 727.8 | -70.6% | 1.0000 | 1.0000 |
| e17_churn_rate_sweep |  | 200K | partial_reset | ratio=3.0 | 273.9 | 887.7 | -69.1% | 0.9968 | 0.9978 |
| e17_churn_rate_sweep |  | 200K | partial_reset | ratio=2.0 | 392.0 | 1028.7 | -61.9% | 0.9250 | 0.9988 |
| e17_churn_rate_sweep |  | 200K | sequential | ratio=1.5 | 406.9 | 1002.1 | -59.4% | 0.9892 | 1.0000 |
| e17_churn_rate_sweep |  | 200K | random | ratio=3.0 | 256.2 | 569.6 | -55.0% | 1.0000 | 1.0000 |
| e17_churn_rate_sweep |  | 200K | sequential | ratio=2.0 | 344.4 | 722.3 | -52.3% | 0.9998 | 0.9998 |
| e14_optimized_buffer |  |  | cluster |  | 10.7 | 20.2 | -46.9% | 0.9992 | 0.9996 |
| e23_multi_seed_variance |  | 200K | cluster |  | 101.8 | 168.2 | -39.5% | 0.9990 | 0.9996 |
| e23_multi_seed_variance |  | 200K | random |  | 98.2 | 161.6 | -39.2% | 0.9980 | 0.9992 |
| e17_churn_rate_sweep |  | 200K | cluster | ratio=3.0 | 232.8 | 380.4 | -38.8% | 0.9998 | 0.9996 |
| e20_router_component_ablation | glove | 200K | cluster |  | 450.8 | 725.1 | -37.8% | 0.9530 | 0.9630 |
| e23_multi_seed_variance |  | 200K | random |  | 100.5 | 161.6 | -37.8% | 0.9988 | 0.9992 |
| e23_multi_seed_variance |  | 200K | cluster |  | 104.7 | 168.2 | -37.8% | 0.9996 | 0.9996 |
| e20_router_component_ablation | msong | 200K | sequential |  | 359.6 | 577.7 | -37.8% | 0.9990 | 0.9990 |
| e16_cross_dataset | random-m | 100K | cluster |  | 57.0 | 89.6 | -36.3% | 0.4994 | 0.3796 |
| e16_cross_dataset | random-m | 100K | cluster |  | 347.5 | 543.9 | -36.1% | 0.9568 | 0.9728 |
| e20_router_component_ablation | msong | 200K | cluster |  | 477.4 | 743.5 | -35.8% | 0.9995 | 0.9995 |
| e16_cross_dataset | glove | 200K | cluster |  | 40.7 | 62.9 | -35.4% | 0.7160 | 0.6485 |
| e23_multi_seed_variance |  | 200K | random |  | 98.2 | 150.9 | -34.9% | 0.9980 | 0.9988 |
| e23_multi_seed_variance |  | 200K | cluster |  | 101.8 | 155.9 | -34.7% | 0.9990 | 0.9990 |
| e20_router_component_ablation | msong | 200K | partial_reset |  | 301.2 | 451.9 | -33.3% | 0.9945 | 0.9970 |
| e23_multi_seed_variance |  | 200K | random |  | 100.5 | 150.9 | -33.3% | 0.9988 | 0.9988 |
| e23_multi_seed_variance |  | 200K | cluster |  | 104.7 | 155.9 | -32.9% | 0.9996 | 0.9990 |
| e16_cross_dataset | msong | 200K | cluster |  | 54.0 | 79.7 | -32.2% | 0.9005 | 0.8660 |
| e16_cross_dataset | glove | 200K | cluster |  | 436.1 | 643.1 | -32.2% | 0.9530 | 0.9630 |
| e23_multi_seed_variance |  | 200K | random |  | 110.6 | 161.6 | -31.6% | 0.9994 | 0.9992 |
| e20_router_component_ablation | glove | 200K | cluster |  | 496.6 | 725.1 | -31.5% | 0.9530 | 0.9630 |
| e20_router_component_ablation | msong | 200K | cluster |  | 509.3 | 743.5 | -31.5% | 0.9995 | 0.9995 |
| e23_multi_seed_variance |  | 200K | cluster |  | 115.3 | 168.2 | -31.4% | 0.9992 | 0.9996 |
| e17_churn_rate_sweep |  | 200K | sequential | ratio=3.0 | 372.6 | 529.9 | -29.7% | 0.7782 | 0.9998 |
| e16_cross_dataset | random-m | 100K | random |  | 68.5 | 97.3 | -29.6% | 0.5882 | 0.5410 |
| e15_delete_pattern_matrix |  | 200K | partial_reset |  | 54.1 | 76.8 | -29.6% | 0.9588 | 0.9454 |
| e20_router_component_ablation | glove | 200K | partial_reset |  | 227.6 | 322.7 | -29.5% | 0.9155 | 0.9230 |
| e20_router_component_ablation | msong | 200K | cluster |  | 524.3 | 743.5 | -29.5% | 0.9995 | 0.9995 |
| e20_router_component_ablation | glove | 200K | cluster |  | 511.8 | 725.1 | -29.4% | 0.9630 | 0.9630 |
| e20_router_component_ablation | msong | 200K | partial_reset |  | 319.6 | 451.9 | -29.3% | 0.9945 | 0.9970 |
| e20_router_component_ablation | glove | 200K | sequential |  | 321.4 | 454.0 | -29.2% | 0.9335 | 0.9335 |
| e16_cross_dataset | msong | 200K | cluster |  | 411.1 | 574.2 | -28.4% | 0.9995 | 0.9995 |
| e14_optimized_buffer |  |  | cluster |  | 17.0 | 23.5 | -27.7% | 0.7564 | 0.7092 |
| e23_multi_seed_variance |  | 200K | cluster |  | 101.8 | 140.7 | -27.6% | 0.9990 | 0.9992 |
| e15_delete_pattern_matrix |  | 1M | random |  | 1191.7 | 1645.9 | -27.6% | 0.9958 | 0.9965 |
| e14_optimized_buffer |  |  | random |  | 8.8 | 12.1 | -27.3% | 0.9996 | 0.9992 |
| e16_cross_dataset | glove | 200K | random |  | 303.6 | 415.8 | -27.0% | 0.9420 | 0.9430 |
| e23_multi_seed_variance |  | 200K | random |  | 110.6 | 150.9 | -26.7% | 0.9994 | 0.9988 |
| e15_delete_pattern_matrix |  | 200K | random |  | 159.6 | 217.6 | -26.7% | 0.9996 | 0.9990 |
| e23_multi_seed_variance |  | 200K | random |  | 98.2 | 133.5 | -26.5% | 0.9980 | 0.9992 |
| e23_multi_seed_variance |  | 200K | cluster |  | 115.3 | 155.9 | -26.1% | 0.9992 | 0.9990 |
| e23_multi_seed_variance |  | 200K | cluster |  | 104.7 | 140.7 | -25.6% | 0.9996 | 0.9992 |
| e23_multi_seed_variance |  | 200K | random |  | 100.5 | 133.5 | -24.7% | 0.9988 | 0.9992 |
| e14_optimized_buffer |  |  | random |  | 17.9 | 23.7 | -24.5% | 0.9268 | 0.9042 |
| e20_router_component_ablation | msong | 200K | partial_reset |  | 341.5 | 451.9 | -24.4% | 0.9970 | 0.9970 |
| e15_delete_pattern_matrix |  | 1M | partial_reset |  | 1044.0 | 1380.8 | -24.4% | 0.9968 | 0.9974 |
| e16_cross_dataset | glove | 200K | random |  | 56.3 | 74.4 | -24.3% | 0.7535 | 0.7240 |
| e16_cross_dataset | msong | 200K | partial_reset |  | 47.1 | 62.2 | -24.3% | 0.9765 | 0.9735 |
| e17_churn_rate_sweep |  | 200K | cluster | ratio=1.0 | 162.1 | 213.3 | -24.0% | 0.9992 | 0.9994 |
| e16_cross_dataset | msong | 200K | random |  | 69.9 | 91.9 | -23.9% | 0.9565 | 0.9420 |
| e23_multi_seed_variance |  | 200K | cluster |  | 101.8 | 133.5 | -23.7% | 0.9990 | 0.9994 |
| e16_cross_dataset | random-m | 100K | random |  | 228.4 | 299.2 | -23.6% | 0.9258 | 0.9376 |
| e20_router_component_ablation | msong | 200K | random |  | 412.4 | 539.4 | -23.5% | 0.9985 | 0.9990 |
| e23_multi_seed_variance |  | 200K | random |  | 123.6 | 161.6 | -23.5% | 0.9990 | 0.9992 |
| e16_cross_dataset | random-m | 100K | partial_reset |  | 68.4 | 89.4 | -23.5% | 0.6304 | 0.6016 |
| e16_cross_dataset | random-m | 100K | partial_reset |  | 171.5 | 223.8 | -23.4% | 0.8738 | 0.8766 |
| e20_router_component_ablation | glove | 200K | random |  | 306.0 | 398.1 | -23.2% | 0.9420 | 0.9430 |
| e15_delete_pattern_matrix |  | 200K | cluster |  | 186.4 | 241.6 | -22.8% | 0.9992 | 0.9994 |
| e15_delete_pattern_matrix |  | 200K | partial_reset |  | 149.2 | 192.8 | -22.6% | 0.9988 | 0.9994 |
| e23_multi_seed_variance |  | 200K | cluster |  | 130.3 | 168.2 | -22.5% | 0.9986 | 0.9996 |
| e20_router_component_ablation | msong | 200K | sequential |  | 447.8 | 577.7 | -22.5% | 0.9990 | 0.9990 |
| e15_delete_pattern_matrix |  | 200K | random |  | 53.8 | 69.4 | -22.4% | 0.9268 | 0.9042 |
| e20_router_component_ablation | glove | 200K | partial_reset |  | 250.5 | 322.7 | -22.4% | 0.9155 | 0.9230 |
| e23_multi_seed_variance |  | 200K | random |  | 98.2 | 126.5 | -22.4% | 0.9980 | 0.9996 |
| e20_router_component_ablation | glove | 200K | random |  | 310.4 | 398.1 | -22.0% | 0.9420 | 0.9430 |
| e23_multi_seed_variance |  | 200K | random |  | 98.2 | 125.9 | -22.0% | 0.9980 | 0.9990 |
| e15_delete_pattern_matrix |  | 200K | cluster |  | 52.9 | 67.8 | -22.0% | 0.9286 | 0.9104 |
| e20_router_component_ablation | msong | 200K | random |  | 420.9 | 539.4 | -22.0% | 0.9985 | 0.9990 |
| e17_churn_rate_sweep |  | 200K | random | ratio=1.0 | 161.1 | 206.2 | -21.9% | 0.9996 | 0.9990 |
| e23_multi_seed_variance |  | 200K | cluster |  | 101.8 | 129.9 | -21.6% | 0.9990 | 0.9994 |
| e23_multi_seed_variance |  | 200K | cluster |  | 104.7 | 133.5 | -21.6% | 0.9996 | 0.9994 |
| e20_router_component_ablation | sift | 200K | cluster |  | 187.3 | 237.5 | -21.1% | 0.9992 | 0.9994 |
| e15_delete_pattern_matrix |  | 1M | cluster |  | 542.7 | 686.5 | -20.9% | 0.9876 | 0.9892 |
| e17_churn_rate_sweep |  | 200K | partial_reset | ratio=1.0 | 130.5 | 164.6 | -20.7% | 0.9988 | 0.9994 |
| e20_router_component_ablation | sift | 200K | cluster |  | 188.7 | 237.5 | -20.6% | 0.9992 | 0.9994 |
| e16_cross_dataset | msong | 200K | random |  | 388.1 | 488.4 | -20.5% | 0.9985 | 0.9990 |
| e23_multi_seed_variance |  | 200K | random |  | 100.5 | 126.5 | -20.5% | 0.9988 | 0.9996 |
| e20_router_component_ablation | sift | 200K | partial_reset |  | 149.8 | 188.4 | -20.5% | 0.9988 | 0.9994 |
| e20_router_component_ablation | msong | 200K | random |  | 429.6 | 539.4 | -20.4% | 0.9990 | 0.9990 |
| e23_multi_seed_variance |  | 200K | random |  | 100.5 | 125.9 | -20.2% | 0.9988 | 0.9990 |
| e20_router_component_ablation | sift | 200K | partial_reset |  | 151.4 | 188.4 | -19.6% | 0.9988 | 0.9994 |
| e20_router_component_ablation | sift | 200K | random |  | 185.9 | 230.7 | -19.4% | 0.9996 | 0.9990 |
| e23_multi_seed_variance |  | 200K | cluster |  | 104.7 | 129.9 | -19.4% | 0.9996 | 0.9994 |
| e16_cross_dataset | glove | 200K | partial_reset |  | 190.2 | 235.5 | -19.2% | 0.9155 | 0.9230 |
| e20_router_component_ablation | glove | 200K | sequential |  | 367.2 | 454.0 | -19.1% | 0.9335 | 0.9335 |
| e16_cross_dataset | glove | 200K | partial_reset |  | 49.2 | 60.8 | -19.1% | 0.7840 | 0.7760 |
| e20_router_component_ablation | sift | 200K | random |  | 186.8 | 230.7 | -19.0% | 0.9996 | 0.9990 |
| e23_multi_seed_variance |  | 200K | random |  | 123.6 | 150.9 | -18.1% | 0.9990 | 0.9988 |
| e23_multi_seed_variance |  | 200K | cluster |  | 115.3 | 140.7 | -18.0% | 0.9992 | 0.9992 |
| e23_multi_seed_variance |  | 200K | random |  | 110.6 | 133.5 | -17.2% | 0.9994 | 0.9992 |
| e20_router_component_ablation | glove | 200K | partial_reset |  | 268.6 | 322.7 | -16.8% | 0.9230 | 0.9230 |
| e16_cross_dataset | msong | 200K | partial_reset |  | 228.2 | 273.9 | -16.7% | 0.9945 | 0.9970 |
| e23_multi_seed_variance |  | 200K | cluster |  | 130.3 | 155.9 | -16.4% | 0.9986 | 0.9990 |
| e15_delete_pattern_matrix |  | 1M | partial_reset |  | 359.1 | 426.9 | -15.9% | 0.8790 | 0.8691 |
| e15_delete_pattern_matrix |  | 1M | random |  | 332.2 | 394.6 | -15.8% | 0.8677 | 0.8512 |
| e23_multi_seed_variance |  | 200K | cluster |  | 142.2 | 168.2 | -15.4% | 0.9992 | 0.9996 |
| e20_router_component_ablation | glove | 200K | partial_reset |  | 274.3 | 322.7 | -15.0% | 0.9230 | 0.9230 |
| e23_multi_seed_variance |  | 200K | random |  | 138.9 | 161.6 | -14.1% | 0.9990 | 0.9992 |
| e23_multi_seed_variance |  | 200K | cluster |  | 115.3 | 133.5 | -13.6% | 0.9992 | 0.9994 |
| e20_router_component_ablation | msong | 200K | partial_reset |  | 392.5 | 451.9 | -13.1% | 0.9970 | 0.9970 |
| e23_multi_seed_variance |  | 200K | random |  | 110.6 | 126.5 | -12.6% | 0.9994 | 0.9996 |
| e23_multi_seed_variance |  | 200K | random |  | 110.6 | 125.9 | -12.2% | 0.9994 | 0.9990 |
| e20_router_component_ablation | msong | 200K | sequential |  | 509.9 | 577.7 | -11.7% | 0.9985 | 0.9990 |
| e22_cpp_tuned |  | 200K | sequential |  | 50.9 | 57.4 | -11.2% | 1.0000 | 0.9030 |
| e23_multi_seed_variance |  | 200K | cluster |  | 115.3 | 129.9 | -11.2% | 0.9992 | 0.9994 |
| e20_router_component_ablation | glove | 200K | random |  | 355.2 | 398.1 | -10.8% | 0.9430 | 0.9430 |
| e17_churn_rate_sweep |  | 200K | random | ratio=0.5 | 81.1 | 90.5 | -10.3% | 0.9940 | 0.9948 |
| e20_router_component_ablation | glove | 200K | cluster |  | 651.1 | 725.1 | -10.2% | 0.9630 | 0.9630 |
| e17_churn_rate_sweep |  | 200K | partial_reset | ratio=0.5 | 75.9 | 83.9 | -9.6% | 0.9932 | 0.9950 |
| e20_router_component_ablation | msong | 200K | random |  | 488.1 | 539.4 | -9.5% | 0.9990 | 0.9990 |
| e23_multi_seed_variance |  | 200K | cluster |  | 142.2 | 155.9 | -8.8% | 0.9992 | 0.9990 |
| e20_router_component_ablation | msong | 200K | cluster |  | 682.4 | 743.5 | -8.2% | 0.9995 | 0.9995 |
| e23_multi_seed_variance |  | 200K | random |  | 138.9 | 150.9 | -7.9% | 0.9990 | 0.9988 |
| e17_churn_rate_sweep |  | 200K | cluster | ratio=0.5 | 88.7 | 96.3 | -7.9% | 0.9968 | 0.9970 |
| e23_multi_seed_variance |  | 200K | random |  | 123.6 | 133.5 | -7.4% | 0.9990 | 0.9992 |
| e23_multi_seed_variance |  | 200K | cluster |  | 130.3 | 140.7 | -7.4% | 0.9986 | 0.9992 |
| e20_router_component_ablation | msong | 200K | partial_reset |  | 420.7 | 451.9 | -6.9% | 0.9950 | 0.9970 |
| e15_delete_pattern_matrix |  | 200K | sequential |  | 195.3 | 209.2 | -6.6% | 0.9990 | 0.9984 |
| e16_cross_dataset | random-m | 100K | sequential |  | 88.3 | 94.4 | -6.5% | 0.5538 | 0.5536 |
| e20_router_component_ablation | sift | 200K | sequential |  | 231.1 | 245.6 | -5.9% | 0.9984 | 0.9984 |
| e15_delete_pattern_matrix |  | 1M | cluster |  | 456.3 | 477.9 | -4.5% | 0.9705 | 0.9673 |
| e17_churn_rate_sweep |  | 200K | partial_reset | ratio=0.25 | 67.5 | 69.7 | -3.1% | 0.9932 | 0.9926 |
| e17_churn_rate_sweep |  | 200K | random | ratio=0.25 | 69.4 | 71.1 | -2.5% | 0.9946 | 0.9946 |
| e23_multi_seed_variance |  | 200K | cluster |  | 130.3 | 133.5 | -2.4% | 0.9986 | 0.9994 |
| e23_multi_seed_variance |  | 200K | random |  | 123.6 | 126.5 | -2.3% | 0.9990 | 0.9996 |
| e23_multi_seed_variance |  | 200K | random |  | 123.6 | 125.9 | -1.8% | 0.9990 | 0.9990 |
| e17_churn_rate_sweep |  | 200K | cluster | ratio=0.25 | 74.3 | 75.5 | -1.6% | 0.9952 | 0.9954 |
| e15_delete_pattern_matrix |  | 200K | sequential |  | 60.6 | 61.5 | -1.4% | 0.9040 | 0.9030 |
| e16_cross_dataset | glove | 200K | sequential |  | 77.3 | 77.8 | -0.6% | 0.7110 | 0.7185 |
| e20_router_component_ablation | sift | 200K | cluster |  | 236.2 | 237.5 | -0.6% | 0.9994 | 0.9994 |
| e20_router_component_ablation | glove | 200K | sequential |  | 452.3 | 454.0 | -0.4% | 0.9340 | 0.9335 |
| e20_router_component_ablation | sift | 200K | random |  | 230.1 | 230.7 | -0.3% | 0.9990 | 0.9990 |
| e14_optimized_buffer |  |  | sequential |  | 23.4 | 23.5 | -0.2% | 0.9040 | 0.9030 |
| e23_multi_seed_variance |  | 200K | cluster |  | 130.3 | 129.9 | +0.4% | 0.9986 | 0.9994 |
| e20_router_component_ablation | glove | 200K | random |  | 400.6 | 398.1 | +0.6% | 0.9430 | 0.9430 |
| e20_router_component_ablation | glove | 200K | partial_reset |  | 325.0 | 322.7 | +0.7% | 0.9130 | 0.9230 |
| e20_router_component_ablation | sift | 200K | partial_reset |  | 189.8 | 188.4 | +0.8% | 0.9994 | 0.9994 |
| e17_churn_rate_sweep |  | 200K | sequential | ratio=0.25 | 74.0 | 73.3 | +0.9% | 0.9928 | 0.9928 |
| e16_cross_dataset | msong | 200K | sequential |  | 100.8 | 99.7 | +1.1% | 0.9420 | 0.9420 |
| e23_multi_seed_variance |  | 200K | cluster |  | 142.2 | 140.7 | +1.1% | 0.9992 | 0.9992 |
| e17_churn_rate_sweep |  | 200K | sequential | ratio=0.5 | 92.7 | 91.3 | +1.5% | 0.9942 | 0.9948 |
| e20_router_component_ablation | sift | 200K | sequential |  | 250.9 | 245.6 | +2.1% | 0.9984 | 0.9984 |
| e16_cross_dataset | glove | 1M | sequential |  | 265.3 | 258.7 | +2.6% | 0.6635 | 0.6595 |
| e20_router_component_ablation | sift | 200K | cluster |  | 244.7 | 237.5 | +3.0% | 0.9994 | 0.9994 |
| e20_router_component_ablation | glove | 200K | random |  | 413.4 | 398.1 | +3.8% | 0.9380 | 0.9430 |
| e20_router_component_ablation | sift | 200K | random |  | 239.6 | 230.7 | +3.9% | 0.9990 | 0.9990 |
| e20_router_component_ablation | sift | 200K | partial_reset |  | 195.9 | 188.4 | +4.0% | 0.9994 | 0.9994 |
| e23_multi_seed_variance |  | 200K | random |  | 138.9 | 133.5 | +4.0% | 0.9990 | 0.9992 |
| e20_router_component_ablation | msong | 200K | cluster |  | 777.5 | 743.5 | +4.6% | 0.9995 | 0.9995 |
| e20_router_component_ablation | msong | 200K | random |  | 570.2 | 539.4 | +5.7% | 0.9990 | 0.9990 |
| e23_multi_seed_variance |  | 200K | cluster |  | 142.2 | 133.5 | +6.6% | 0.9992 | 0.9994 |
| e20_router_component_ablation | msong | 200K | sequential |  | 616.2 | 577.7 | +6.7% | 0.9985 | 0.9990 |
| e15_delete_pattern_matrix |  | 1M | sequential |  | 369.6 | 340.7 | +8.5% | 0.8280 | 0.8274 |
| e20_router_component_ablation | glove | 200K | cluster |  | 793.9 | 725.1 | +9.5% | 0.9505 | 0.9630 |
| e23_multi_seed_variance |  | 200K | cluster |  | 142.2 | 129.9 | +9.5% | 0.9992 | 0.9994 |
| e23_multi_seed_variance |  | 200K | random |  | 138.9 | 126.5 | +9.8% | 0.9990 | 0.9996 |
| e23_multi_seed_variance |  | 200K | random |  | 138.9 | 125.9 | +10.3% | 0.9990 | 0.9990 |
| e20_router_component_ablation | msong | 200K | sequential |  | 648.5 | 577.7 | +12.2% | 0.9985 | 0.9990 |
| e16_cross_dataset | glove | 200K | sequential |  | 507.4 | 447.8 | +13.3% | 0.9340 | 0.9335 |
| e16_cross_dataset | random-m | 100K | sequential |  | 321.6 | 282.6 | +13.8% | 0.9292 | 0.9352 |
| e20_router_component_ablation | glove | 200K | sequential |  | 517.6 | 454.0 | +14.0% | 0.9340 | 0.9335 |
| e17_churn_rate_sweep |  | 200K | sequential | ratio=1.0 | 266.5 | 232.4 | +14.7% | 0.9990 | 0.9984 |
| e20_router_component_ablation | glove | 200K | sequential |  | 527.3 | 454.0 | +16.2% | 0.9340 | 0.9335 |
| e20_router_component_ablation | sift | 200K | partial_reset |  | 222.7 | 188.4 | +18.2% | 0.9988 | 0.9994 |
| e16_cross_dataset | msong | 200K | sequential |  | 692.6 | 566.2 | +22.3% | 0.9985 | 0.9990 |
| e20_router_component_ablation | sift | 200K | sequential |  | 304.7 | 245.6 | +24.0% | 0.9990 | 0.9984 |
| e20_router_component_ablation | sift | 200K | cluster |  | 297.1 | 237.5 | +25.1% | 0.9992 | 0.9994 |
| e20_router_component_ablation | sift | 200K | random |  | 288.7 | 230.7 | +25.1% | 0.9996 | 0.9990 |
| e20_router_component_ablation | sift | 200K | sequential |  | 309.0 | 245.6 | +25.8% | 0.9990 | 0.9984 |
| e14_optimized_buffer |  |  | sequential |  | 15.5 | 12.3 | +25.9% | 0.9990 | 0.9984 |
| e20_router_component_ablation | sift | 200K | sequential |  | 310.4 | 245.6 | +26.3% | 0.9990 | 0.9984 |
| e15_delete_pattern_matrix |  | 1M | sequential |  | 2231.1 | 1602.6 | +39.2% | 0.9952 | 0.9954 |
| e22_cpp_tuned |  | 200K | random |  | 110.6 | 58.0 | +90.6% | 0.9996 | 0.9042 |
| e22_cpp_tuned |  | 200K | cluster |  | 116.4 | 58.4 | +99.2% | 0.9986 | 0.9104 |
| e22_cpp_tuned |  | 200K | partial_reset |  | 116.1 | 57.8 | +100.8% | 0.0020 | 0.9454 |
| e16_cross_dataset | msong | 1M | sequential |  | 4313.3 | 2096.6 | +105.7% | 0.9935 | 0.9945 |
| e22_cpp_tuned |  | 200K | sequential |  | 118.3 | 57.4 | +106.1% | 1.0000 | 0.9030 |
| e22_cpp_tuned |  | 200K | cluster |  | 123.3 | 58.4 | +111.2% | 0.9986 | 0.9104 |
| e22_cpp_tuned |  | 200K | random |  | 123.4 | 58.0 | +112.7% | 0.9996 | 0.9042 |
| e16_cross_dataset | glove | 1M | sequential |  | 3738.1 | 1739.4 | +114.9% | 0.8710 | 0.8795 |
| e22_cpp_tuned |  | 200K | partial_reset |  | 143.7 | 57.8 | +148.6% | 0.9946 | 0.9454 |
| e22_cpp_tuned |  | 200K | cluster |  | 163.4 | 58.4 | +179.7% | 0.9886 | 0.9104 |
| e22_cpp_tuned |  | 200K | random |  | 214.8 | 58.0 | +270.3% | 0.9814 | 0.9042 |
| e22_cpp_tuned |  | 200K | sequential |  | 212.5 | 57.4 | +270.4% | 0.9858 | 0.9030 |
| e22_cpp_tuned |  | 200K | partial_reset |  | 262.6 | 57.8 | +354.3% | 0.4820 | 0.9454 |
| e22_cpp_tuned |  | 200K | sequential |  | 288.0 | 57.4 | +401.8% | 0.6720 | 0.9030 |
| e22_cpp_tuned |  | 200K | cluster |  | 366.1 | 58.4 | +526.8% | 0.9986 | 0.9104 |
| e22_cpp_tuned |  | 200K | random |  | 367.8 | 58.0 | +533.9% | 0.9996 | 0.9042 |
| e22_cpp_tuned |  | 200K | partial_reset |  | 484.1 | 57.8 | +737.6% | 0.0060 | 0.9454 |
