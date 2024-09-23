

        Recall: Epoch-75 0.7479  0.8847  0.9462  0.9797 => proxy_fixed_bynumpy
        Epoch-0 0.5268  Epoch-11 0.7072  Epoch-43 0.7409  Epoch-95 0.7513  Epoch-189 0.7532 => train-2020-03-28-16-57-00

        V1. proxy_save expect achieving 76.0 => Epoch-241 0.7605  0.8942 => proxy_load => again => Epoch-55 0.7517 (bs16)
        Epoch-0 0.5203 Epoch-1 0.5753 Epoch-13 0.7038 Epoch-55 0.7402  Epoch-173 0.7421 => train-2020-04-02-10-20-02
        Epoch: [150]    Time 0.41       Loss 8.603      Accuracy 0.184  proxP 0.567     proxN 0.001     realP 0.634     realN 0.002     drift 0.000

        V2.proxy_save_updateloss_before expect achieving 73.0 => Recall: Epoch-199 0.7328  => proxy_load=> again => Epoch-131 0.7450 (bs150)
        Epoch-0 0.5071 Epoch-1 0.5605 Epoch-19 0.7022 Epoch-153 0.7341
        Epoch: [150]    Time 0.43       Loss 9.495      Accuracy 0.049  proxP 0.499     proxN 0.003     realP 0.632     realN 0.002     drift 0.000
        
        bs16 num4 beta0.35 (loss * 1.0, proxy_bs16) 
        Epoch-0 0.2542 Epoch-1 0.5270 Epoch-9 0.7135 Epoch-31 0.7417 Epoch-51 0.7456 Epoch-85 0.7478 => train-2020-04-05-20-39-54
        Epoch: [150]    Time 0.44       Loss 0.340      Accuracy 0.183  proxP 0.550     proxN 0.002     realP 0.662     realN 0.014     drift 0.550
        => load ( bs16 num4 beta0.35 loss * 1.0)
        Epoch-0 0.5925 Epoch-1 0.6516 Epoch-5 0.7087 Epoch-15 0.7335 => train-2020-04-10-19-00-30
        Epoch: [150]    Time 0.16       Loss 0.281      Accuracy 0.187  proxP 0.573     proxN 0.001     realP 0.647     realN 0.008    

        bs16 num4 beta0.35 (loss * 30.0, proxy_bs16_loss30) => train-2020-04-10-18-51-04
        Epoch-0 0.3073 Epoch-1 0.5154 Epoch-9 0.7015  Epoch-27 0.7426  Epoch-37 0.7500  Epoch-77 0.7605 Epoch-115 0.7651 
        Epoch: [116]    Time 0.42       Loss 9.352      Accuracy 0.248  proxP 0.589     proxN 0.001     realP 0.675     realN 0.005     drift 0.573
        => load (bs16 num4 beta0.35 loss * 30.0) =>  train-2020-04-12-18-18-42
        Epoch-0 0.5791 Epoch-1 0.6408 Epoch-3 0.6872 Epoch-5 0.7028 Epoch-7 0.7122 Epoch-9 0.7211  Epoch-15 0.7321 Epoch-25 0.7409 Epoch-55 0.7431 Epoch-149 0.7320 
        Epoch: [150]    Time 0.15       Loss 7.550      Accuracy 0.268  proxP 0.612     proxN 0.001     realP 0.667     realN 0.003     drift 0.000
        => load (bs16 num4 beta0.35 loss * 30.0 only pos pair) => train-2020-04-12-21-11-20 Interesting !!!
        Epoch-0 0.5307 Epoch-1 0.5928 Epoch-3 0.6565 Epoch-9 0.7085 Epoch-11 0.7156 Epoch-15 0.7236 Epoch-23 0.7306 Epoch-53 0.7381 Epoch-149 0.7288 
        Epoch: [054]    Time 0.15       Loss -7.437     Accuracy 0.124  proxP 0.624     proxN 0.003     realP 0.738     realN 0.008     drift 0.000

        bs16 num4 beta0.35 (loss * 30.0, proxy_bs16updatebeforeloss_loss30) 
        Epoch-0 0.6124 Epoch-1 0.6305 Epoch-25 0.7033 Epoch-55 0.7326 Epoch-101 0.7404 (很震荡)
        Epoch: [102]    Time 0.53       Loss 8.203      Accuracy 0.358  proxP 0.743     proxN 0.002     realP 0.743     realN 0.204     drift 0.586
        => load (bs16 num4 beta0.35 loss * 30.0)
        Epoch-0 0.5758 Epoch-1 0.6326  Epoch-3 0.6686  Epoch-5 0.6860 Epoch-9 0.7048 Epoch-11 0.7103  Epoch-17 0.7221 Epoch-43 0.7291
        Epoch: [150]    Time 0.15       Loss 8.378      Accuracy 0.070  proxP 0.549     proxN 0.000     realP 0.666     realN 0.003     drift 0.000

        bs16 num4 beta0.2 (loss * 1.0) 
        Epoch-0 0.3157  Epoch-1 0.5346 Epoch-7 0.7031 Epoch-21 0.7411 Epoch-51 0.7537  => train-2020-04-03-17-36-09
        recall75: Epoch: [052]    Time 0.10       Loss 0.265      Accuracy 0.346  proxP 0.571     proxN 0.013     realP 0.673     realN 0.024     drift 0.566

        bs16 num4 beta0.35 (loss * 1.0) 
        Epoch-0 0.2935 Epoch-1 0.5465 Epoch-9 0.7144 Epoch-25 0.7402   Epoch-59 0.7456  Epoch-77 0.7466 => train-2020-04-05-20-59-56
        Epoch: [150]    Time 0.17       Loss 0.343      Accuracy 0.186  proxP 0.549     proxN 0.002     realP 0.662     realN 0.014     drift 0.549
        
        bs16 num4 beta0.35 (loss = loss * 1.0 + 1000.0)
        Epoch-0 0.2974 Epoch-1 0.5421 Epoch-9 0.7121 Epoch-25 0.7408 Epoch-45 0.7509 => train-2020-04-05-20-56-30
        Epoch: [150]    Time 0.17       Loss 1000.341   Accuracy 0.187  proxP 0.549     proxN 0.002     realP 0.662     realN 0.013     drift 0.550

        bs16 num4 beta0.35 (loss * 15.0) 
        Epoch-0 0.3071 Epoch-1 0.5443 Epoch-9 0.7052 Epoch-25 0.7413 Epoch-33 0.7500 Epoch-69 0.7610 Epoch-91 0.7661 => train-2020-04-10-22-15-48
        Epoch: [150]    Time 0.12       Loss 4.521      Accuracy 0.259  proxP 0.599     proxN 0.001     realP 0.682     realN 0.004     drift 0.581

        bs16 num4 beta0.35 (loss * 30.0) => 在另外一个vs里, 已做完实验
        Epoch-0 0.2871 Epoch-1 0.5239 Epoch-9 0.7007 Epoch-27 0.7438 Epoch-35 0.7517 Epoch-63 0.7608 Epoch-83 0.7650   Epoch-111 0.7673 =>  train-2020-04-01-12-07-06
        Epoch: [150]    Time 0.12       Loss 9.157      Accuracy 0.254  proxP 0.593     proxN 0.001     realP 0.679     realN 0.004     drift 0.577

        bs16 num4 beta0.35 (loss * 60.0)
        Epoch-0 0.2795 Epoch-1 0.5333 Epoch-9 0.7026 Epoch-27 0.7414 Epoch-37 0.7502 Epoch-65 0.7602 Epoch-91 0.7631 => train-2020-04-10-22-15-17
        Epoch: [150]    Time 0.12       Loss 18.491     Accuracy 0.247  proxP 0.589     proxN 0.001     realP 0.676     realN 0.004     drift 0.573

        bs16 num4 beta0.35 (loss * 0.5) => train-2020-04-03-17-52-11 
        Epoch-0 0.3108 Epoch-1 0.5450 Epoch-7 0.7009 Epoch-51 0.7315 => toobad interesting
        Epoch: [052]    Time 0.09       Loss 0.185      Accuracy 0.149  proxP 0.525     proxN 0.004     realP 0.647     realN 0.020     drift 0.529
        

        every_step_freshbank 
        
        train-2020-04-12-22-03-23 => SGD 5e-5 loss*500.0 bs16 num4 weight_decay 1e-4 (equivalent to lr 2.5e-2 wd 2e-7 loss * 1.0)
        Epoch-0 0.1859 Epoch-1 0.3345 Epoch-3 0.5501 Epoch-5 0.6264 Epoch-7 0.6557 Epoch-13 0.7007  Epoch-29 0.7302 Epoch-47 0.7410 (lr decrease) Epoch-71 0.7513 Epoch-95 0.7560
        Epoch: [150]    Time 0.19       Loss 146.888    Accuracy 0.278  proxP 0.604     proxN 0.001     realP 0.697     realN 0.006     drift 0.598

        # train-2020-04-13-15-47-40 => SGD 5e-5 loss*500.0 bs16 num4 fixed lr
        # Epoch-0 0.1538 Epoch-1 0.3782 Epoch-3 0.5655 Epoch-5 0.6299 Epoch-7 0.6587 Epoch-15 0.7067 Epoch-33 0.7310 Epoch-49 0.7424
        # Epoch: [150]    Time 0.12       Loss 143.829    Accuracy 0.248  proxP 0.600     proxN 0.001     realP 0.711     realN 0.011     drift 0.611

        train-2020-04-12-22-20-20 => SGD 5e-4 loss*30.0 bs16 num4 weight_decay 1e-4 (equivalent to lr 1.5e-2 wd 3e-6 loss * 1.0)
        Epoch-0 0.1662 Epoch-1 0.4363 Epoch-3 0.5911  Epoch-5 0.6534 Epoch-7 0.6749 Epoch-11 0.7007  Epoch-25 0.7304 Epoch-35 0.7411 Epoch-51 0.7485 Epoch-61 0.7515 Epoch-81 0.7540 (lr 1.6e-5) Epoch-89 0.7612 Epoch-99 0.7623
        Epoch: [150]    Time 0.13       Loss 8.959      Accuracy 0.269  proxP 0.599     proxN 0.001     realP 0.690     realN 0.006     drift 0.590

        train-2020-04-12-22-21-27 => bs16 num4 SGD 5.0e-03 loss * 30.0 weight_decay 1e-4 (equivalent to lr 1.5e-1 wd 3e-6 loss * 1.0) Epoch-0 0.0000
        => refresh bank once in first epoch => Epoch-0 0.0000 (Warnup importances 一开始feat不好，bs太小导致梯度噪声也大，lr也大也会导致开始就陷入局部不好的点，所以体现warmup的重要性)
        => refresh bank once in first epoch bs100 num5 => train-2020-04-14-22-53-35 => 
        Epoch-0 0.5443 Epoch-1 0.5933 Epoch-3 0.6507 Epoch-5 0.6745 Epoch-9 0.7015 Epoch-41 0.7431 Epoch-83 0.7493 
        Epoch: [150]    Time 0.50       Loss 0.282      Accuracy 0.253  proxP 0.613     proxN 0.001     realP 0.713     realN 0.012     drift 0.636

        train-2020-04-13-11-54-49 => ADAM bs16 num4 loss*500.0 weight_decay 1e-4
        Epoch-0 0.2840 Epoch-1 0.5279 Epoch-3 0.6257 Epoch-5 0.6690 Epoch-7 0.6883 Epoch-9 0.7012 Epoch-19 0.7322 Epoch-27 0.7427 Epoch-37 0.7504 Epoch-73 0.7602 Epoch-105 0.7642
        Epoch: [150]    Time 0.19       Loss 150.444    Accuracy 0.262  proxP 0.597     proxN 0.001     realP 0.681     realN 0.004     drift 0.580

        bs16 num4 beta0.35 (loss * 30.0) => 在另外一个vs里, 已做完实验
        Epoch-0 0.2871 Epoch-1 0.5239 Epoch-9 0.7007 Epoch-27 0.7438 Epoch-35 0.7517 Epoch-63 0.7608 Epoch-83 0.7650   Epoch-111 0.7673 =>  train-2020-04-01-12-07-06
        Epoch: [150]    Time 0.12       Loss 9.157      Accuracy 0.254  proxP 0.593     proxN 0.001     realP 0.679     realN 0.004     drift 0.577

        loss * 30.0 wd0.0 (refresh bank once in first epoch) =>  train-2020-04-14-23-13-36
        Epoch-0 0.5494 Epoch-1 0.6086 Epoch-3 0.6602 Epoch-5 0.6879 Epoch-7 0.7026 Epoch-25 0.7428  Epoch-39 0.7515 Epoch-75 0.7606 Epoch-125 0.7634 
        Epoch: [150]    Time 0.23       Loss 8.871      Accuracy 0.262  proxP 0.600     proxN 0.001     realP 0.695     realN 0.007     drift 0.595

        loss * 500.0 wd0.0  (refresh bank once in first epoch) =>  train-2020-04-14-23-17-11
        Epoch-0 0.5540  Epoch-1 0.6083 Epoch-3 0.6621 Epoch-9 0.7083 Epoch-27 0.7455 Epoch-39 0.7506 Epoch-65 0.7604  Epoch-109 0.7633 
        Epoch: [150]    Time 0.17       Loss 146.762    Accuracy 0.257  proxP 0.600     proxN 0.001     realP 0.695     realN 0.007     drift 0.595

        loss * 50000.0 wd0.0 (refresh bank once in first epoch) => train-2020-04-14-23-04-19
        Epoch-0 0.5537 Epoch-1 0.6053 Epoch-3 0.6597 Epoch-7 0.6998 Epoch-25 0.7429 Epoch-37 0.7500 Epoch-69 0.7601 Epoch-111 0.7635 
        Epoch: [150]    Time 0.13       Loss 14705.308  Accuracy 0.261  proxP 0.602     proxN 0.001     realP 0.695     realN 0.007     drift 0.595

        lr[3, 100] loss * 30.0 wd0.0 (refresh bank once in first epoch) => train-2020-04-18-17-50-16 => ?
        lr[6, 100] loss * 30.0 wd0.0 (refresh bank once in first epoch) => train-2020-04-18-17-52-08 => ?
        lr5e-3 loss * 1.0 wd0.0 (refresh bank once in first epoch) => train-2020-04-18-17-56-22 => ?
        lr5e-1 loss * 1.0 wd0.0 => train-2020-04-18-17-58-19 => Epoch-0 1.0000
        lr5e-3 loss * 100.0 wd0.0 =>  train-2020-04-18-18-30-35  => ?
        lr5e-3 loss * 1000000.0 wd0.0 =>  train-2020-04-18-18-33-34 => ?
        lr50.0  loss * 1.0 wd0.0 => train-2020-04-18-18-26-22 => Epoch-0 1.0000

        lr5e-5 loss * 1.0 wd0.0 (refresh bank once in first epoch) => train-2020-04-18-20-39-52 ?
        lr5e-5[3, 100] loss * 1.0 wd0.0 (refresh bank once in first epoch) => train-2020-04-18-20-38-45 ?
        loss *  500000000.0  lr 5e-5 => train-2020-04-18-19-11-49 ?

        load only pos + have grad
        load pos and neg + have grad