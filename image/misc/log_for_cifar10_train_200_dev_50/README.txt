The results for hyperparameter tuning with 200 training images and 50 dev images (it is actually 48 dev images when evaluating with TPUs, since TPUs require batch_size to be a multiple of 8) are shown in 1.txt and 2.txt.

The run is the number of experiments we run to compute the mean and std for a set of hyperparameter. We filter experiments with less than 10 runs.

In the experiment for 1.txt, we first random sampled train_steps of 100000 or 50000, tsa (as train_prob_threshold_anneal) of log_schedule, linear_schedule, exp_schedule (as log_linear_begin, linear, log_linear_end in), ent_min of 0, 0.1, 0.3, weight_decay_rate of 5e-4, 1e-3, unsup_coeff of 1 and 3. 

We found that train_prob_threshold_anneal=log_linear_begin, ent_min=0.1 and unsup_coeff=3 are better. So we fixed train_prob_threshold_anneal, ent_min and try to increase unsup_coeff.

Then, in the experiment for 2.txt, we random sampled train_steps of 100000 or 50000, unsup_ratio of 30, 40, weight_decay_rate of 5e-4, 7e-4, 1e-3, uda_softmax_temp (as unsup_temp) of 0.9, -1, uda_confidence_thresh (as unsup_high_thresh) of 0.8, -1, unsup_coeff of 6, 3. 

We take the top performing set of hyperparameter except that we lower the weight_decay_rate from 0.001 to 0.0007 since they lead to similar performance for 200 training images. For 250 images, reducing regularization might lead to better performance. 

In the experiment for 3.txt, we run experiments with best hyperparameters for 10 times and get the test performance.
